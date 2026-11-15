"""Private, persistent workspace for outcomes submitted by the researcher."""

import hashlib
import hmac
import json
from collections import OrderedDict
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from http.cookies import CookieError, SimpleCookie
from math import isfinite
from pathlib import Path
import secrets
import sqlite3
from threading import Lock
from time import monotonic, time
from uuid import uuid4

from oligo_engine.reward.portable import CHEMISTRY, MODEL_SHA256

SESSION = "oligoal_bench"
SESSION_AGE = 7 * 24 * 3600
DRAFT_AGE = 1800
ATLAS_ENDPOINT = "atlas_inhibition"
ASSAYS = {
    "inhibition": {"% inhibition"},
    "dose_response": {"% inhibition"},
    "ic50": {"nM", "µM"},
    "neurotoxicity": {"FOB score"},
    "hepatotoxicity": {"U/L", "IU/L", "mg/dL", "mmol/L", "µmol/L", "g/L", "ratio"},
}
CONTEXT_FIELDS = (
    "cell_line", "species", "method", "dose_nm", "hours", "route", "detail"
)
SCHEMA = """
CREATE TABLE IF NOT EXISTS session (digest TEXT PRIMARY KEY, expires REAL NOT NULL);
CREATE TABLE IF NOT EXISTS project (
    id TEXT PRIMARY KEY, profile TEXT NOT NULL, context TEXT NOT NULL,
    budget INTEGER NOT NULL DEFAULT 50, created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batch (
    id TEXT PRIMARY KEY, project TEXT NOT NULL REFERENCES project(id),
    request TEXT NOT NULL, name TEXT NOT NULL, created TEXT NOT NULL,
    candidates TEXT NOT NULL, evidence TEXT NOT NULL, UNIQUE(project, request)
);
CREATE TABLE IF NOT EXISTS measurement (
    id TEXT PRIMARY KEY, project TEXT NOT NULL REFERENCES project(id),
    candidate TEXT NOT NULL, request TEXT NOT NULL, created TEXT NOT NULL,
    UNIQUE(project, request)
);
CREATE TABLE IF NOT EXISTS result (
    measurement TEXT NOT NULL REFERENCES measurement(id),
    revision INTEGER NOT NULL, payload TEXT NOT NULL, created TEXT NOT NULL,
    PRIMARY KEY(measurement, revision)
);
CREATE TABLE IF NOT EXISTS note (
    project TEXT NOT NULL REFERENCES project(id), revision INTEGER NOT NULL,
    body TEXT NOT NULL, created TEXT NOT NULL, PRIMARY KEY(project, revision)
);
"""


class BenchError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def obj(text: str) -> dict | list:
    return json.loads(text)


def dump(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def string(value: object, limit: int, *, required: bool = False) -> str:
    if not isinstance(value, str) or len(value) > limit or (required and not value.strip()):
        raise BenchError("A required field is missing or exceeds its length limit.")
    return value.strip()


def integer(value: object, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise BenchError(f"Enter a whole number between {low} and {high}.")
    return value


def number(value: object, *, positive: bool = False) -> float:
    if (
        type(value) not in (int, float)
        or not isfinite(value)
        or abs(value) > 1e15
        or (positive and value <= 0)
    ):
        raise BenchError("Enter a finite measurement or a positive assay condition.")
    return float(value)


def validated_result(data: dict) -> dict:
    kind = data.get("kind")
    unit = data.get("unit")
    if not isinstance(kind, str) or not isinstance(unit, str) or unit not in ASSAYS.get(kind, ()):
        raise BenchError("Choose a measurement type and a compatible unit.")
    state = data.get("status", "measured")
    if state not in ("measured", "failed"):
        raise BenchError("Choose a measured or failed result.")
    qualifier = data.get("qualifier", "exact")
    if qualifier not in ("exact", "lt", "gt") or (kind != "ic50" and qualifier != "exact"):
        raise BenchError("Use a censoring bound only for IC₅₀.")
    context_data = data.get("context")
    if not isinstance(context_data, dict):
        raise BenchError("Enter the assay conditions.")
    context = {
        key: string(context_data.get(key, ""), 250)
        for key in CONTEXT_FIELDS if key not in ("dose_nm", "hours")
    }
    for key in ("dose_nm", "hours"):
        raw = context_data.get(key)
        context[key] = None if raw is None else number(raw, positive=True)
    value = None
    points = []
    if state == "measured" and kind == "dose_response":
        raw_points = data.get("points")
        if not isinstance(raw_points, list) or not 2 <= len(raw_points) <= 80:
            raise BenchError("Enter 2–80 dose-response points.")
        for point in raw_points:
            if not isinstance(point, list) or len(point) != 2:
                raise BenchError("Each curve point needs a dose and inhibition.")
            points.append([number(point[0], positive=True), number(point[1])])
        if len({point[0] for point in points}) < 2:
            raise BenchError("A curve needs at least two distinct doses.")
        points.sort()
    elif state == "measured":
        value = number(data.get("value"), positive=kind == "ic50")
    if kind in ("neurotoxicity", "hepatotoxicity") and not context["detail"]:
        raise BenchError("Enter the biomarker or score definition in assay details.")
    notes = string(data.get("notes", ""), 3000, required=state == "failed")
    charge = data.get("charge", True)
    if type(charge) is not bool:
        raise BenchError("Specify whether this measurement uses one assay.")
    return {
        "kind": kind, "unit": unit, "status": state, "qualifier": qualifier,
        "context": context, "value": value, "points": points, "notes": notes,
        "charge": charge,
    }


def compatible(result: dict) -> bool:
    context = result["context"]
    return (
        result["kind"] == "inhibition"
        and result["status"] == "measured"
        and context["cell_line"].casefold() == "hepg2"
        and context["species"].casefold() == "human"
        and context["method"].casefold() == "electroporation"
        and context["dose_nm"] == 4500
        and context["hours"] == 24
    )


def adapted_scores(candidates: dict, records: list[dict]) -> tuple[dict, list]:
    """Fit a project-specific Bayesian linear residual around the frozen Atlas means."""
    training = [
        row for row in records
        if compatible(row["result"])
        and candidates[row["candidate"]]["predictions"].get(ATLAS_ENDPOINT) is not None
    ]
    prior = {
        key: candidate["predictions"][ATLAS_ENDPOINT]
        for key, candidate in candidates.items()
        if candidate["predictions"].get(ATLAS_ENDPOINT) is not None
    }
    if not training:
        return prior, []
    import numpy as np

    def features(ids):
        sequences = [candidates[key]["sequence"] for key in ids]
        symbols = np.array([list(sequence) for sequence in sequences])
        one_hot = (symbols[:, :, None] == np.array(list("ACGT"))).reshape(len(ids), 80)
        return np.column_stack((np.ones(len(ids)), one_hot))

    index = [row["candidate"] for row in training]
    x = features(index)
    y = np.array([row["result"]["value"] - prior[key] for row, key in zip(training, index)])
    # Fixed prior/noise ratio. The posterior mean is displayed as a prediction.
    weights = np.linalg.solve(x.T @ x + 10 * np.eye(x.shape[1]), x.T @ y)
    ids = list(prior)
    corrected = features(ids) @ weights
    return {key: prior[key] + float(delta) for key, delta in zip(ids, corrected)}, training


class Bench:
    def __init__(self, directory: Path, origins: list[str]):
        self.directory = directory
        self.origins = frozenset(origins)
        secret = directory / "access-key"
        if not self.origins:
            raise ValueError("Bench requires an allowed browser origin.")
        if (directory.stat().st_mode | secret.stat().st_mode) & 0o077:
            raise ValueError("Restrict the Bench directory to mode 0700 and the access-key to mode 0600.")
        self.key = secret.read_text().strip()
        if len(self.key) < 32:
            raise ValueError("Bench access key must have at least 32 characters.")
        self.database = directory / "workspace.sqlite3"
        self.drafts: OrderedDict = OrderedDict()
        self.lock = Lock()
        with self.db() as conn:
            conn.executescript(SCHEMA)
        self.database.chmod(0o600)

    @contextmanager
    def db(self, write: bool = False):
        with closing(sqlite3.connect(self.database, timeout=10)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            with conn:
                conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
                yield conn

    def draft(self, payload: dict) -> str:
        key = secrets.token_urlsafe(24)
        snapshot = {name: payload[name] for name in ("transcript", "chemistry", "summary", "endpoints", "rows", "context_id")}
        with self.lock:
            while self.drafts and (
                len(self.drafts) >= 128
                or next(iter(self.drafts.values()))[0] < monotonic()
            ):
                self.drafts.popitem(last=False)
            self.drafts[key] = (monotonic() + DRAFT_AGE, snapshot)
        return key

    def browser_origin(self, origin: str, content_type: str) -> None:
        if origin not in self.origins or content_type.split(";")[0].strip() != "application/json":
            raise BenchError("Bench requests must come from the workspace page.", 403)

    @staticmethod
    def cookie(token: str = "", age: int = 0) -> str:
        return f"{SESSION}={token}; Path=/api/bench; Max-Age={age}; Secure; HttpOnly; SameSite=Strict"

    def login(self, body: dict) -> str:
        key = string(body.get("key"), 200, required=True)
        if not hmac.compare_digest(key.encode(), self.key.encode()):
            raise BenchError("Access key did not match.", 401)
        token = secrets.token_urlsafe(32)
        with self.db(write=True) as conn:
            conn.execute("DELETE FROM session WHERE expires <= ?", (time(),))
            conn.execute("INSERT INTO session VALUES (?, ?)", (digest(token), time() + SESSION_AGE))
        return self.cookie(token, SESSION_AGE)

    def session(self, cookie: str) -> str:
        try:
            value = SimpleCookie(cookie).get(SESSION)
        except CookieError:
            value = None
        if value and len(value.value) < 100:
            key = digest(value.value)
            with self.db() as conn:
                if conn.execute("SELECT 1 FROM session WHERE digest=? AND expires>?", (key, time())).fetchone():
                    return key
        raise BenchError("Sign in to open Bench Workspace.", 401)

    @staticmethod
    def project(conn, project: str):
        row = conn.execute("SELECT * FROM project WHERE id=?", (project,)).fetchone()
        if row is None:
            raise BenchError("Project was not found.", 404)
        return row

    @staticmethod
    def batches(conn, key: str) -> list[dict]:
        rows = conn.execute("SELECT * FROM batch WHERE project=? ORDER BY rowid DESC", (key,))
        return [
            {**dict(row), "candidates": obj(row["candidates"]), "evidence": obj(row["evidence"])}
            for row in rows
        ]

    @staticmethod
    def measurements(conn, key: str) -> list[dict]:
        rows = conn.execute("""
            SELECT m.id, m.candidate, r.revision, r.payload, r.created
            FROM measurement m JOIN result r ON r.measurement=m.id
            WHERE m.project=? AND r.revision=(
                SELECT max(revision) FROM result WHERE measurement=m.id
            )
            ORDER BY m.rowid DESC
        """, (key,))
        return [
            {"id": row["id"], "candidate": row["candidate"], "revision": row["revision"],
             "created": row["created"], "result": obj(row["payload"])}
            for row in rows
        ]

    @staticmethod
    def candidates(batches: list[dict]) -> dict:
        return {
            candidate["candidate_id"]: candidate
            for batch in batches
            if batch["evidence"]["method"] == "import"
            for candidate in batch["candidates"]
        }

    @staticmethod
    def proposed(batches, measurements) -> set:
        tested = {m["candidate"] for m in measurements}
        return {
            c["candidate_id"] for batch in batches if batch["evidence"]["method"] != "import"
            for c in batch["candidates"]
        } - tested

    @staticmethod
    def notes(conn, key: str) -> dict:
        row = conn.execute(
            "SELECT revision, body, created FROM note WHERE project=? ORDER BY revision DESC LIMIT 1", (key,)
        ).fetchone()
        return dict(row) if row else {"revision": 0, "body": "", "created": None}

    def state(self, key: str) -> dict:
        with self.db() as conn:
            projects = [
                {"id": row["id"], **obj(row["context"])}
                for row in conn.execute("SELECT * FROM project ORDER BY rowid DESC")
            ]
            if not key and projects:
                key = projects[0]["id"]
            if not key:
                return {"projects": projects, "project": None}
            db_project = self.project(conn, key)
            batches, measurements = self.batches(conn, key), self.measurements(conn, key)
            pending = self.proposed(batches, measurements)
            spent = sum(m["result"]["charge"] for m in measurements)
            return {
                "projects": projects,
                "project": {"id": key, "budget": db_project["budget"], **obj(db_project["context"])},
                "batches": batches, "measurements": measurements,
                "pending": sorted(pending), "spent": spent, "notes": self.notes(conn, key),
            }

    @staticmethod
    def insert_batch(conn, key, request, candidates, evidence) -> dict:
        count = conn.execute("SELECT count(*) FROM batch WHERE project=?", (key,)).fetchone()[0]
        batch_id = uuid4().hex
        conn.execute(
            "INSERT INTO batch VALUES (?, ?, ?, ?, ?, ?, ?)",
            (batch_id, key, request, f"Batch {count + 1}", timestamp(), dump(candidates), dump(evidence)),
        )
        return {"project": key, "batch": batch_id}

    def receive(self, body: dict) -> dict:
        run = string(body.get("run"), 64, required=True)
        ids = body.get("candidates")
        if not isinstance(ids, list) or not 1 <= len(ids) <= 100 or any(not isinstance(i, str) or len(i) > 240 for i in ids):
            raise BenchError("Select at least one candidate from the design run.")
        request = "import:" + digest(dump([run, sorted(set(ids))]))
        with self.db(write=True) as conn:
            previous = conn.execute("SELECT project, id FROM batch WHERE request=?", (request,)).fetchone()
            if previous:
                return {"project": previous["project"], "batch": previous["id"]}
            with self.lock:
                saved = self.drafts.get(run)
            if not saved or saved[0] < monotonic():
                raise BenchError("The design run expired. Run the design again and resend it.", 410)
            snapshot = saved[1]
            chosen = set(ids)
            rows = [
                {**row, "design_rank": rank}
                for rank, row in enumerate(snapshot["rows"], 1) if row["candidate_id"] in chosen
            ]
            if len(rows) != len(chosen):
                raise BenchError("Some selected candidates were outside the design run.")
            key = digest(dump([snapshot["context_id"], snapshot["chemistry"], snapshot["summary"]["model_version"]]))
            conn.execute(
                "INSERT OR IGNORE INTO project (id, profile, context, created) VALUES (?, ?, ?, ?)",
                (key, snapshot["chemistry"], dump({
                    "transcript": snapshot["transcript"],
                    "chemistry": snapshot["chemistry"],
                    "assay": snapshot["summary"].get("assay"),
                    "model_version": snapshot["summary"]["model_version"],
                    "endpoints": snapshot["endpoints"],
                }), timestamp()),
            )
            current = self.measurements(conn, key)
            evidence = {
                "method": "import", "run": run,
                "design": snapshot["summary"], "transcript_context_id": snapshot["context_id"],
                "observations": [{"id": m["id"], "revision": m["revision"]} for m in current],
                "notes_revision": self.notes(conn, key)["revision"],
            }
            return self.insert_batch(conn, key, request, rows, evidence)

    def recommend(self, conn, project, body: dict, request: str) -> dict:
        key = project["id"]
        old = conn.execute("SELECT id FROM batch WHERE project=? AND request=?", (key, request)).fetchone()
        if old:
            return {"project": key, "batch": old["id"]}
        count = integer(body.get("count", 5), 1, 20)
        batches = self.batches(conn, key)
        current = self.measurements(conn, key)
        pool = self.candidates(batches)
        proposed = self.proposed(batches, current)
        tried = {m["candidate"] for m in current}
        spent = sum(m["result"]["charge"] for m in current)
        count = min(count, project["budget"] - spent - len(proposed))
        available = [candidate for key, candidate in pool.items() if key not in tried | proposed]
        if count <= 0:
            raise BenchError("There are no assay slots available within the current budget.")
        if not available:
            raise BenchError("Every imported candidate has a result or is already proposed. Send more candidates from Design.")
        context = obj(project["context"])
        atlas = project["profile"] == CHEMISTRY and context["model_version"] == MODEL_SHA256
        means, training = adapted_scores(pool, current) if atlas else ({}, [])
        if atlas:
            available.sort(key=lambda row: (-means[row["candidate_id"]], row["candidate_id"]))
        else:
            available.sort(key=lambda row: (row["design_rank"], row["candidate_id"]))
        selected = [
            {**row, "bench_prediction": means.get(row["candidate_id"])}
            for row in available[:count]
        ]
        evidence = {
            "method": "bayesian_atlas_v1" if training else ("atlas_prior" if atlas else "design_rank"),
            "observations": [{"id": m["id"], "revision": m["revision"]} for m in current],
            "compatible_observations": [{"id": m["id"], "revision": m["revision"]} for m in training],
            "notes_revision": self.notes(conn, project["id"])["revision"],
            "model_version": context["model_version"],
            "budget": project["budget"], "spent_before": spent, "proposed_before": len(proposed),
        }
        return self.insert_batch(conn, project["id"], request, selected, evidence)

    def save_result(self, conn, key: str, body: dict, request: str) -> dict:
        prior_id = body.get("measurement")
        result = validated_result(body)
        if prior_id:
            prior_id = string(prior_id, 40, required=True)
            previous = conn.execute("""
                SELECT r.revision, r.payload FROM measurement m JOIN result r ON r.measurement=m.id
                WHERE m.project=? AND m.id=? ORDER BY r.revision DESC LIMIT 1
            """, (key, prior_id)).fetchone()
            if previous is None:
                raise BenchError("Result was not found in this project.", 404)
            revision = integer(body.get("revision"), 1, 1_000_000)
            if previous["revision"] != revision:
                raise BenchError("This result changed. Reload it before correcting it.", 409)
            result["charge"] = obj(previous["payload"])["charge"]
        else:
            previous = conn.execute("SELECT id FROM measurement WHERE project=? AND request=?", (key, request)).fetchone()
            if previous:
                return {"measurement": previous["id"]}
            candidate = string(body.get("candidate"), 240, required=True)
            if candidate not in self.candidates(self.batches(conn, key)):
                raise BenchError("Candidate was not found in this project.", 404)
            prior_id = uuid4().hex
            conn.execute(
                "INSERT INTO measurement VALUES (?, ?, ?, ?, ?)",
                (prior_id, key, candidate, request, timestamp()),
            )
            revision = 0
        conn.execute(
            "INSERT INTO result VALUES (?, ?, ?, ?)", (prior_id, revision + 1, dump(result), timestamp())
        )
        return {"measurement": prior_id}

    def mutate(self, action: str, body: dict, session: str) -> dict:
        if action == "import":
            return self.receive(body)
        if action == "logout":
            with self.db(write=True) as conn:
                conn.execute("DELETE FROM session WHERE digest=?", (session,))
            return {"ok": True}
        key = string(body.get("project"), 64, required=True)
        with self.db(write=True) as conn:
            project = self.project(conn, key)
            if action == "recommend":
                request = "recommend:" + string(body.get("request"), 80, required=True)
                return self.recommend(conn, project, body, request)
            if action == "result":
                request = string(body.get("request"), 80, required=True)
                return self.save_result(conn, key, body, request)
            if action == "rename":
                name = string(body.get("name"), 80, required=True)
                batch = string(body.get("batch"), 40, required=True)
                updated = conn.execute("UPDATE batch SET name=? WHERE project=? AND id=?", (name, key, batch))
                if not updated.rowcount:
                    raise BenchError("Batch was not found in this project.", 404)
            elif action == "settings":
                budget = integer(body.get("budget"), 1, 100_000)
                notes = string(body.get("notes"), 12_000)
                current = self.notes(conn, key)
                if body.get("notes_revision") != current["revision"] or body.get("prior_budget") != project["budget"]:
                    raise BenchError("Project settings changed. Reload before saving.", 409)
                conn.execute("UPDATE project SET budget=? WHERE id=?", (budget, key))
                if notes != current["body"]:
                    conn.execute("INSERT INTO note VALUES (?, ?, ?, ?)", (key, current["revision"] + 1, notes, timestamp()))
            else:
                raise BenchError("Unknown workspace operation.", 404)
        return {"ok": True}

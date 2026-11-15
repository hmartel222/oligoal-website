"""Local ASO design workspace and JSON API."""

import argparse
import csv
import io
import json
import logging
import threading
from dataclasses import asdict
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from oligo_engine.data.catalog import Catalog
from oligo_engine.design import candidates as design
from oligo_engine.design.aso_space import (
    Action,
    ASODesignSpace,
    TargetContext,
    identity,
)
from oligo_engine.design.engine import ENDPOINT_SPECS, DesignEngine
from oligo_engine.design.structure import duplex_pdb, pymol_script
from oligo_engine.design.transcripts import SYMBOL, Transcript, TranscriptError, fetch
from oligo_engine.reward.oligogym import OligoGymInference
from oligo_engine.reward.portable import CHEMISTRY as MOE_20
from oligo_engine.reward.portable import PortableEfficacy
from oligo_engine.reward.scoring import (
    AssayProfile,
    EndpointManifest,
    FrozenEndpoint,
    FrozenScorer,
    ScoringSession,
)

STATIC = Path(__file__).resolve().parent / "static"
DEFAULT_POLICY = STATIC.parents[1] / "artifacts" / "aso-app-policy"
DEFAULT_ATLAS_MODEL = STATIC.parents[1] / "artifacts" / "supported-aso-v1" / "efficacy.json"
ENGINE = DesignEngine()
MODEL_LOCK = threading.Lock()
MOE = "ps_moe_gapmer"
OME = "ps_2ome_gapmer"
RADIUS = HORIZON = 8
REGIONS = 8
MAX_TRANSCRIPT = 100_000
MAX_BODY = 110_000
# OligoGym deployment models do not use user-supplied assay covariates.
# The session requires a fixed context; these sentinel values never reach a model.
APP_CONTEXT = AssayProfile("OligoGym", "assay-specific", "sequence-only", 1, 1)
DEPLOYMENT = "exploratory_oligogym_deployment"
ENDPOINTS = {
    "efficacy_aso": ("Knockdown", "% inhibition", "ASOptimizer GP"),
    "neurotox_moe": ("Acute neurotox", "mouse FOB", "MOE Neurotox GP · advisory"),
    "tlr7_inhibition": ("TLR7 response", "% of control", "Alharbi TLR7 RF"),
    "tlr8_potentiation": ("TLR8 response", "× control", "Alharbi TLR8 GP"),
    "ic50": ("Potency (IC₅₀)", "nM", None),
    "hepatotoxicity": ("Hepatotoxicity", None, None),
    "atlas_inhibition": ("Predicted knockdown", "% inhibition", "Atlas 20-mer Ridge"),
}
ENDPOINT_CHEMISTRIES = {
    **{endpoint: spec["chemistry"] for endpoint, spec in ENDPOINT_SPECS.items()},
    "atlas_inhibition": MOE_20,
}


@lru_cache(maxsize=1)
def deployment_scorer() -> FrozenScorer:
    """Freeze the existing exploratory fits for web use, separate from benchmarks."""
    catalog = Catalog()
    heads = []
    for key in ("efficacy_aso", "neurotox_moe"):
        fitted = ENGINE.model(key)
        spec = ENDPOINT_SPECS[key]
        training_file = catalog.resolve(spec["dataset"])["data_file"]
        training_hash = catalog.manifest["files"][training_file]["sha256"]
        objective = key == "efficacy_aso"
        config = fitted.model.config
        upstream = {k: v for k, v in config["upstream"].items() if k != "oligogym_path"}
        fit_identity = identity(
            {**config, "upstream": upstream, "dataset": training_hash, "spec": spec}
        )
        heads.append(
            FrozenEndpoint(
                EndpointManifest(
                    key,
                    key + "_deployment",
                    ENDPOINTS[key][1],
                    MOE,
                    fit_identity,
                    training_hash,
                    "deployment_all_available_rows",
                    identity(spec),
                    APP_CONTEXT,
                    "objective" if objective else "advisory",
                    tuple(fitted.label_stats["values"]) if objective else (),
                    direction="higher_is_better" if objective else "lower_is_better",
                    provenance=DEPLOYMENT,
                ),
                OligoGymInference(fitted, key),
            )
        )
    return FrozenScorer(tuple(heads), DEPLOYMENT)


def target_context(transcript: Transcript) -> TargetContext:
    sequence = transcript.sequence.upper().replace("U", "T")
    if not 19 <= len(sequence) <= MAX_TRANSCRIPT:
        raise ValueError(f"Transcript must contain 19–{MAX_TRANSCRIPT:,} bases.")
    return TargetContext(
        transcript.gene,
        transcript.transcript_id,
        sequence,
        transcript.species,
        transcript.source,
    )


def resolve_transcript(gene: str, sequence: str = "") -> Transcript:
    if not SYMBOL.fullmatch(gene):
        raise ValueError("Enter a valid human gene symbol.")
    if not sequence.strip():
        return fetch(gene)
    lines = [line.strip() for line in sequence.splitlines() if line.strip()]
    if lines[0].startswith(">"):
        lines.pop(0)
    bases = "".join("".join(lines).split()).upper().replace("U", "T")
    if not 19 <= len(bases) <= MAX_TRANSCRIPT or set(bases) - set("ACGTN"):
        raise ValueError("Paste one transcript of 19–100,000 A/C/G/T/U/N bases.")
    return Transcript(
        gene, "pasted-" + identity(bases)[:10], bases, "supplied", "pasted"
    )


def design_spaces(target: TargetContext) -> list[ASODesignSpace]:
    sites = [
        pos
        for pos in range(len(target.sequence) - 18)
        if "N" not in target.sequence[pos : pos + 19]
    ]
    if not sites:
        raise ValueError("This transcript has no unambiguous 19-base windows.")
    if len(sites) <= 2 * RADIUS + 1:
        # Keep short, possibly disconnected regions reachable from at least one start.
        anchors = [sites[len(sites) // 2], *sites]
    else:
        anchors = [
            sites[((2 * i + 1) * len(sites)) // (2 * REGIONS)] for i in range(REGIONS)
        ]
    spaces, covered = [], set()
    for anchor in anchors:
        if anchor not in covered:
            space = ASODesignSpace(target, anchor, RADIUS, HORIZON)
            spaces.append(space)
            covered.update(space.reachable())
    return spaces


def design_panel(
    target: TargetContext, scorer: FrozenScorer, policy_path: Path | None
) -> tuple[list[dict], dict]:
    from oligo_engine.rl.environment import ASOEnv

    spaces = design_spaces(target)
    unique_budget = len({site for space in spaces for site in space.reachable()})
    session = ScoringSession(
        scorer, target, APP_CONTEXT, unique_budget, len(spaces) * 2 * (HORIZON + 2)
    )
    if policy_path is not None:
        from oligo_engine.rl.ppo import load_policy, rollout

        policy = load_policy(policy_path, ASOEnv(spaces[0], session))
        for space in spaces:
            rollout(policy, ASOEnv(space, session))
    else:
        # Traverse both legal paths from each anchor through the same Gym environment.
        for space in spaces:
            env = ASOEnv(space, session)
            for direction in (Action.LEFT, Action.RIGHT):
                env.reset()
                while True:
                    action = direction if env.action_masks()[direction] else Action.STOP
                    _, _, done, truncated, _ = env.step(action)
                    if done or truncated:
                        break
    rows = []
    for record in session.archive.values():
        row, score = record["candidate"], record["score"]
        if score["reward"] is not None:
            rows.append(
                {
                    **row,
                    "predictions": {
                        e["endpoint_id"]: e["raw"] for e in score["endpoints"]
                    },
                    "objective": score["reward"],
                }
            )
    rows.sort(
        key=lambda row: (
            -row["objective"],
            -row["predictions"]["efficacy_aso"],
            row["start"],
        )
    )
    return rows, {
        "method": "ppo" if policy_path else "design_space_search",
        "ranked_by": "Predicted knockdown",
        "regions": len(spaces),
        "scored": session.misses,
    }


def immune_panel(target: TargetContext) -> tuple[list[dict], dict]:
    pool = design.tile(
        target.gene,
        target.transcript_id,
        target.sequence,
        chemistry=OME,
        step=max(1, len(target.sequence) // 140),
    )
    if not pool:
        raise ValueError("This transcript has no unambiguous 20-base windows.")
    rows = ENGINE.score(pool)
    for row in rows:
        row["predictions"] = {
            key: prediction["value"] for key, prediction in row["predictions"].items()
        }
    rows.sort(
        key=lambda row: (abs(row["predictions"]["tlr8_potentiation"] - 1), row["start"])
    )
    return rows, {
        "method": "immune_panel",
        "ranked_by": "Distance from TLR8 assay control (1×)",
        "scored": len(rows),
    }


def atlas_panel(target: TargetContext, model: PortableEfficacy) -> tuple[list[dict], dict]:
    candidates = design.tile(
        target.gene, target.transcript_id, target.sequence, chemistry=MOE_20
    )
    if not candidates:
        raise ValueError("This transcript has no unambiguous 20-base windows.")
    ranked = [(model.predict(candidate), candidate) for candidate in candidates]
    ranked.sort(key=lambda item: (-item[0], item[1].start))
    rows = [
        {**vars(candidate), "predictions": {"atlas_inhibition": score}}
        for score, candidate in ranked
    ]
    return rows, {
        "method": "atlas_efficacy",
        "ranked_by": "Predicted knockdown",
        "scored": len(rows),
        "assay": asdict(model.assay),
    }


def panel(
    params: dict[str, str],
    policy: Path | None,
    atlas_model: PortableEfficacy | None = None,
    *,
    full: bool = False,
) -> dict:
    chemistry = params.get("chemistry", MOE)
    if chemistry not in design.DESIGNS:
        raise ValueError("Unsupported design chemistry.")
    if chemistry == MOE_20 and atlas_model is None:
        raise ValueError("The Atlas 20-mer efficacy checkpoint is unavailable.")
    limit = min(100, max(1, int(params.get("limit", "24"))))
    gene = params.get("gene", "").strip().upper()
    sequence = params.get("sequence", "")
    transcript = resolve_transcript(gene or ("PASTED" if sequence else ""), sequence)
    target = target_context(transcript)
    if chemistry == MOE_20 and atlas_model is not None:
        rows, summary = atlas_panel(target, atlas_model)
    else:
        with MODEL_LOCK:
            if chemistry == MOE:
                rows, summary = design_panel(target, deployment_scorer(), policy)
            else:
                rows, summary = immune_panel(target)
    return {
        "transcript": transcript.as_dict(),
        "chemistry": chemistry,
        "design": design.DESIGNS[chemistry],
        "summary": {**summary, "ranked_count": len(rows)},
        "rows": rows if full else rows[:limit],
        "endpoints": {
            key: {
                "title": title,
                "units": units,
                "model": model,
                "status": "unavailable"
                if model is None
                else (
                    "scored"
                    if ENDPOINT_CHEMISTRIES[key] == chemistry
                    else "not_applicable"
                ),
            }
            for key, (title, units, model) in ENDPOINTS.items()
        },
    }


def csv_panel(payload: dict) -> str:
    buffer = io.StringIO()
    endpoints = list(payload["endpoints"])
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "rank",
            "gene",
            "transcript",
            "sequence",
            "target_rna",
            "start_1_based",
            "end_1_based",
            "chemistry",
            "helm",
            "method",
            *endpoints,
        ]
    )
    for i, row in enumerate(payload["rows"], 1):
        transcript = payload["transcript"]
        writer.writerow(
            [
                i,
                transcript["gene"],
                transcript["transcript_id"],
                row["sequence"],
                row["target_window"].replace("T", "U"),
                row["start"] + 1,
                row["end"],
                row["chemistry"],
                row["helm"],
                payload["summary"]["method"],
                *(row["predictions"].get(key) for key in endpoints),
            ]
        )
    return buffer.getvalue()


def structure(params: dict[str, str]) -> tuple[str, str, str | None]:
    sequence = params.get("sequence", "").upper()
    target = params.get("target", "").upper().replace("T", "U")
    chemistry = params.get("chemistry", MOE)
    if chemistry not in design.DESIGNS:
        raise ValueError("Unsupported design chemistry.")
    template = design.DESIGNS[chemistry]
    if len(sequence) != template["length"] or set(sequence) - set("ACGT"):
        raise ValueError("ASO sequence does not match the design template.")
    target = target or design.reverse_complement(sequence).replace("T", "U")
    pdb = duplex_pdb(
        sequence,
        chemistry,
        template["wing5"],
        template["wing3"],
        "ASO-RNA duplex",
        target_window=target,
    )
    kind = params.get("format", "pdb")
    if kind == "pml":
        return (
            pymol_script(pdb, len(sequence), template["wing5"], template["wing3"]),
            ("text/plain; charset=utf-8"),
            "aso-duplex.pml",
        )
    if kind != "pdb":
        raise ValueError("Unknown structure format.")
    return pdb, "chemical/x-pdb", "aso-duplex.pdb" if "download" in params else None


class Handler(SimpleHTTPRequestHandler):
    policy: Path | None = None
    atlas_model: PortableEfficacy | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def _send(
        self,
        body: str,
        content_type: str = "application/json",
        status: int = 200,
        filename: str | None = None,
    ) -> None:
        encoded = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        if filename:
            self.send_header(
                "Content-Disposition", f'attachment; filename="{filename}"'
            )
        self.end_headers()
        self.wfile.write(encoded)

    def _api(self, path: str, params: dict[str, str]) -> None:
        try:
            if path == "/api/structure":
                body, content_type, filename = structure(params)
                self._send(body, content_type, filename=filename)
            elif path in ("/api/design", "/api/panel.csv"):
                full = path == "/api/panel.csv" and params.get("scope") == "full"
                payload = panel(params, self.policy, self.atlas_model, full=full)
                if path == "/api/panel.csv":
                    self._send(
                        csv_panel(payload),
                        "text/csv; charset=utf-8",
                        filename="aso-candidates-full.csv" if full else "aso-candidates.csv",
                    )
                else:
                    payload["csv"] = csv_panel(payload)
                    self._send(json.dumps(payload, allow_nan=False))
            elif path == "/api/meta":
                self._send(
                    json.dumps(
                        {
                            "chemistries": design.DESIGNS,
                            "ppo_configured": self.policy is not None,
                            "atlas_efficacy_configured": self.atlas_model is not None,
                        }
                    )
                )
            else:
                self._send('{"error":"Unknown endpoint."}', status=404)
        except (ValueError, TranscriptError) as error:
            self._send(json.dumps({"error": str(error)}), status=400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            logging.exception("Design API failed")
            self._send(
                '{"error":"The design runtime could not score this run. '
                'Check the local server log."}',
                status=503,
            )

    def do_GET(self) -> None:
        url = urlsplit(self.path)
        if url.path.startswith("/api/"):
            self._api(url.path, {k: v[-1] for k, v in parse_qs(url.query).items()})
        else:
            super().do_GET()

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path not in ("/api/design", "/api/panel.csv"):
            self._send('{"error":"Unknown endpoint."}', status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
            if not 0 < length <= MAX_BODY:
                raise ValueError("Missing or oversized request.")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict) or any(
                not isinstance(v, (str, int)) for v in body.values()
            ):
                raise ValueError("Expected a JSON object of design options.")
            params = {key: str(value) for key, value in body.items()}
        except (ValueError, UnicodeDecodeError) as error:
            self._send(json.dumps({"error": str(error)}), status=400)
            return
        self._api(path, params)


def train_policy(output: Path, training_gene: str, validation_gene: str) -> None:
    from oligo_engine.rl.environment import ASOEnv
    from oligo_engine.rl.ppo import train

    if output.exists():
        raise FileExistsError(output)
    scorer = deployment_scorer()
    envs = []
    for gene in (training_gene, validation_gene):
        target = target_context(resolve_transcript(gene.upper()))
        spaces = design_spaces(target)
        space = spaces[len(spaces) // 2]
        session = ScoringSession(
            scorer, target, APP_CONTEXT, len(space.reachable()), 8192
        )
        envs.append(ASOEnv(space, session))
    train(*envs, output, steps=1024)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-warm", action="store_true")
    parser.add_argument("--policy", type=Path, help="Trusted ASO policy directory")
    parser.add_argument("--atlas-efficacy", type=Path, help="Verified Atlas model JSON")
    parser.add_argument("--train-policy", type=Path, metavar="OUTPUT")
    parser.add_argument("--training-gene", default="HTT")
    parser.add_argument("--validation-gene", default="APP")
    args = parser.parse_args()
    if args.train_policy:
        train_policy(args.train_policy, args.training_gene, args.validation_gene)
        return
    Handler.policy = args.policy or (
        DEFAULT_POLICY if DEFAULT_POLICY.is_dir() else None
    )
    if Handler.policy and not all(
        (Handler.policy / name).is_file() for name in ("policy.zip", "manifest.json")
    ):
        parser.error("Expected policy.zip and manifest.json in the policy directory.")
    atlas_path = args.atlas_efficacy
    if atlas_path is None and DEFAULT_ATLAS_MODEL.exists():
        atlas_path = DEFAULT_ATLAS_MODEL
    if atlas_path is not None:
        try:
            Handler.atlas_model = PortableEfficacy(atlas_path)
        except (OSError, ValueError) as error:
            parser.error(str(error))
    if not args.no_warm:
        print("Fitting the OligoGym deployment models…", flush=True)
        deployment_scorer()
    with ThreadingHTTPServer((args.host, args.port), Handler) as server:
        print(f"OligoAL · http://{args.host}:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

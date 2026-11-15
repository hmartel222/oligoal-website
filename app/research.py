"""Loopback-only access to verified research model runtimes."""

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from threading import Lock

SOURCE_MODEL = re.compile(
    r"atlas-scaling-v1/source-(2869|4781)-(51[012])-"
    r"(?:(frozen|lora)_chemistry/best\.pt|(ridge|xgb)/model\.pkl)$"
)
LABELS = {"frozen": "RNA-FM frozen head", "lora": "RNA-FM LoRA",
          "ridge": "Atlas Ridge", "xgb": "Atlas XGBoost"}


class Research:
    def __init__(self, config_path: Path, origin: str):
        self.config = config = json.loads(config_path.read_text())
        self.checkpoints = Path(config["checkpoints"])
        self.origin = origin
        self.lock = Lock()
        manifest = json.loads((self.checkpoints / "manifest.json").read_text())
        self.catalog = [{"id": "ridge", "label": "PPO companion source Ridge"}]
        for entry in manifest["checkpoints"]:
            if match := SOURCE_MODEL.fullmatch(entry["path"]):
                split, seed, neural, classical = match.groups()
                self.catalog.append({
                    "id": entry["path"], "label": f"{LABELS[neural or classical]} · {split} · seed {seed}",
                })
        allowed = {entry["id"] for entry in self.catalog}
        self.models = frozenset(allowed)
        self.default = "atlas-scaling-v1/source-4781-510-lora_chemistry/best.pt"
        for key in ("policy_python", "atlas_python", "project_src", "atlas_src",
                    "rnafm_src", "rnafm_dependencies", "backbone"):
            if not Path(config[key]).exists():
                raise ValueError(f"Missing local model configuration path: {key}.")

    def request(self, action, candidates, assay, **details):
        if not candidates:
            raise ValueError("This model needs at least one candidate.")
        if not self.lock.acquire(blocking=False):
            raise ValueError("Another local model is running. Try again when it finishes.")
        try:
            atlas = action == "atlas"
            config = self.config
            paths = [config["atlas_src"], config["rnafm_src"], config["rnafm_dependencies"]] if atlas else [config["project_src"]]
            python = config["atlas_python" if atlas else "policy_python"]
            args = [python, "-B", str(Path(__file__).with_name("research_worker.py")), str(self.checkpoints)]
            if atlas:
                args += ["--backbone", config["backbone"], "--cache", config["cache"]]
            output = subprocess.run(
                args, text=True, capture_output=True, timeout=180, check=False,
                env={**os.environ, "PYTHONPATH": os.pathsep.join(paths), "OMP_NUM_THREADS": "2",
                     "OPENBLAS_NUM_THREADS": "1", "MPLCONFIGDIR": config["cache"]},
                input=json.dumps({
                    "action": action,
                    "candidates": [{k: row[k] for k in ("candidate_id", "helm")} for row in candidates],
                    "assay": assay, **details,
                }, allow_nan=False),
            )
            if output.stderr:
                logging.info("Research worker %s", output.stderr.strip()[-4000:])
            if output.returncode:
                raise ValueError("The local model stopped unexpectedly. Check the local server log.")
            response = json.loads(output.stdout.splitlines()[-1])
            if "error" in response:
                raise ValueError(response["error"])
            return response
        except subprocess.TimeoutExpired as error:
            raise ValueError("The local model took longer than three minutes.") from error
        finally:
            self.lock.release()

    def score(self, model, snapshot):
        from oligo_engine.reward.portable import CHEMISTRY

        if not isinstance(model, str) or model not in self.models:
            raise ValueError("Choose a local model from the menu.")
        if snapshot["chemistry"] != CHEMISTRY:
            raise ValueError("Local shortlist scoring requires the Atlas 20-mer MOE profile.")
        rows = snapshot["rows"]
        assay = snapshot["summary"]["assay"]
        response = self.request("ridge" if model == "ridge" else "atlas", rows, assay, model=model)
        return {
            "model": model, "model_version": response["model_version"],
            "scores": dict(zip((row["candidate_id"] for row in rows), response["scores"], strict=True)),
        }

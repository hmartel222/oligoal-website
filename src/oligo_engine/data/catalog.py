"""Integrity-checked access to the released, source-pinned OligoGym collection."""

import csv
import gzip
import hashlib
import json
from collections import Counter
from math import isfinite
from pathlib import Path
from random import Random

from oligo_engine.audit_upstream import PINNED_REVISION
from oligo_engine.contracts import Dataset, Observation, SplitManifest
from oligo_engine.data.oracle import LabelOracle
from oligo_engine.evaluation.metrics import HeldOutEvaluator

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "oligogym"
MISSING_TARGETS = {
    "",
    "nan",
    "none",
    "null",
    "na",
    "n/a",
    "unknown",
    "unk",
    "undefined",
}


class Catalog:
    def __init__(self, root: Path = DEFAULT_DATA_ROOT):
        self.root = root
        self.manifest = json.loads((root / "manifest.json").read_text())
        if self.manifest["upstream_revision"] != PINNED_REVISION:
            raise ValueError("Dataset manifest has an unsupported source revision")
        entries = self.manifest["datasets"]
        if len(entries) != 12 or len({r["name"] for r in entries}) != 12:
            raise ValueError("Expected the complete 12-dataset OligoGym collection")
        self.entries = tuple(entries)

    def resolve(self, key: str) -> dict:
        for entry in self.entries:
            if key.casefold() in {
                str(entry[k]).casefold() for k in ("key", "name", "benchmark_alias")
            }:
                return dict(entry)
        raise ValueError(f"Unknown dataset {key!r}; use --list-datasets")

    def verify_file(self, name: str) -> Path:
        if Path(name).name != name or name not in self.manifest["files"]:
            raise ValueError("File is outside the pinned data manifest")
        path = self.root / name
        expected = self.manifest["files"][name]
        if (
            path.stat().st_size != expected["bytes"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]
        ):
            raise ValueError(f"Dataset integrity check failed: {name}")
        return path

    def load(
        self, key: str, max_rows: int | None = None, selection_seed: int = 2026
    ) -> "LoadedDataset":
        entry = self.resolve(key)
        if max_rows is not None and (type(max_rows) is not int or max_rows < 4):
            raise ValueError("max_rows must be at least four or omitted for full data")
        metadata = json.loads(self.verify_file(entry["metadata_file"]).read_text())
        source = self.verify_file(entry["data_file"])
        digest = self.manifest["files"][entry["data_file"]]["sha256"]
        rng = Random(selection_seed)
        selected: list[tuple[int, dict]] = []
        count = 0
        with gzip.open(source, "rt", newline="") as file:
            reader = csv.DictReader(file)
            if not {"x", "y", "y_raw", "targets", "smiles", "fasta"} <= set(
                reader.fieldnames or []
            ):
                raise ValueError(f"Required columns missing from {source.name}")
            for index, row in enumerate(reader):
                count += 1
                if not row["x"] or not isfinite(float(row["y"])):
                    raise ValueError(f"Missing sequence/nonfinite label at row {index}")
                # Keep unused, often very large SMILES/FASTA in the source file;
                # avoid retaining them in the pool used by models and policies.
                row = {
                    k: row[k]
                    for k in ("x", "y", "targets", "cell_line", "concentration")
                    if k in row
                }
                # Reservoir inclusion depends on row position and RNG, never labels.
                if max_rows is None or len(selected) < max_rows:
                    selected.append((index, row))
                else:
                    replacement = rng.randrange(count)
                    if replacement < max_rows:
                        selected[replacement] = (index, row)
        if count != entry["rows"]:
            raise ValueError(f"Unexpected row count in {source.name}")
        selected.sort(key=lambda item: item[0])
        observations, labels = [], {}
        for index, row in selected:
            identity = f"{digest[:16]}:{entry['name']}:{index}"
            target = row["targets"].strip()
            if target.casefold() in MISSING_TARGETS:
                target = ""
            observations.append(
                Observation(
                    sample_id=identity,
                    sequence=row["x"],
                    target_id=target,
                    modality=metadata["modality"],
                    chemistry="encoded_in_source_HELM",
                    assay=metadata["label_desc"],
                    context=(("source_target_id", row["targets"]),)
                    + tuple(
                        (k, row[k]) for k in ("cell_line", "concentration") if k in row
                    ),
                )
            )
            labels[identity] = float(row["y"])
        dataset = Dataset(
            observations=tuple(observations),
            name=entry["name"],
            source_revision=PINNED_REVISION,
            content_sha256=digest,
            endpoint=metadata["label_desc"],
            provenance="measured",
            metadata={
                "key": entry["key"],
                "benchmark_alias": entry["benchmark_alias"],
                "modality": metadata["modality"],
                "collection": metadata["collection"],
                "source": metadata["source"],
                "source_rows": count,
                "selected_rows": len(selected),
                "requested_max_rows": max_rows,
                "selection_seed": selection_seed,
                "selection": "label_blind_reservoir",
                "model_system": metadata["model_system"],
                "caveats": entry["caveats"],
                "label_processing": "upstream y unchanged; y_raw retained in source",
                "endpoint_direction": "not_assumed",
            },
        )
        return LoadedDataset(dataset, labels)

    def validate_all(self) -> dict:
        for name in self.manifest["files"]:
            self.verify_file(name)
        reports = []
        for entry in self.entries:
            sequences: Counter[str] = Counter()
            targets = set()
            missing = count = 0
            with gzip.open(self.root / entry["data_file"], "rt", newline="") as file:
                reader = csv.DictReader(file)
                columns = list(reader.fieldnames or [])
                if not {"x", "y", "y_raw", "targets", "smiles", "fasta"} <= set(
                    columns
                ):
                    raise ValueError("Required source columns are missing")
                for index, row in enumerate(reader):
                    count += 1
                    if not row["x"] or not isfinite(float(row["y"])):
                        raise ValueError(
                            f"Missing sequence/nonfinite label at row {index}"
                        )
                    sequences[row["x"]] += 1
                    target = row["targets"].strip()
                    if target.casefold() in MISSING_TARGETS:
                        missing += 1
                    else:
                        targets.add(target)
            if count != entry["rows"]:
                raise ValueError(f"Unexpected row count in {entry['data_file']}")
            reports.append(
                {
                    "key": entry["key"],
                    "name": entry["name"],
                    "rows": count,
                    "unique_sequences": len(sequences),
                    "repeated_sequence_rows": count - len(sequences),
                    "known_targets": len(targets),
                    "missing_target_rows": missing,
                    "columns": columns,
                    "target_split_supported": missing == 0 and len(targets) > 1,
                    "caveats": entry["caveats"],
                }
            )
        return {
            "source_revision": PINNED_REVISION,
            "status": "validated",
            "datasets": reports,
            "total_rows": sum(r["rows"] for r in reports),
        }


class LoadedDataset:
    """Infrastructure-owned labels never travel with public dataset records."""

    def __init__(self, dataset: Dataset, labels: dict[str, float]):
        self.dataset = dataset
        self.__labels = dict(labels)

    def capabilities(
        self, split: SplitManifest
    ) -> tuple[LabelOracle, HeldOutEvaluator, LabelOracle]:
        split.validate(self.dataset)
        train = {i: self.__labels[i] for i in split.train_ids}
        return (
            LabelOracle(train, split.train_ids),
            HeldOutEvaluator(
                {i: self.__labels[i] for i in split.test_ids}, split.test_ids
            ),
            LabelOracle(train, split.train_ids),
        )

    def public_metadata(self) -> dict:
        return {k: v for k, v in vars(self.dataset).items() if k != "observations"}

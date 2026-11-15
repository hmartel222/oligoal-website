"""Deterministic toy data; never a biological dataset or experimental result."""

import hashlib
import json
from itertools import product
from random import Random

from oligo_engine.contracts import Dataset, Observation, SplitManifest
from oligo_engine.data.oracle import LabelOracle
from oligo_engine.evaluation.metrics import HeldOutEvaluator


def make_fixture() -> tuple[Dataset, SplitManifest, LabelOracle, HeldOutEvaluator]:
    rng = Random(2026)
    sequences = ["".join(bases) for bases in product("ACGU", repeat=4)]
    rng.shuffle(sequences)
    records = []
    labels = {}
    for index, sequence in enumerate(sequences[:48]):
        sample_id = f"synthetic:{index:03d}"
        features = tuple(float("ACGU".index(base)) / 3 for base in sequence)
        records.append(
            Observation(
                sample_id,
                sequence,
                f"toy-target-{index // 8}",
                "synthetic",
                "synthetic",
                "synthetic linear toy response",
                features,
            )
        )
        labels[sample_id] = (
            0.5 * features[0]
            + 0.3 * features[1]
            - 0.2 * features[2]
            + rng.uniform(-0.02, 0.02)
        )
    digest = hashlib.sha256(
        json.dumps(
            [(r.sample_id, r.sequence, labels[r.sample_id]) for r in records],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    dataset = Dataset(
        tuple(records),
        "synthetic-contract-fixture",
        "fixture-v1",
        digest,
        "synthetic_response",
        "synthetic",
    )
    train = tuple(row.sample_id for row in records[:36])
    test = tuple(row.sample_id for row in records[36:])
    split = SplitManifest("synthetic-fixed", 2026, train, test)
    return (
        dataset,
        split,
        LabelOracle({i: labels[i] for i in train}, train),
        HeldOutEvaluator({i: labels[i] for i in test}, test),
    )

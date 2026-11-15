"""Label-free diagnostic holdouts; never aliases for the official benchmark split."""

from collections import defaultdict
from math import ceil
from random import Random

from oligo_engine.contracts import Dataset, SplitManifest


class UnsupportedSplit(ValueError):
    pass


def make_split(
    dataset: Dataset, strategy: str, seed: int, test_size: float = 0.2
) -> SplitManifest:
    if strategy not in {"sequence_group", "target"}:
        raise UnsupportedSplit(
            "Supported local splits: sequence_group, target. "
            "Official nucleobase reproduction uses the separate reproduce command."
        )
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between zero and one")
    groups: dict[str, list[str]] = defaultdict(list)
    for row in dataset.observations:
        if strategy == "target" and not row.target_id:
            raise UnsupportedSplit("Target holdout requires known targets on every row")
        groups[row.sequence if strategy == "sequence_group" else row.target_id].append(
            row.sample_id
        )
    if len(groups) < 2:
        raise UnsupportedSplit("Holdout requires at least two distinct groups")
    ordered = sorted(groups)
    Random(seed).shuffle(ordered)
    ntest = ceil(test_size * len(ordered))
    if ntest == len(ordered):
        raise UnsupportedSplit("test_size leaves no training groups")
    test_ids = {identity for group in ordered[:ntest] for identity in groups[group]}
    manifest = SplitManifest(
        strategy=strategy,
        seed=seed,
        train_ids=tuple(
            r.sample_id for r in dataset.observations if r.sample_id not in test_ids
        ),
        test_ids=tuple(
            r.sample_id for r in dataset.observations if r.sample_id in test_ids
        ),
        parameters=(
            ("test_size_groups", str(test_size)),
            ("group_count", str(len(groups))),
            ("implementation", "local_group_holdout_v1"),
            ("benchmark_comparable", "false"),
        ),
    )
    try:
        manifest.validate(dataset)
    except ValueError as error:
        raise UnsupportedSplit(str(error)) from error
    return manifest

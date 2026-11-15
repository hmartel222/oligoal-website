"""Label-free repeat and partition audit saved with diagnostic runs."""

from collections import Counter

from oligo_engine.contracts import Dataset, SplitManifest


def audit_split(dataset: Dataset, split: SplitManifest) -> dict:
    """Count exact HELM repeats and actual target membership in each partition."""
    split.validate(dataset)
    train_ids = set(split.train_ids)
    partitions = {"train": [], "test": []}
    for row in dataset.observations:
        partitions["train" if row.sample_id in train_ids else "test"].append(row)

    results = {}
    for name, rows in partitions.items():
        sequences = Counter(row.sequence for row in rows)
        targets = Counter(row.target_id for row in rows if row.target_id)
        results[name] = {
            "rows": len(rows),
            "unique_helm": len(sequences),
            "repeated_helm_groups": sum(n > 1 for n in sequences.values()),
            "rows_in_repeated_helm_groups": sum(n for n in sequences.values() if n > 1),
            "repeated_helm_extra_rows": len(rows) - len(sequences),
            "known_target_groups": len(targets),
            "missing_target_rows": sum(not row.target_id for row in rows),
            "target_row_counts": dict(sorted(targets.items())),
        }

    train = partitions["train"]
    test = partitions["test"]
    return {
        "identity": "exact_HELM",
        "sequence_overlap_groups": len(
            {row.sequence for row in train} & {row.sequence for row in test}
        ),
        "target_overlap_groups": len(
            {row.target_id for row in train if row.target_id}
            & {row.target_id for row in test if row.target_id}
        ),
        "partitions": results,
    }

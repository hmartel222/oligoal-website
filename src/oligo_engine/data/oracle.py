"""Training-only retrospective label access, owned by evaluation infrastructure."""

from math import isfinite
from typing import Mapping


class LabelOracle:
    """Capability for a single campaign. Not a security sandbox for hostile code."""

    def __init__(self, labels: Mapping[str, float], eligible_ids: tuple[str, ...]):
        if len(eligible_ids) != len(set(eligible_ids)):
            raise ValueError("Duplicate eligible IDs")
        if set(labels) != set(eligible_ids):
            raise ValueError("Oracle must own exactly the eligible training labels")
        if any(not isfinite(value) for value in labels.values()):
            raise ValueError(
                "Labels must be finite measured or explicit fixture values"
            )
        self.__labels = dict(labels)
        self.eligible_ids = eligible_ids
        self.__revealed: dict[str, float] = {}

    @property
    def revealed_ids(self) -> tuple[str, ...]:
        return tuple(self.__revealed)

    def reveal(self, ids: tuple[str, ...]) -> tuple[float, ...]:
        # Validate the entire batch first: invalid requests cannot partially reveal.
        if len(ids) != len(set(ids)) or not ids:
            raise ValueError("Reveal requires a nonempty batch of unique IDs")
        if any(i not in self.__labels or i in self.__revealed for i in ids):
            raise ValueError(
                "Cannot reveal held-out, unknown or previously acquired IDs"
            )
        values = tuple(self.__labels[i] for i in ids)
        self.__revealed.update(zip(ids, values, strict=True))
        return values

    def acquired_labels(self, ids: tuple[str, ...]) -> tuple[float, ...]:
        if any(i not in self.__revealed for i in ids):
            raise ValueError("Unrevealed labels are inaccessible")
        return tuple(self.__revealed[i] for i in ids)

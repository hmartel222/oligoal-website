"""Metrics and a held-out capability which never returns evaluation labels."""

from math import isfinite, sqrt
from statistics import fmean
from typing import Mapping

from oligo_engine.contracts import Metrics, Prediction


def regression_metrics(truth: tuple[float, ...], means: tuple[float, ...]) -> Metrics:
    if not truth or len(truth) != len(means):
        raise ValueError("Truth and predictions must align and be nonempty")
    if any(not isfinite(x) for x in truth + means):
        raise ValueError("Metrics require finite values")
    errors = [a - b for a, b in zip(truth, means, strict=True)]
    mx, my = fmean(truth), fmean(means)
    dx = [x - mx for x in truth]
    dy = [y - my for y in means]
    denominator = sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    reason = None
    pearson = None
    if len(truth) < 2:
        reason = "fewer_than_two_rows"
    elif not denominator:
        reason = "constant_truth_or_prediction"
    else:
        pearson = max(
            -1.0,
            min(1.0, sum(x * y for x, y in zip(dx, dy, strict=True)) / denominator),
        )
    return Metrics(
        pearson,
        reason,
        sqrt(fmean(e * e for e in errors)),
        fmean(abs(e) for e in errors),
    )


class HeldOutEvaluator:
    def __init__(self, labels: Mapping[str, float], ids: tuple[str, ...]):
        if len(ids) != len(set(ids)) or set(labels) != set(ids) or not ids:
            raise ValueError("Evaluator must own exactly the held-out IDs")
        if any(not isfinite(x) for x in labels.values()):
            raise ValueError("Evaluation labels must be finite")
        self.__labels = dict(labels)
        self.ids = ids

    def score(self, predictions: tuple[Prediction, ...]) -> Metrics:
        return regression_metrics(
            tuple(self.__labels[i] for i in self.ids),
            tuple(p.mean for p in predictions),
        )

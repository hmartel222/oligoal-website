"""Score designed candidates with the frozen per-endpoint predictors.

This is a **deployment** path, not an evaluation path. Each endpoint model is refit
once, at startup, on every labelled row of its dataset; there is no held-out
partition here and no accuracy is measured here. The accuracy a user should believe
is the 30-seed held-out number from the matching study document, which is carried
through to the API response so the surface can show it next to every prediction.

Two rules are enforced rather than documented:

1. **Chemistry.** A predictor is only applied to candidates in the chemistry it was
   fitted on. Applying the 2'OMe-trained TLR models to a 2'-MOE candidate is the
   same error as scoring an MOE candidate with an LNA toxicity model, which this
   project already corrected once; here it raises instead.
2. **No combined score.** Endpoints are reported separately, in their own units.
   Efficacy is not modelled at all yet, and the response says so explicitly rather
   than letting a safety ranking read as a potency ranking.
"""

import csv
import gzip
from dataclasses import dataclass

from oligo_engine.data.catalog import Catalog
from oligo_engine.design.candidates import Candidate

# Each entry names the dataset, the chemistry it may be applied to, the arm that won
# that endpoint's 30-seed comparison, and the held-out accuracy to display.
ENDPOINT_SPECS = {
    "neurotox_moe": {
        "dataset": "MOE_Neurotox_1",
        "title": "Acute neurotoxicity",
        "detail": "3h functional observational battery score in mice",
        "units": "FOB score (0 safe to 7 toxic)",
        "chemistry": "ps_moe_gapmer",
        "model": "gp",
        "encoding": "chem",
        "max_sequence_length": 19,
        "base_annotation": "m5C",
        "direction": "higher_is_worse",
        "held_out_r": 0.597,
        "held_out_note": "GP, base+sugar+phosphate, 30 seeds, nucleobase split",
        "document": "docs/neurotox-moe-predictor.md",
    },
    "efficacy_aso": {
        "dataset": "Hwang_2024_1",
        "title": "Target knockdown",
        "detail": "percentage inhibition of target mRNA vs untreated control",
        "units": "% inhibition (higher is more knockdown)",
        "chemistry": "ps_moe_gapmer",
        "model": "gp",
        "encoding": "explicit",
        "alphabets": {
            "base": ("A", "C", "G", "T", "U"),
            "sugar": ("d", "moe"),
            "phosphate": ("p", "sp", "EMPTY"),
        },
        "max_sequence_length": 20,
        # ASOptimizer writes plain cytosine for this same chemistry.
        "base_annotation": "plain",
        # Fitted only on the rows whose chemistry matches this design space.
        "subset": {
            "sugars": ("d", "moe"),
            "phosphates": ("p", "sp", "EMPTY"),
            "bases": ("A", "C", "G", "T", "U"),
            "max_length": 20,
        },
        "direction": "higher_is_better",
        "held_out_r": 0.317,
        "held_out_note": "GP, base+sugar+phosphate, 30 seeds, nucleobase split",
        "document": "docs/aso-efficacy-predictor.md",
        "caveats": [
            "Fitted on 5 targets (DGAT2 dominant), so a new gene is extrapolation.",
            "Training lengths are 16, 17, 18 and 20; a 19-mer interpolates.",
            "Patent-sourced labels; expect selection effects.",
        ],
    },
    "tlr7_inhibition": {
        "dataset": "Alharbi_2020_2",
        "title": "TLR7 inhibition",
        "detail": "NF-kB reporter in HEK293-TLR7, relative to resiquimod = 100",
        "units": "percent of resiquimod (100 = unmodulated)",
        "chemistry": "ps_2ome_gapmer",
        "model": "rf",
        "encoding": "base",
        "max_sequence_length": 20,
        "direction": "deviation_from_baseline",
        "baseline": 100.0,
        "held_out_r": 0.598,
        "held_out_note": "RF, one-hot base, 30 seeds, nucleobase split",
        "document": "docs/tlr-immunomodulation-predictors.md",
    },
    "tlr8_potentiation": {
        "dataset": "Alharbi_2020_1",
        "title": "TLR8 potentiation",
        "detail": "NF-kB reporter in HEK293-TLR8, relative to resiquimod = 1.0",
        "units": "fold of resiquimod (1.0 = unmodulated)",
        "chemistry": "ps_2ome_gapmer",
        "model": "gp",
        "encoding": "base",
        "max_sequence_length": 20,
        "direction": "deviation_from_baseline",
        "baseline": 1.0,
        "held_out_r": 0.666,
        "held_out_note": "GP, one-hot base, 30 seeds, nucleobase split",
        "document": "docs/tlr-immunomodulation-predictors.md",
    },
}
MISSING_ENDPOINTS = {
    "efficacy_sirna": (
        "siRNA potency is modelled (Huesken, r 0.616) but deliberately not shown: "
        "it is a different modality and must not be applied to ASO candidates. ASO "
        "knockdown is modelled separately, on ASOptimizer."
    )
}


def _labels(dataset_name: str) -> list[float]:
    """Read the vendored labels in source order, after the integrity check."""
    catalog = Catalog()
    entry = catalog.resolve(dataset_name)
    path = catalog.verify_file(entry["data_file"])
    with gzip.open(path, "rt", newline="") as handle:
        return [float(row["y"]) for row in csv.DictReader(handle)]


@dataclass
class EndpointModel:
    key: str
    spec: dict
    model: object
    featurizer: object
    train_features: object
    label_stats: dict

    def applies_to(self, chemistry: str) -> bool:
        return chemistry == self.spec["chemistry"]

    def predict(self, candidates: list[Candidate]) -> list[dict]:
        import numpy as np

        off = [c for c in candidates if not self.applies_to(c.chemistry)]
        if off:
            raise ValueError(
                f"{self.key} was fitted on {self.spec['chemistry']}; refusing to "
                f"score {len(off)} candidate(s) in another chemistry"
            )
        annotation = self.spec.get("base_annotation")
        observations = tuple(c.observation() for c in candidates)
        if annotation:
            from dataclasses import replace

            from oligo_engine.design.candidates import annotate_bases

            observations = tuple(
                replace(o, sequence=annotate_bases(o.sequence, annotation))
                for o in observations
            )
        features = self.featurizer.transform(observations)
        predictions = self.model.predict(observations)
        from sklearn.metrics import pairwise_distances

        novelty = pairwise_distances(features, self.train_features).min(axis=1)
        labels = self.label_stats
        results = []
        for prediction, distance in zip(predictions, novelty, strict=True):
            value = float(prediction.mean)
            percentile = float((np.asarray(labels["values"]) <= value).mean() * 100.0)
            results.append(
                {
                    "endpoint": self.key,
                    "value": value,
                    "uncertainty": float(prediction.uncertainty),
                    "uncertainty_type": prediction.uncertainty_type,
                    "training_percentile": percentile,
                    "nearest_labelled_distance": float(distance),
                    "novelty_percentile": float(
                        (labels["self_distances"] <= distance).mean() * 100.0
                    ),
                }
            )
        return results


class DesignEngine:
    """Lazily fits one predictor per endpoint and scores candidate panels."""

    def __init__(self) -> None:
        self._models: dict[str, EndpointModel] = {}

    def endpoints_for(self, chemistry: str) -> list[str]:
        return [
            key
            for key, spec in ENDPOINT_SPECS.items()
            if spec["chemistry"] == chemistry
        ]

    def model(self, key: str) -> EndpointModel:
        if key in self._models:
            return self._models[key]
        import numpy as np
        from sklearn.metrics import pairwise_distances

        from oligo_engine.features.upstream import (
            BASE_ALPHABET,
            MODIFIED_BASE_ALPHABET,
            PHOSPHATE_ALPHABET,
            SUGAR_ALPHABET,
            OneHotBaseFeaturizer,
        )
        from oligo_engine.models.baseline import (
            OneHotGaussianProcess,
            OneHotRandomForest,
        )

        spec = ENDPOINT_SPECS[key]
        observations = Catalog().load(spec["dataset"]).dataset.observations
        labels = _labels(spec["dataset"])
        if len(labels) != len(observations):
            raise ValueError(f"{key}: label and observation counts differ")
        subset = spec.get("subset")
        if subset:
            from oligo_engine.design.candidates import matches_chemistry

            keep = [
                index
                for index, row in enumerate(observations)
                if matches_chemistry(
                    row.sequence,
                    subset["sugars"],
                    subset["phosphates"],
                    subset["bases"],
                    subset["max_length"],
                )
            ]
            if not keep:
                raise ValueError(f"{key}: no training rows match its chemistry")
            observations = tuple(observations[i] for i in keep)
            labels = [labels[i] for i in keep]
        if spec["encoding"] == "explicit":
            featurizer = OneHotBaseFeaturizer(
                spec["max_sequence_length"],
                component_alphabets={
                    component: tuple(symbols)
                    for component, symbols in spec["alphabets"].items()
                },
            )
        elif spec["encoding"] == "chem":
            featurizer = OneHotBaseFeaturizer(
                spec["max_sequence_length"],
                component_alphabets={
                    "base": MODIFIED_BASE_ALPHABET,
                    "sugar": SUGAR_ALPHABET,
                    "phosphate": PHOSPHATE_ALPHABET,
                },
            )
        else:
            featurizer = OneHotBaseFeaturizer(
                spec["max_sequence_length"], BASE_ALPHABET
            )
        factory = OneHotGaussianProcess if spec["model"] == "gp" else OneHotRandomForest
        # Seed 0 fixes the deployment fit; it is not an evaluation seed.
        model = factory(seed=0, featurizer=featurizer)
        model.fit(tuple(observations), tuple(labels))
        train_features = featurizer.transform(tuple(observations))
        # Distances within the labelled set give novelty a scale to be read against.
        within = pairwise_distances(train_features, train_features)
        np.fill_diagonal(within, np.inf)
        self._models[key] = EndpointModel(
            key=key,
            spec=spec,
            model=model,
            featurizer=featurizer,
            train_features=train_features,
            label_stats={
                "values": labels,
                "mean": float(np.mean(labels)),
                "sd": float(np.std(labels)),
                "self_distances": within.min(axis=1),
            },
        )
        return self._models[key]

    def score(self, candidates: list[Candidate]) -> list[dict]:
        """Attach every applicable endpoint's prediction to each candidate."""
        if not candidates:
            return []
        chemistries = {c.chemistry for c in candidates}
        if len(chemistries) != 1:
            raise ValueError("Score one chemistry at a time")
        (chemistry,) = chemistries
        scored = [
            {
                "candidate_id": c.candidate_id,
                "sequence": c.sequence,
                "helm": c.helm,
                "start": c.start,
                "end": c.end,
                "target_window": c.target_window,
                "chemistry": c.chemistry,
                "gc_fraction": c.gc_fraction,
                "cpg_count": c.cpg_count,
                "guanine_distance_from_3prime": c.guanine_distance_from_3prime,
                "longest_g_run": c.longest_g_run,
                "hairpin_dg": c.hairpin_dg,
                "predictions": {},
            }
            for c in candidates
        ]
        for key in self.endpoints_for(chemistry):
            for row, prediction in zip(
                scored, self.model(key).predict(candidates), strict=True
            ):
                row["predictions"][key] = prediction
        return scored

    def endpoint_metadata(self, chemistry: str) -> dict:
        applicable = self.endpoints_for(chemistry)
        return {
            "applicable": {k: ENDPOINT_SPECS[k] for k in applicable},
            "not_applicable": {
                k: {
                    **v,
                    "reason": (
                        f"fitted on {v['chemistry']}; applying it to "
                        f"{chemistry} candidates would extrapolate across chemistry"
                    ),
                }
                for k, v in ENDPOINT_SPECS.items()
                if k not in applicable
            },
            "not_modelled": MISSING_ENDPOINTS,
        }

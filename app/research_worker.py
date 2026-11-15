"""On-demand local inference in the interpreter pinned to each research model."""

import argparse
import hashlib
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PPO_FILES = {
    "policy.zip": "65e1bd78391e19909256c446ebfa1990a9e628b5099486686c9c4debbae3b517",
    "source_parameters.npz": "e8aa5a42e0eb479366c7ea9583d1a65210b2aa946c2e356446f039be0828b3c6",
    "source_prior.json": "168f78013a9aabb1fbfffb3efcc58c0454a43a76da52d1d8ca7ab339aab93a03",
}
SOURCE_PATH = re.compile(
    r"atlas-scaling-v1/source-(?:2869|4781)-51[012]-"
    r"(?:(?:frozen|lora)_chemistry/best\.pt|(?:ridge|xgb)/model\.pkl)$"
)


def verified(path, expected):
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != expected:
        raise ValueError(f"Checksum mismatch for {path.name}.")
    return path


def score_frame(rows, assay):
    return pd.DataFrame({"HELM Annotation": [row["helm"].replace("[m5C]", "[5meC]") for row in rows], **assay})


def policy(request, root):
    from gymnasium.utils.seeding import np_random
    from sb3_contrib import MaskablePPO

    from oligo_engine.experimental_rl.belief import encode
    from oligo_engine.experimental_rl.environment import (
        ExperimentalSelectionEnv,
        Expert,
        make_belief,
        sequence_vectors,
    )

    model_dir = root / "experimental-selection-ppo-3x32-v1"
    paths = {name: verified(model_dir / name, digest) for name, digest in PPO_FILES.items()}
    source = json.loads(paths["source_prior.json"].read_text())
    frame = score_frame(request["candidates"], request["assay"])
    x, chemistry, sequences = encode(frame, 32)
    with np.load(paths["source_parameters.npz"], allow_pickle=False) as parameters:
        prior = np.asarray(x @ parameters["coefficients"].reshape(-1) + parameters["intercept"]).reshape(-1)
    rows = request["candidates"]
    if request["action"] == "ridge":
        return {"scores": prior.tolist(), "model_version": source["model_identity"]}

    # Reuse the project's masked observation and acquisition experts. This adapter
    # has no label oracle. It reconstructs state exclusively from submitted results.
    env = ExperimentalSelectionEnv.__new__(ExperimentalSelectionEnv)
    env._ids = tuple(row["candidate_id"] for row in rows)
    env.assay_context = request["assay"]
    env._prior = prior
    env._chem = chemistry
    env._seq = sequence_vectors(sequences)
    env._remaining = np.ones(len(rows), dtype=bool)
    env._chem_gap = np.full(len(rows), np.inf)
    env._seq_gap = np.full(len(rows), np.inf)
    env.budget = request["budget"]
    env.threshold = 80.0
    env.version = "v1"
    env.allow_stop = False
    env.done = False
    seed = int(hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:8], 16)
    env.np_random = np_random(seed)[0]
    belief = make_belief(env.version, prior, chemistry, source["noise_assumption_std"])
    env.ledger = []
    by_id = {candidate: i for i, candidate in enumerate(env._ids)}
    for result in request["observed"]:
        i = by_id[result["candidate"]]
        value = result["value"]
        env.ledger.append({"observed": value})
        env._seq_gap = np.minimum(env._seq_gap, np.maximum(1 - env._seq @ env._seq[i], 0))
        env._chem_gap = np.minimum(env._chem_gap, np.maximum(1 - chemistry @ chemistry[i], 0))
        belief.update(i, value)
    for candidate in request["unavailable"]:
        env._remaining[by_id[candidate]] = False
    env._prediction = belief.predict()
    model = MaskablePPO.load(paths["policy.zip"], device="cpu")
    action, _ = model.predict(env._observation(), action_masks=env.action_masks(), deterministic=True)
    expert = Expert(int(action))
    index = env.propose(expert)
    return {
        "candidate_id": env._ids[index], "prediction": float(env._prediction.mean[index]),
        "expert": expert.name, "model_version": PPO_FILES["policy.zip"],
        "source_model_version": source["model_identity"],
    }


def atlas(request, root, backbone, cache):
    import torch
    from oligo_engine.atlas.baselines import CATEGORIES
    from oligo_engine.models.aso_rnafm import ASOPredictor, ChemistryHead

    model = request["model"]
    if not SOURCE_PATH.fullmatch(model):
        raise ValueError("Choose a supported source-trained Atlas model.")
    manifest = json.loads((root / "manifest.json").read_text())
    receipt = next(entry for entry in manifest["checkpoints"] if entry["path"] == model)
    path = verified(root / model, receipt["sha256"])
    frame = score_frame(request["candidates"], request["assay"])
    torch.set_num_threads(2)
    category_value = lambda name: str(request["assay"][name]).casefold()

    if model.endswith(".pkl"):
        with path.open("rb") as handle:
            predictor = pickle.load(handle)
        context = predictor.features.context
    else:
        adapt = "-lora_chemistry/" in model
        state = torch.load(path, weights_only=False, map_location="cpu")
        predictor = ASOPredictor(backbone, cache, 0, adapt)
        model_state = state["head"]
        width = model_state["output.0.weight"].shape[1] - 64
        predictor.head = ChemistryHead(width)
        predictor.restore(state)
        context = predictor.context.context

    encoder = context.named_transformers_["cat"]
    for category, observed_values in zip(CATEGORIES, encoder.categories_):
        matches = [value for value in observed_values if str(value).casefold() == category_value(category)]
        if not matches:
            raise ValueError(f"{category} is outside this checkpoint's fitted categories.")
        frame[category] = matches[0]
    scores = predictor.predict(frame)
    if not np.isfinite(scores).all():
        raise ValueError("The checkpoint returned a non-finite result.")
    return {"scores": scores.tolist(), "model_version": receipt["sha256"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    request = json.load(sys.stdin)
    try:
        if request["action"] == "atlas":
            output = atlas(request, args.root, args.backbone, args.cache)
        else:
            output = policy(request, args.root)
        print(json.dumps(output, allow_nan=False), flush=True)
    except (ValueError, KeyError, OSError) as error:
        print(json.dumps({"error": str(error)}), flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
        print(json.dumps({"error": "The local model could not run. Check the local server log."}), flush=True)


if __name__ == "__main__":
    main()

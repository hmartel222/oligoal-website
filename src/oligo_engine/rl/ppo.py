"""Optional pinned MaskablePPO training and identity-checked inference reload."""

import hashlib
import importlib.metadata
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from oligo_engine.rl.environment import ACTION_SCHEMA, OBSERVATION_SCHEMA, ASOEnv
from oligo_engine.rl.search import summarize

RUNTIME = {
    "gymnasium": "1.1.1",
    "stable-baselines3": "2.6.0",
    "sb3-contrib": "2.6.0",
    "torch": "2.3.1",
}


def verify_runtime() -> dict:
    versions = {key: importlib.metadata.version(key) for key in RUNTIME}
    if any(versions[key].split("+")[0] != val for key, val in RUNTIME.items()):
        raise RuntimeError(f"Unverified optional RL runtime: {versions}")
    return versions


def rollout(model: object, env: ASOEnv, seed: int = 0) -> dict:
    started = perf_counter()
    observation, _ = env.reset(seed=seed)
    done, actions, rewards = False, [], []
    while not done:
        action, _ = model.predict(
            observation, action_masks=env.action_masks(), deterministic=True
        )
        observation, reward, terminated, truncated, _ = env.step(int(action))
        actions.append(int(action))
        rewards.append(float(reward))
        done = terminated or truncated
    return {
        **summarize(env.session, "ppo", "truncated" if truncated else "complete"),
        "actions": actions,
        "terminal_position": env.position,
        "return": sum(rewards),
        "terminal_reward": rewards[-1],
        "wall_seconds": perf_counter() - started,
    }


def train(
    train_env: ASOEnv,
    validation_env: ASOEnv,
    output: Path,
    steps: int = 2048,
    seed: int = 7,
) -> tuple[object, dict]:
    """Select only on validation utility; exact training resume is not promised."""
    started = perf_counter()
    versions = verify_runtime()
    import torch
    from sb3_contrib import MaskablePPO

    if output.exists():
        raise FileExistsError(output)
    if steps < 64 or steps % 64:
        raise ValueError("Training steps must be a positive multiple of 64")
    if train_env.design_space.target.gene == validation_env.design_space.target.gene:
        raise ValueError("Policy train and validation targets must be distinct")
    if train_env.session.scorer.identity != validation_env.session.scorer.identity:
        raise ValueError("Training and validation scorers differ")
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        seed=seed,
        gamma=1.0,
        n_steps=64,
        batch_size=64,
        n_epochs=5,
        learning_rate=0.003,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [32, 32]},
        device="cpu",
        verbose=0,
    )
    best, history = -float("inf"), []
    training_wall, validation_wall = 0.0, 0.0
    interval = min(512, steps)
    completed = 0
    while completed < steps:
        block = min(interval, steps - completed)
        training_started = perf_counter()
        model.learn(block, reset_num_timesteps=completed == 0)
        training_wall += perf_counter() - training_started
        completed += block
        probe = rollout(model, validation_env, seed)
        validation_wall += probe["wall_seconds"]
        value = probe["terminal_reward"]
        history.append({"steps": completed, "validation_terminal_reward": value})
        if value > best:
            best = value
            model.save(output / "policy.zip")
    checkpoint = output / "policy.zip"
    manifest = {
        "schema": "aso-policy-v1",
        "runtime": versions,
        "seed": seed,
        "steps": steps,
        "action_schema": ACTION_SCHEMA,
        "observation_schema": OBSERVATION_SCHEMA,
        "observation_shape": train_env.observation_space.shape,
        "scorer_identity": train_env.session.scorer.identity,
        "assay": asdict(train_env.session.assay),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "training_space": asdict(train_env.design_space),
        "validation_space": asdict(validation_env.design_space),
        "horizon": train_env.design_space.horizon,
        "radius": train_env.design_space.radius,
        "training_cost": train_env.session.receipt(),
        "validation_cost": validation_env.session.receipt(),
        "history": history,
        "resume": "inference_only_no_exact_training_resume",
        "training_wall_seconds": training_wall,
        "validation_wall_seconds": validation_wall,
        "total_wall_seconds": perf_counter() - started,
        "hyperparameters": {
            "gamma": 1.0,
            "n_steps": 64,
            "batch_size": 64,
            "n_epochs": 5,
            "learning_rate": 0.003,
            "ent_coef": 0.01,
            "net_arch": [32, 32],
            "device": "cpu",
            "threads": 1,
            "deterministic_algorithms": True,
            "checkpoint_selection": "highest_validation_terminal_reward_first_tie",
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return load_policy(output, validation_env), manifest


def load_policy(output: Path, env: ASOEnv) -> object:
    verify_runtime()
    from sb3_contrib import MaskablePPO

    manifest = json.loads((output / "manifest.json").read_text())
    checkpoint = output / "policy.zip"
    if (
        hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        != manifest["checkpoint_sha256"]
    ):
        raise ValueError("Policy checkpoint hash mismatch")
    expected = (
        manifest["scorer_identity"] == env.session.scorer.identity
        and manifest["action_schema"] == list(ACTION_SCHEMA)
        and manifest["observation_schema"] == OBSERVATION_SCHEMA
        and manifest["observation_shape"] == list(env.observation_space.shape)
        and manifest["assay"] == asdict(env.session.assay)
        and manifest["horizon"] == env.design_space.horizon
        and manifest["radius"] == env.design_space.radius
    )
    if not expected:
        raise ValueError("Policy/scorer/space schema mismatch")
    return MaskablePPO.load(checkpoint, device="cpu")

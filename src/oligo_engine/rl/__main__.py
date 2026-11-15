"""Reproducible ASO fixture panel, search comparison and optional PPO smoke."""

import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path

from oligo_engine.rl.evaluate import PolicyPartitions, paired_summary
from oligo_engine.rl.fixture import fixture_session, fixture_space
from oligo_engine.rl.search import METHODS, search


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "score-panel", "compare"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ppo", action="store_true")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/rl/aso_fixture_v1.toml")
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--unique-budget", type=int)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    if config["schema"] != "aso-fixture-v1" or config["purpose"] != "synthetic_fixture":
        raise ValueError("This CLI accepts synthetic fixture profiles only")
    for key in ("steps", "seed", "unique_budget"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
        setattr(args, key, config[key])
    if (
        config["start"],
        config["radius"],
        config["horizon"],
        config["gamma"],
        config["request_budget"],
        config["shortlist_size"],
    ) != (1, 8, 8, 1.0, 100, 5):
        raise ValueError("Unsupported fixture schema settings")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    space = fixture_space("FINAL")
    partitions = PolicyPartitions(("TRAIN",), ("VALIDATION",), ("FINAL",))
    reference = search(space, fixture_session(space), "exhaustive", args.seed)
    runs = (
        [reference]
        if args.command == "score-panel"
        else [
            search(space, fixture_session(space, args.unique_budget), method, args.seed)
            for method in METHODS
            if method != "exhaustive"
        ]
    )
    extra = {}
    if args.ppo and args.command == "score-panel":
        raise ValueError("score-panel does not train a policy")
    if args.ppo:
        from oligo_engine.rl.environment import ASOEnv
        from oligo_engine.rl.ppo import load_policy, rollout, train

        training, validation = fixture_space("TRAIN"), fixture_space("VALIDATION")
        model, manifest = train(
            ASOEnv(training, fixture_session(training, max_requests=args.steps * 3)),
            ASOEnv(validation, fixture_session(validation)),
            args.output / "policy",
            args.steps,
            args.seed,
        )
        env = ASOEnv(space, fixture_session(space, args.unique_budget))
        first = rollout(model, env, args.seed)
        reloaded = load_policy(args.output / "policy", env)
        second = rollout(
            reloaded,
            ASOEnv(space, fixture_session(space, args.unique_budget)),
            args.seed,
        )
        parity_keys = ("actions", "terminal_position", "return", "archive")
        parity = all(first[k] == second[k] for k in parity_keys)
        if not parity:
            raise AssertionError("Reload probe differs")
        runs.append(first)
        extra = {"ppo_manifest": manifest, "reload_parity": parity}
    report = {
        **paired_summary(runs, reference),
        "runs": runs,
        "reference": reference,
        "partitions": partitions.__dict__,
        "seed": args.seed,
        "resolved_config": config,
        "python": sys.version,
        "command": sys.argv,
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "code_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "caveat": "Synthetic fixture; no measured ASO or biological validation",
        **extra,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: report[k] for k in ("status", "reference_optimum", "results")}, indent=2
        )
    )


if __name__ == "__main__":
    main()

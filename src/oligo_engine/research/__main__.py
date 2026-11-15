"""Inspect research tracks or create a label-free nested-budget plan."""

import argparse
import json
import tomllib
from pathlib import Path

from oligo_engine.research.budgets import nested_budget_plan


def load_catalog(path: Path) -> list[dict]:
    """Validate declared study metadata, without executing study commands."""
    data = tomllib.loads(path.read_text())
    if data.get("schema_version") != 1:
        raise ValueError("unsupported catalog schema")
    studies = data.get("studies", [])
    required = {"id", "track", "status", "question", "next_step", "evidence"}
    allowed = {"completed_development", "infrastructure_only", "planned"}
    seen = set()
    for study in studies:
        if set(study) != required:
            raise ValueError("study entries must use the declared catalog fields")
        if any(not isinstance(v, str) or not v.strip() for v in study.values()):
            raise ValueError("study fields must be nonempty strings")
        if study["id"] in seen or study["status"] not in allowed:
            raise ValueError("duplicate study or invalid status")
        seen.add(study["id"])
    if not studies:
        raise ValueError("catalog contains no studies")
    return studies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    catalog = commands.add_parser("list", help="show declared status, not live jobs")
    catalog.add_argument(
        "--catalog", type=Path, default=Path("configs/research/studies.toml")
    )
    plan = commands.add_parser("plan", help="freeze IDs; does not train models")
    plan.add_argument("--pool", type=Path, required=True)
    plan.add_argument("--budgets", type=int, nargs="+", required=True)
    plan.add_argument("--seed", type=int, required=True)
    plan.add_argument("--validation-fraction", type=float, default=0.2)
    plan.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "list":
            print(json.dumps(load_catalog(args.catalog), indent=2))
            return
        payload = json.loads(args.pool.read_text())
        if set(payload) != {"records", "excluded_ids"}:
            raise ValueError("pool needs exactly records and excluded_ids")
        if not isinstance(payload["records"], list) or not all(
            isinstance(row, dict) for row in payload["records"]
        ):
            raise ValueError("records must be a list of id/group objects")
        if not isinstance(payload["excluded_ids"], list) or not all(
            isinstance(i, str) and i for i in payload["excluded_ids"]
        ):
            raise ValueError("excluded_ids must be an explicit list of strings")
        result = nested_budget_plan(
            payload["records"],
            args.budgets,
            args.seed,
            args.validation_fraction,
            payload["excluded_ids"],
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
        print(f"Allocation saved to {args.output}; no training or evaluation ran.")
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

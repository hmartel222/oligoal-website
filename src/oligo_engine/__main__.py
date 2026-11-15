"""Run a synthetic fixture or a catalog-based measured diagnostic configuration."""

import argparse
import tomllib
from pathlib import Path

from oligo_engine.acquisition.random import RandomAcquisition
from oligo_engine.contracts import RunConfig
from oligo_engine.data.fixture import make_fixture
from oligo_engine.evaluation.runner import run_campaign
from oligo_engine.models.fixture import FixtureNearestNeighbor
from oligo_engine.reporting.artifacts import write_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/fixture.toml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    if config.get("dataset") != "synthetic-contract-fixture":
        from oligo_engine.collection import run_collection
        from oligo_engine.data.catalog import Catalog

        try:
            catalog = Catalog(args.data_root) if args.data_root else Catalog()
            result = run_collection(catalog, config, args.output)
        except (ValueError, OSError, KeyError) as error:
            parser.error(str(error))
        if result["status"] != "complete":
            parser.exit(1, f"Incomplete diagnostic suite; inspect {args.output}\n")
        print("Measured diagnostic suite complete; Gate 1 remains open.")
        return
    seeds = config["seeds"]
    if (
        not seeds
        or any(type(s) is not int for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        parser.error("Seeds must be distinct integers")
    runs = []
    try:
        for seed in seeds:
            dataset, split, oracle, evaluator = make_fixture()
            _, _, reference_oracle, _ = make_fixture()
            runs.append(
                run_campaign(
                    dataset,
                    split,
                    oracle,
                    evaluator,
                    FixtureNearestNeighbor,
                    RandomAcquisition(),
                    RunConfig(seed=seed, **config["budget"]),
                    reference_oracle,
                )
            )
        write_artifacts(tuple(runs), args.output)
    except ValueError as error:
        parser.error(str(error))
    print(f"Completed {len(runs)} SYNTHETIC fixture runs in {args.output}")
    print("Gate 1 remains open; these are not OligoGym results.")


if __name__ == "__main__":
    main()

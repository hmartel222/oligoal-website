# ASO environment: implementation and execution

November 14, 2026. The user authorized execution after reviewing the implementation
plan. The engineering environment, scorer boundary, search baselines and masked PPO
fixture now run locally. **A real constrained ASO optimization benchmark is pending
trusted model artifacts and profile validation.** No biological result is claimed.

## Delivered scope

| Plan package | Actual delivery | Remaining condition |
| --- | --- | --- |
| P0 | Optional pinned RL group, updated lock, fixture config, optional CI job and Python 3.11 installation | CI is configured, not remotely observed |
| P1 | Offline `ASODesignSpace`, STOP/LEFT/RIGHT, 19-mer MOE 5-10-4 construction, connected reachable region, recognition/chemistry tests | Real transcripts must arrive through a pinned manifest |
| P2 | Real-candidate frozen scorer, endpoint/model roles, bounded training-reference percentiles, raw screens, diagnostics, trusted artifact hashing, Atlas/OligoGym inference adapters | Actual fitted-bundle loading and owner prediction parity remain unverified |
| P3 | Gymnasium environment with fixed observation schema, masks, terminal rewards, atomic budget truncation and charged reset/intermediate scores | Real-profile coverage is not validated |
| P4 | Unchanged, random, greedy/local, integer genetic and exhaustive search | No measured optimizer superiority established |
| P5 | MaskablePPO training, validation-only checkpoint selection, saved-policy identity checks and deterministic inference reload | Exact training resume is not implemented |
| P6 | Paired scoring-cap checks, complete candidate archives, score ledgers/curves, explicit missingness, diversity and deferred exact-assay evaluation helper | Final measured enrichment and multi-target uncertainty need real data/protocol |
| P7 | Fixture panel and costed comparison executed | Real panel, validated screens and locked real benchmark not executed |

The implementation stays on the existing reward-adapter branch. This agent acts as
integrator; no other workers or worktrees were assigned. The existing mock adapter
and its tests remain intact. No Atlas model/study or preliminary-siRNA environment
was merged wholesale. Only the reviewed Atlas chemistry parser was selectively
ported, with source revision documented in its module.

## Runtime and reproducible commands

The isolated project `.venv` uses Python **3.11.16**, installed with Poetry **2.1.4**.
Direct RL pins match the inspected preliminary harness: Gymnasium **1.1.1**, SB3
**2.6.0**, sb3-contrib **2.6.0**, torch **2.3.1**. The optional science group retains
numpy **1.26.4**, scipy **1.15.3**, sklearn **1.4.0**. Lock resolution added RL
dependencies without changing any previously locked package version.

From the repository root, using Poetry 2.1.4 and a Python 3.11 interpreter:

```sh
poetry env use python3.11
make setup-rl
make lint format-check import-check test
poetry run pytest -q tests/test_aso_rl_optional.py

# Dependency-light finite search and panel paths:
poetry run python -m oligo_engine.rl score-panel --output artifacts/aso-panel
poetry run python -m oligo_engine.rl compare --output artifacts/aso-search

# Masked training, validation selection, final rollout and reload probe:
poetry run python -m oligo_engine.rl smoke --ppo --output artifacts/aso-ppo

# Explicitly smaller inference budget; training cost is still separately recorded:
poetry run python -m oligo_engine.rl compare --ppo --unique-budget 6 \
  --output artifacts/aso-budget6
```

All three commands currently use **synthetic fixtures only**. Their default config
is `configs/rl/aso_fixture_v1.toml`; CLI seed/step/scoring-cap overrides are written
to the resolved configuration. Existing output directories are refused. `score-panel`
enumerates the fixture; `compare`/`smoke` compare search methods, optionally training
PPO with `--ppo`. The environment and adapters accept explicit real-candidate
objects programmatically, but the CLI does not pretend to load absent real bundles.

On this host Poetry was invoked through cached `uv tool run --from poetry==2.1.4`;
`poetry sync --with science,rl --no-interaction` installed the project environment.
No pre-existing shared training runtime was modified. Cache/font warnings from
optional plotting dependencies were nonfatal; a writable `MPLCONFIGDIR` and
`XDG_CACHE_HOME` can be set outside the source tree.

Mask handling follows the inspected
[sb3-contrib 2.6.0 API](https://sb3-contrib.readthedocs.io/en/v2.6.0/modules/ppo_mask.html).
Both training and the custom validation/final rollout use masks. Generic Gymnasium
random-action checking does not respect masks; dedicated mask/transition tests are
used here. The wrapper follows the
[Gymnasium reset/step contract](https://gymnasium.farama.org/api/env/), distinguishing
terminal horizon/STOP from external budget truncation.

## Fixture evidence

The fixture is a 28-base synthetic transcript with ten possible 19-mer sites and an
artificial increasing site utility. Start is 1, radius/horizon 8, seed 7, gamma 1,
2,048 PPO training steps. Train/validation/final targets have distinct synthetic IDs
but identical task structure; this is not evidence of biological target transfer.
There are no measured labels, potency-trained weights or biological screens here.

| Method | Best visited utility, cap 10 | Unique scores used | Best utility, cap 6 |
| --- | ---: | ---: | ---: |
| Unchanged start | 0.15 | 1 | 0.15 |
| Random | 0.95 | 10 | 0.95 |
| Greedy/local | 0.95 | 10 | 0.55 |
| Integer genetic | 0.95 | 10 | 0.95 |
| PPO | 0.95 | 9 | 0.65 |
| Exhaustive reference | 0.95 | 10 | Reference evaluated separately |

At cap 6, PPO truncates before the next uncharged movement; its best visited utility
is 0.65 and its incomplete episode return is 0. This is not a terminal rejection of
the attempted unscored molecule. Random/genetic search outperform PPO on this tiny
budgeted fixture. At cap 10, all active search methods reach the known optimum.
There is no claim of PPO superiority from these checks.

PPO training made **2,250 scoring requests**, **10 unique scores**, **20 actual head
calls**. Validation made **36 requests**, **9 unique scores**, **18 head calls**.
Those costs are additional to fresh final inference, and are not hidden by the
cache. Saved-policy reload produced identical actions, candidate archives and
terminal reward. Manifests distinguish train/validation wall time, scoring wall
time and fresh-rollout wall time; these times are machine-dependent.

Local ignored receipts:

- `artifacts/aso-rl-fixture-20261114-costed/report.json`: current full-cost smoke,
  checkpoint, manifest and candidate archive.
- `artifacts/aso-rl-fixture-20261114-budget6/report.json`: smaller-cap negative
  comparison and truncation behavior. This earlier receipt predates the additional
  trainer wall-time fields; it retains scoring times and counts.

Reports record code revision and dirty status, runtime, resolved config, full endpoint
manifests, raw outputs, missingness, assay-qualified assessment IDs, independent
method caches, per-head call counts, request/unique-score traces and shortlist.
All results have explicit synthetic provenance. Generated reports/checkpoints stay
ignored, as do the original PDF and all model weights.

## Verification

`make lint format-check import-check test` passed in the installed Python 3.11
environment: **186 core tests** in the final run, including the added empty-feasible-
set case. All **5 optional RL tests** pass,
including actual training to the known optimum, direct scorer/environment parity,
mask behavior, budget truncation and checkpoint reload. The focused ASO suite also
passed after scorer/adapter updates. Optional plotting emitted 14 dependency
deprecation warnings; no test failed. `poetry check --lock`, documentation links,
`git diff --check` and artifact ignore checks passed. Poetry emitted existing
metadata deprecation notices. No remote CI run was claimed.

Production-boundary tests cover methylcytosine projection against the actual ported
Atlas parser, methylation/sugar/linkage/terminal preservation, label-free input
columns, model identity, raw uncertainty retention, missing required artifacts,
hash mismatch before reconstruction, fixed-assay preflight, raw screen rejection,
nonfinite inference and exact budget accounting. Model parity tests currently use
injected probes, **not an unavailable trained Atlas checkpoint**.

## Prerequisites for the real stage

Fetched and verified `origin/codex/atlas-lora-scaling` at **f197d79** without merging
it. Its completed report confirms the negative LoRA development result and the
200-target-label adaptation setting. The ported chemistry parser is unchanged
between the previously inspected `1b3fc1e` and `f197d79`.

The real stage requires a trusted potency bundle with fitted preprocessing, feature
order/output scaling, source revision, training/split identities, chemistry/context
support and fixed prediction probes. No such trained artifact is present in this
checkout. The model source/report refers to machine-local paths on another checkout;
those are not portable model delivery. The user has been asked for its location.

Before any real constrained PPO run, freeze source-only versus target-adapted access,
the assay profile, versioned transcript/target partitions, cross-source overlap
audit, ranking/reference construction, scoring budget and shortlist size. MOE
neurotoxicity is advisory until a measured screening event, cutoff and missed-case/
retention evidence are validated. Unsupported TLR chemistry remains not applicable;
unknown safety is not zero risk. The first real output should be a diagnostic panel
with selected Atlas inhibition, ASOptimizer cross-check and MOE neurotoxicity.

Measured feedback, predictor refitting and synthetic augmentation remain later
experiments. Expanded Gate 1 is unchanged. New ASOs remain unvalidated hypotheses.

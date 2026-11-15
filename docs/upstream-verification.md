# Pinned upstream verification — 2026-11-13

Historical verification log. November 14 scope update: all released source data
are now copied byte-for-byte into data/oligogym/ at the user's request, with license,
metadata and a checksum manifest. Cross-dataset diagnostics use this local catalog;
the generalized official reproduction command accepts `--dataset` and derives its
reference from the selected dataset. See dataset-pipelines.md. The earlier failed
Huesken attempts and unresolved Gate 1 remain as recorded below.

Later scope update: the published best models on all seven efficacy datasets now
run end to end, 35 full-data fits. The headline Huesken CNN agrees within the
frozen 0.03 numerical tolerance (0.609600 vs 0.630914); four datasets pass that
criterion, three remain outside. The historical statements below about the CNN
not yet being rerun describe the earlier attempts only. Current complete results
and limitations are in [efficacy reproduction](gate1-efficacy-reproduction.md).

## Source and environment

Upstream: [Roche/OligoGym at 97f5b9f](https://github.com/Roche/OligoGym/tree/97f5b9f58d9e445a8ecb0218037af7465c3708c0).
Fetched to a temporary checkout; its tracked source is unchanged. No datasets,
weights or source PDFs were copied into this repository. Source audit records SHA256
for registry, Huesken resources, split, features, models, benchmark CSV/script,
analysis notebook, pyproject and upstream lock.

Poetry 2.1.4 `install --only main` succeeded under Python 3.11.13. Model import then
failed with `ModuleNotFoundError: catboost`. Inspection also found unconditional
`torch_geometric` and `torchinfo` imports. Installed catboost 1.2.8,
torch-geometric 2.6.1 and torchinfo 1.8.0 into the temporary upstream environment;
the first two are undeclared upstream, and torchinfo is declared only in its dev
group. These additions are explicit environment deviations. No upstream source or
local project dependency lock was patched to disguise them.

Runtime: numpy 1.26.4, pandas 2.2.3, sklearn 1.4.0, scipy 1.15.3, torch 2.7.0;
macOS ARM64. An initial Poetry attempt selected Python 3.13 and began building numpy;
it was interrupted and replaced with the explicit Python 3.11 environment. It
produced no benchmark result.

## Verified identities and behavior

- Registry key `Huesken` resolves to `Ichihara_2007_1`, 2,431 rows / 30 targets.
  The real downloader and split were executed. `download("siRNA1")` raises ValueError.
  Analysis.ipynb cell 5 explicitly maps `siRNA1` to `Huesken`; the historical alias
  question is resolved at the pinned source.
- Huesken y ranges from 0.0 to 134.1. Do not assume all percentage-inhibition labels
  are bounded by 100 or silently clip them.
- Nucleobase splitting is group holdout, not cluster stratification. For seed 0,
  eight clusters and test_size=0.2 produce 1,896 train and 535 test rows, with no
  exact HELM overlap. The function returns six values in this configuration, despite
  its docstring promising cluster labels. `strand="RNA1"` does not restrict the
  kmer branch; KMersCounts defaults apply. See EVALUATION.md for the precise protocol.
- Splitter Git history has only the initial commit and c6eeebf, whose change repairs
  the validation branch's group-label reference. No changed train/test branch was
  found within this repository history. This does not prove the CSV's original
  execution environment or unpublished generating code matched current source.
- TLR7's registry description says TLR8 potentiation; TLR8's says TLR7 inhibition.
  Do not infer which assay is correct from keys alone.
- The GP wrapper inherits `predict(X)` returning means; it does not expose a
  `return_std` argument. Posterior uncertainty requires an explicit adapter and
  separate empirical calibration checks.

## Frozen linear-baseline rerun

Selected a published linear configuration before rerunning: KMersCounts k=[1,2,3],
modification_abundance=True, LinearModel type=standard, official nucleobase split,
8 clusters, test_size=0.2, no validation split, seeds 0–4. This is a reproduction
attempt for **that configuration**, not an attempt to claim the CNN winning score.

The upstream CSV has 45,950 rows. The five matching reference rows have mean r
0.6158645398542795 and sample SD 0.009902476290615764. The best overall siRNA1 CNN
configuration has mean 0.6309139529356906; it has not been rerun here.

| Seed | Train rows | Test rows | Local test Pearson r |
| ---: | ---: | ---: | ---: |
| 0 | 1,896 | 535 | 0.5981873115 |
| 1 | 1,648 | 783 | 0.5711477406 |
| 2 | 1,864 | 567 | 0.5728190668 |
| 3 | 1,890 | 541 | 0.6208409288 |
| 4 | 1,754 | 677 | 0.5611663267 |

Local mean **0.5848322748698586**, sample SD **0.02432406683154687**. Difference
from the published configuration is **-0.03103226498442091**. The frozen absolute
tolerance was 0.03; it was not met. A second execution with diagnostic logging
reproduced all five scores exactly. **Gate 1 remains open.** No seeds, threshold,
split or model were changed to make the result pass.

Diagnostic evidence: 150 input features, fitted rank 115–116, smallest singular
values around 1e-14 and maximum absolute coefficients from 5.3e13 to 2.7e14.
This indicates severe numerical ill-conditioning in this unregularized linear
configuration. It is a plausible contributor to platform/solver sensitivity, not
a demonstrated explanation for the published discrepancy. Do not silently substitute
ridge or change rank tolerance and call that the same baseline. Original split
seeds are not specified by the upstream benchmark script, another unresolved source
of variability. Next verification issue: recover original manifests/environment
(especially BLAS/LAPACK), or define and justify a separately named baseline protocol
before measured acquisition. Do not hunt for favorable seeds.

## Independent Random Forest cross-check

November 14 correction: this earlier attempt applied `max_depth=20` directly.
The audited benchmark's `prepare_model` filters that argument out, creating an
unlimited-depth forest. Thus it was not a faithful effective-model reproduction.
The result remains below as historical evidence. The fixed, source-helper study
and exact repeat are recorded in [Gate 1 reproducibility study](gate1-reproducibility-study.md).
Its native-helper mean is 0.559769 versus the 0.591037 reference, still outside the
unchanged 0.03 tolerance. No whole-discrepancy explanation or gate pass is claimed.

After diagnosing the linear training matrix's numerical instability, froze a
second published configuration in `configs/reproduce-rf.toml`: OneHotEncoder base
only, RandomForestModel with 100 trees and maximum depth 20, identical official
splitting and seeds 0–4. The choice was motivated by training-matrix diagnostics;
no linear seed, split, model or tolerance was changed. Random Forest uses an explicit
random_state=seed and n_jobs=2; historical model seeds remain unavailable.

| Seed | Local RF test Pearson r |
| ---: | ---: |
| 0 | 0.5387041891 |
| 1 | 0.5138994067 |
| 2 | 0.5536174279 |
| 3 | 0.5906665692 |
| 4 | 0.5985932204 |

Mean **0.5590961626513135** versus published RF configuration mean
**0.5910367239114824**, difference **-0.0319405612601689**. This also misses the
unchanged 0.03 tolerance. It supports investigating historical splits/environment
rather than attributing the whole discrepancy to the linear solver. It does not
prove a particular cause. Both attempts remain recorded; there is no numerical
Gate 1 pass and no measured acquisition claim. Do not continue selecting models
or seeds just to obtain a passing score.

Run this cross-check with the same reproduction command below, substituting
`--config configs/reproduce-rf.toml` and a fresh output path. The local evidence is
`artifacts/rf-baseline-rerun.json`.

## Commands

Core setup uses the existing Poetry lock. Upstream setup requires network access
and a writable temporary directory. Example from the project root:

```sh
git clone https://github.com/Roche/OligoGym.git /tmp/oligoal-upstream
git -C /tmp/oligoal-upstream checkout --detach 97f5b9f58d9e445a8ecb0218037af7465c3708c0
POETRY_VIRTUALENVS_PATH=/tmp/oligoal-envs poetry -C /tmp/oligoal-upstream env use python3.11
POETRY_VIRTUALENVS_PATH=/tmp/oligoal-envs poetry -C /tmp/oligoal-upstream install --only main
OLIGOAL_UP_ENV=$(POETRY_VIRTUALENVS_PATH=/tmp/oligoal-envs poetry -C /tmp/oligoal-upstream env info --path)
"$OLIGOAL_UP_ENV/bin/python" -m pip install catboost==1.2.8 torch-geometric==2.6.1 torchinfo==1.8.0
poetry run python -m oligo_engine.audit_upstream --checkout /tmp/oligoal-upstream --output artifacts/source-audit-new.json
PYTHONPATH=src MPLCONFIGDIR=/tmp/oligoal-mpl LOKY_MAX_CPU_COUNT=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "$OLIGOAL_UP_ENV/bin/python" -m oligo_engine.reproduce \
  --checkout /tmp/oligoal-upstream --config configs/reproduce-linear.toml \
  --output artifacts/linear-rerun-new.json
```

These reproduce the attempted protocol; they do not promise historical agreement
or platform-independent floating-point scores. Supplemental transitive dependencies
are not a fully locked scientific environment yet. The local evidence files are
`artifacts/upstream-source-audit.json`, `artifacts/linear-baseline-rerun.json`, and
`artifacts/linear-baseline-diagnostics.json`; the last includes the corrected alias
audit and numerical diagnostics. They are ignored artifacts, not committed data.

## Project checks

Core tests require only Ruff and pytest. They cover atomic label reveal, denial of
held-out/unrevealed labels, label poisoning, fit-ID accounting, deferred evaluation,
invalid selections, exact partial budgets, target/sequence split integrity,
source-revision guards, strict artifact serialization, and CLI execution. Final
check results are reported by the implementation response; CI has not run remotely.

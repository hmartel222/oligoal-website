# OligoAL ASO workbench

Run locally from the repo root with Python 3.11, the `science` and `rl` Poetry
groups, and the pinned OligoGym checkout importable in that environment. See
[upstream setup](../docs/upstream-verification.md) and
[RL execution](../docs/aso-rl-execution.md) for the pinned versions.

```sh
poetry run python app/server.py --port 8000
```

Open [localhost:8000](http://127.0.0.1:8000), enter a human gene symbol, and
run the design. Gene lookup uses Ensembl and caches the resulting cDNA. Paste
a transcript to work offline. Each run resolves its own transcript and
replaces the results. CSV exports the rows from that run.
**Export shortlist CSV** downloads the visible rows. **Export full CSV**
ranks the same submitted transcript and profile with the same deterministic
settings and downloads every ranked candidate. The browser keeps showing
the shortlist.

## Design and model coverage

- **2′-MOE** uses the actual `ASODesignSpace`, `ASOEnv`, and frozen scoring session.
  It explores bounded windows across the transcript and ranks visited candidates
  by the ASOptimizer GP's predicted knockdown. The MOE GP reports its acute
  neurotoxicity estimate as an advisory result. The server fits the exploratory
  deployment models once from the released labels. Those fits and this interface
  are separate from split-bound scientific benchmarks.
- **2′-OMe immune panel** uses the TLR7 RF and TLR8 GP on matching molecules.
  The panel ranks by distance from the TLR8 assay control and reports both
  reporter responses separately. The current Gymnasium environment supports MOE.
- **Atlas efficacy · 20 nt** scores every unambiguous transcript window
  using the frozen ASO Atlas position-specific Ridge model. The fixed assay
  is human HepG2, electroporation, 4,500 nM and 24 hours. The model uses full-PS
  5-10-5 MOE gapmers with methylcytosine. Provide the verified model checkpoint
  using the [root setup instructions](../README.md) to enable this mode.
- IC₅₀ and hepatotoxicity remain unavailable. Knockdown is displayed as percent
  inhibition.

## Exploratory PPO policy

With no policy, the MOE path traverses legal moves in the same environment.
To train an exploratory policy locally from the existing knockdown scorer:

```sh
poetry run python app/server.py --train-policy artifacts/aso-app-policy
poetry run python app/server.py --port 8000
```

Training downloads the HTT and APP transcripts for its separate training and
validation environments. Override them with `--training-gene` and
`--validation-gene`. The server detects this ignored artifact directory, or
accepts `--policy PATH` for a trusted checkpoint and manifest. Policy loading
validates the hash and scorer/action/observation compatibility. The UI reports
which method ran. This small exploratory fit establishes pipeline execution.
Its evaluation scope is distinct from a locked multi-target PPO benchmark.

## Duplex viewing and PyMOL

Select a candidate on Design or open **Molecular structure** in Bench. Zoom,
pan, rotate, or select a heavy atom for its coordinates. Switch between ball
and stick and space-filling, pause the motion preview, or expand the viewer.
**Full transcript** represents all nucleotides as a schematic trace with the
short ASO bound at its actual coordinates on the selected transcript. The
binding site uses a template-built heavy-atom model. The rest of the transcript
has no inferred 3D fold. The motion preview is an illustrative animation.

The site serves the pinned Three.js 0.179.1 renderer locally. Chain A is the
gapmer and chain B is the complementary window in both structure downloads.
The nucleotide geometry is derived from [PDB 1FIX](https://www.rcsb.org/structure/1FIX)
and the [wwPDB Chemical Component Dictionary](https://www.wwpdb.org/data/ccd).
Modified sugar and phosphorothioate conformations are approximate.

For desktop PyMOL, download the self-contained **PyMOL script** and run
`@/path/to/aso-duplex.pml` from the PyMOL command line, using the actual
downloaded filename. It loads and styles both strands. The PDB can also be
opened directly in any PDB viewer. [Open-source PyMOL](https://github.com/schrodinger/pymol-open-source)
can read both files. The commercial [PyMOL distribution](https://pymol.org/)
has its own licensing. The workbench needs no PyMOL credentials.


## Local research checkpoints

The optional research service runs only on loopback and requires an independent
Bench directory. Use the website's regular Python environment to start it:

```sh
export PYTHONPATH="$PWD/src:$PWD/vendor/OligoGym"
.venv/bin/python app/server.py --no-warm --port 8131 \
  --bench-dir /path/to/local-private-bench \
  --bench-origin http://127.0.0.1:8131 \
  --research-config /path/to/research.json
```

Provide these keys in `research.json` as absolute paths:

- `checkpoints` points to `model_checkpoints` in the OligoAL research repository.
- `project_src` is the current research checkout's `src` directory.
- `policy_python` is a Python 3.11 interpreter with the research project's
  pinned scientific and experimental RL dependencies.
- `atlas_src` is `src` from OligoAL commit `f197d79`.
- `atlas_python` is a Python 3.11 interpreter with PyTorch 2.7.0 and the
  scientific dependencies from the Atlas scaling study, including XGBoost.
- `rnafm_src` is the clean RNA-FM source checkout at
  `348951516e0963d22bbb33b3c9fc18c89081d38e`.
- `rnafm_dependencies` points to a directory with the RNA-FM dependencies
  (`ptflops==0.7.2.2`, `torchvision==0.22.0`).
- `backbone` is the verified external `RNA-FM_pretrained.pth`.
- `cache` is a writable local directory for RNA-FM token caching.

Run an Atlas 20-mer design, then use **Score shortlist** to try a source-trained
Atlas Ridge, XGBoost, frozen RNA-FM or LoRA checkpoint, or the PPO companion Ridge
model. These are research predictions on the existing shortlist and its fixed
assay profile. Target-specific fits and intermediate training snapshots remain
research artifacts with their original applicability.

In local Bench, the published 3×32 experimental-selection PPO proposes one
candidate at a time. It uses its companion frozen Ridge predictor, a 50-assay
budget, and at least 10 distinct compatible recorded results to initialize its
observation. Resolve pending assays before asking PPO for another candidate.
The adapter has no future outcomes and recommendations do not purchase assays.
The policy is an experimental transfer from retrospective development campaigns.

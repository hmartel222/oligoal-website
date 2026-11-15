#!/usr/bin/env python3
"""Pre-fetch Ensembl transcripts so the deployed site can resolve gene names.

    .venv/bin/python scripts/warm-genes.py MAPT SOD1 HTT

A cold gene lookup makes two sequential Ensembl REST calls before the design
run even starts. Through Netlify's proxy, which drops any request still open
after about 30 seconds, that is usually fatal; on localhost it merely feels
slow. Fetching each gene once writes artifacts/transcripts/<species>.<GENE>.json,
and every later lookup reads that file instead of the network.

Run this for every gene you intend to demo, while the tunnel is irrelevant and
nobody is watching.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oligo_engine.design.transcripts import TranscriptError, cache_path, fetch  # noqa: E402

# Reasonable defaults if none are given: common antisense targets.
DEFAULT_GENES = ["MAPT", "SOD1", "HTT", "APOE", "SNCA"]


def main() -> int:
    args = sys.argv[1:]
    if any(a in ("-h", "--help") for a in args):
        print(__doc__)
        return 0
    genes = [g.strip().upper() for g in args if g.strip()] or DEFAULT_GENES
    print(f"Warming {len(genes)} gene(s). Each cold fetch can take 10-40 seconds.\n")

    failures = []
    for gene in genes:
        path = cache_path(gene, "homo_sapiens")
        if path.exists():
            print(f"  {gene:<8} already cached")
            continue
        start = time.monotonic()
        try:
            transcript = fetch(gene)
        except ModuleNotFoundError:
            print("\nRun this with the project venv: .venv/bin/python scripts/warm-genes.py")
            return 1
        except (TranscriptError, OSError) as error:
            failures.append(gene)
            print(f"  {gene:<8} FAILED  {error}")
            continue
        elapsed = time.monotonic() - start
        print(
            f"  {gene:<8} cached in {elapsed:5.1f}s  "
            f"{transcript.transcript_id}  {len(transcript.sequence)} nt"
        )

    print(f"\nCache directory: {cache_path('X', 'homo_sapiens').parent}")
    if failures:
        print(f"Could not fetch: {', '.join(failures)}")
        print("Those genes will still try the network at demo time.")
        return 1
    print("Every gene above now resolves instantly, including through Netlify.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

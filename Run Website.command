#!/bin/zsh
set -eu
cd -- "$(dirname -- "$0")"
if [[ ! -x .venv/bin/python ]]; then
  printf 'Set up the website Python environment using README.md.\n' >&2
  exit 1
fi
export PYTHONPATH="$PWD/src:$PWD/vendor/OligoGym"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
exec .venv/bin/python -B app/server.py --port 8128 "$@"

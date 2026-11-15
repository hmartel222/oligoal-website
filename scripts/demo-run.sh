#!/bin/sh
# Start the OligoAL server for the demo, with the bench workspace enabled.
#   sh scripts/demo-run.sh
set -eu

cd -- "$(dirname -- "$0")/.."

if [ ! -x .venv/bin/python ]; then
  printf 'No .venv here. Set up the environment using README.md first.\n' >&2
  exit 1
fi

export PYTHONPATH="$PWD/src:$PWD/vendor/OligoGym"

printf 'Starting OligoAL. The models take about 90 seconds to fit.\n'
exec .venv/bin/python app/server.py --port 8128 \
  --bench-dir "$HOME/oligoal-bench" \
  --bench-origin https://oligoal.netlify.app

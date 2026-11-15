#!/bin/sh
# Start the OligoAL server for the demo, with the bench workspace enabled.
#   sh scripts/demo-run.sh                           (defaults to the current site)
#   sh scripts/demo-run.sh https://your-site.netlify.app
#
# The argument is the NETLIFY url, not the tunnel url: it is the origin the
# browser is on, and bench requests from any other origin are refused.
set -eu

cd -- "$(dirname -- "$0")/.."

site=${1:-https://funny-kheer-33b421.netlify.app}
case "$site" in
  https://*) ;;
  *) printf 'Pass your full Netlify URL, e.g. https://funny-kheer-33b421.netlify.app\n' >&2; exit 1 ;;
esac

if [ ! -x .venv/bin/python ]; then
  printf 'No .venv here. Set up the environment using README.md first.\n' >&2
  exit 1
fi

export PYTHONPATH="$PWD/src:$PWD/vendor/OligoGym"

printf 'Starting OligoAL for %s. The models take about 90 seconds to fit.\n' "$site"
exec .venv/bin/python app/server.py --port 8128 \
  --bench-dir "$HOME/oligoal-bench" \
  --bench-origin "$site"

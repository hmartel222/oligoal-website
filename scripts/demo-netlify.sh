#!/bin/sh
# Point the Netlify site at the tunnel currently running on this laptop.
#
#   sh scripts/demo-netlify.sh https://something.trycloudflare.com
#   sh scripts/demo-netlify.sh https://something.trycloudflare.com --deploy
#
# Rebuilds dist/ with an /api/* proxy aimed at the tunnel. Without --deploy it
# stops there and you drag dist/ onto Netlify yourself; with --deploy it pushes
# to production through the Netlify CLI. Run it again after every tunnel
# restart, because the trycloudflare address changes each time.
set -eu

cd -- "$(dirname -- "$0")/.."

if [ $# -lt 1 ]; then
  printf 'Usage: sh scripts/demo-netlify.sh https://something.trycloudflare.com [--deploy]\n' >&2
  exit 1
fi

tunnel=$1
shift

API_ORIGIN="$tunnel" sh scripts/netlify-build.sh

if [ "${1:-}" = "--deploy" ]; then
  printf '\nDeploying dist/ to Netlify production...\n'
  npx --yes netlify-cli deploy --prod --dir=dist
else
  printf '\ndist/ is ready. Either:\n'
  printf '  - drag the dist folder onto https://app.netlify.com/drop, or\n'
  printf '  - rerun this with --deploy to push it through the Netlify CLI.\n'
fi

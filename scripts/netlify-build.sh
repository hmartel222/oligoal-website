#!/bin/sh
# Assemble the Netlify publish directory from app/static and write the redirect
# rules. Netlify cannot run the Python API, so /api/* is proxied to the backend
# named by API_ORIGIN. See netlify.toml.
set -eu

cd -- "$(dirname -- "$0")/.."

rm -rf dist
mkdir -p dist
cp -R app/static/. dist/

# /bench is a clean URL for bench.html, matching the local server's routing.
printf '/bench  /bench.html  200\n' > dist/_redirects

if [ -n "${API_ORIGIN:-}" ]; then
  origin=${API_ORIGIN%/}
  case "$origin" in
    https://*) ;;
    *)
      printf 'netlify-build: API_ORIGIN must start with https:// (got %s)\n' "$origin" >&2
      exit 1
      ;;
  esac
  printf '/api/*  %s/api/:splat  200\n' "$origin" >> dist/_redirects
  printf 'netlify-build: proxying /api/* to %s\n' "$origin"
else
  # Deploy the frontend anyway, but make the missing backend legible in the UI
  # instead of surfacing a parse error.
  cat > dist/api-unavailable.json <<'JSON'
{"error":"The OligoAL API is not configured for this deploy. Set API_ORIGIN in the Netlify site environment to the backend's HTTPS origin, then redeploy."}
JSON
  printf '/api/*  /api-unavailable.json  503\n' >> dist/_redirects
  printf 'netlify-build: API_ORIGIN is unset; /api/* returns 503\n' >&2
fi

printf 'netlify-build: published %s files\n' "$(find dist -type f | wc -l | tr -d ' ')"

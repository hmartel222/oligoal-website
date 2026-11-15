#!/bin/sh
# One-time demo setup: bench access key, QR dependency, first dist/ build.
#   sh scripts/hackathon-setup.sh
set -eu

cd -- "$(dirname -- "$0")/.."
repo=$(pwd)
bench="$HOME/oligoal-bench"

mkdir -p "$bench"
chmod 700 "$bench"
if [ ! -e "$bench/access-key" ]; then
  (umask 077 && python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > "$bench/access-key")
fi
chmod 600 "$bench/access-key"
printf 'Bench workspace ready at %s. Read your key from access-key.\n' "$bench"

if python3 -c 'import segno' 2>/dev/null; then
  printf 'QR library already installed.\n'
else
  printf 'Installing the QR library...\n'
  python3 -m pip install --quiet segno || python3 -m pip install --quiet --break-system-packages segno
fi

# Build an initial dist/ so there is something to create the Netlify site with.
API_ORIGIN="https://placeholder.invalid" sh scripts/netlify-build.sh >/dev/null
printf 'Built %s/dist\n' "$repo"

cat <<DONE

Next:
  1. Open https://app.netlify.com/drop and drag this folder onto it:
       $repo/dist
  2. On the new site: Site configuration -> Change site name -> oligoal
     Your URL becomes https://oligoal.netlify.app
  3. Make the QR (once):
       python3 scripts/demo-qr.py https://oligoal.netlify.app

DONE

if [ "$(uname)" = "Darwin" ]; then
  open "$repo/dist" 2>/dev/null || true
fi

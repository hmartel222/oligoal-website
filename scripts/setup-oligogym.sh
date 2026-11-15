#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
revision=97f5b9f58d9e445a8ecb0218037af7465c3708c0
checkout=vendor/OligoGym

if [ -d "$checkout/.git" ]; then
    if [ "$(git -C "$checkout" rev-parse HEAD)" != "$revision" ]; then
        printf 'Expected %s at revision %s.\n' "$checkout" "$revision" >&2
        exit 1
    fi
    exit 0
fi

git clone --no-checkout --filter=blob:none https://github.com/Roche/OligoGym.git "$checkout"
git -C "$checkout" sparse-checkout set oligogym
git -C "$checkout" checkout --detach "$revision"

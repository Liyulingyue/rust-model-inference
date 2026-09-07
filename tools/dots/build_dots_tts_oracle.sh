#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 OFFICIAL_CHECKOUT" >&2
    exit 2
fi

oracle_dir=$(cd "$1" 2>/dev/null && pwd -P) || {
    echo "not a directory: $1" >&2
    exit 2
}
if ! git -C "$oracle_dir" rev-parse --git-dir >/dev/null 2>&1; then
    echo "not a git checkout: $oracle_dir" >&2
    exit 2
fi

pinned=32407a55228630475c48ecdb2c4e2c0f9c09e030
script_dir=$(cd "$(dirname "$0")" && pwd -P)
patch="$script_dir/dots-tts-oracle-trace.patch"
origin=$(git -C "$oracle_dir" remote get-url origin)
build_root=$(mktemp -d "${TMPDIR:-/tmp}/dots-tts-oracle.XXXXXX")
build_dir="$build_root/repo"

git clone --no-checkout "$origin" "$build_dir" >&2
git -C "$build_dir" fetch origin "$pinned" >&2
git -C "$build_dir" checkout --detach "$pinned" >&2
git -C "$build_dir" apply --check "$patch" >&2
git -C "$build_dir" apply "$patch" >&2

printf '%s\n' "$build_dir"

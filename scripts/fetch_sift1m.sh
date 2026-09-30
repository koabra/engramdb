#!/usr/bin/env bash
set -euo pipefail

destination="${1:-/tmp/engramdb-sift1m}"
base_url="${SIFT1M_MIRROR:-https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main}"
mkdir -p "$destination"

download() {
  local filename="$1"
  local expected_size="$2"
  local target="$destination/$filename"
  if [[ -f "$target" ]] && [[ "$(stat -c %s "$target")" == "$expected_size" ]]; then
    return
  fi
  rm -f "$target.tmp"
  curl --fail --location --retry 4 --retry-all-errors \
    --output "$target.tmp" "$base_url/$filename"
  if [[ "$(stat -c %s "$target.tmp")" != "$expected_size" ]]; then
    printf 'unexpected size for %s\n' "$filename" >&2
    exit 1
  fi
  mv "$target.tmp" "$target"
}

download sift_base.fvecs 516000000
download sift_query.fvecs 5160000
download sift_groundtruth.ivecs 4040000

sha256sum \
  "$destination/sift_base.fvecs" \
  "$destination/sift_query.fvecs" \
  "$destination/sift_groundtruth.ivecs"

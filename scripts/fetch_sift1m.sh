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

(
  cd "$destination"
  printf '%s  %s\n' \
    21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816 sift_base.fvecs \
    f7fc9be140accdfd64116c2fa2365ecdb69b8f084970c6b0532db5ff79ac8fdc sift_query.fvecs \
    2b71de0a8d5a83e6a84eec3e23fb8b611d8801dd9b3a6cd62f070ab65ea65f4f sift_groundtruth.ivecs \
    | sha256sum --check
  sha256sum sift_base.fvecs sift_query.fvecs sift_groundtruth.ivecs
)

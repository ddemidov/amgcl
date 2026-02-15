#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${SRC_DIR:-${PWD}}/build_conda"
mkdir -p "$BUILD_DIR"

if [ -z "${GIT_DESCRIBE_TAG:-}" ]; then
  echo "ERROR: GIT_DESCRIBE_TAG is not set. Set it to the current release tag (e.g. v1.4.5)." >&2
  echo "Run: export GIT_DESCRIBE_TAG=\"$(git -C \"${SRC_DIR:-$(pwd)}\" describe --tags --abbrev=0)\"" >&2
  exit 1
fi

cmake -S "${SRC_DIR:-${PWD}}" \
    -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DAMGCL_BUILD_TESTS=OFF \
    -DAMGCL_BUILD_EXAMPLES=OFF

cmake --build "$BUILD_DIR" -- -j"${CPU_COUNT:-1}"
cmake --install "$BUILD_DIR"

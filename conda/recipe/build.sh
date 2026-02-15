#!/usr/bin/env bash
set -euxo pipefail

if [ -z "${GIT_DESCRIBE_TAG:-}" ]; then
  echo "ERROR: GIT_DESCRIBE_TAG is not set. Set it to the current release tag (e.g. v1.4.5)." >&2
  echo "Run: export GIT_DESCRIBE_TAG=\"$(git -C \"${SRC_DIR:-$(pwd)}\" describe --tags --abbrev=0)\"" >&2
  exit 1
fi
export GIT_DESCRIBE_TAG

cmake -S . -B build \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DAMGCL_BUILD_TESTS=OFF \
    -DAMGCL_BUILD_EXAMPLES=OFF

cmake --build build -- -j"${CPU_COUNT:-1}"
cmake --install build

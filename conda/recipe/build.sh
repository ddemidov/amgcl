#!/usr/bin/env bash
set -euxo pipefail

if [ -z "${GIT_DESCRIBE_TAG:-}" ]; then
  GIT_DESCRIBE_TAG="$(git -C "${SRC_DIR:-$(pwd)}" describe --tags --abbrev=0)"
fi
export GIT_DESCRIBE_TAG

cmake -S . -B build \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DAMGCL_BUILD_TESTS=OFF \
    -DAMGCL_BUILD_EXAMPLES=OFF

cmake --build build -- -j"${CPU_COUNT:-1}"
cmake --install build

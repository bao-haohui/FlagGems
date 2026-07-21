#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  echo "ASCEND_HOME_PATH is not set. Source the active CANN setenv.bash first." >&2
  exit 2
fi

arch="${INDEX_FILL_P0_NPU_ARCH:-dav-2201}"
arch_id="${INDEX_FILL_P0_NPU_ARCH_ID:-2201}"
build_dir="${INDEX_FILL_P0_BUILD_DIR:-build}"

cmake -S . -B "${build_dir}" \
  -DINDEX_FILL_P0_NPU_ARCH="${arch}" \
  -DINDEX_FILL_P0_NPU_ARCH_ID="${arch_id}"
cmake --build "${build_dir}" -j"${CMAKE_BUILD_PARALLEL_LEVEL:-4}"

"${build_dir}/index_fill_ascendc_p0" "$@"

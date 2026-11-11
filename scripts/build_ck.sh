#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source_dir=${1:?Usage: build_ck.sh /path/to/rocm-libraries [build-dir]}
build_dir=${2:-"$repo_root/build/miopen-gfx1103"}
source_dir=$(realpath "$source_dir")
base_commit=39d8d60365d2272a5366f76ccb3bfba24b219eaf
if [[ $(git -C "$source_dir" rev-parse HEAD) != "$base_commit" ]]; then
    echo "Expected rocm-libraries commit $base_commit" >&2
    exit 1
fi
if git -C "$source_dir" apply --check "$repo_root/patches/miopen-gfx1103.patch" 2>/dev/null; then
    git -C "$source_dir" apply "$repo_root/patches/miopen-gfx1103.patch"
elif ! git -C "$source_dir" apply --reverse --check "$repo_root/patches/miopen-gfx1103.patch" 2>/dev/null; then
    echo 'Source does not match the patch' >&2
    exit 1
fi
: "${ROCM_DEVEL:?Set ROCM_DEVEL to the merged _rocm_sdk_devel directory; see docs/setup.md}"
cmake -S "$source_dir/projects/miopen" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="$ROCM_DEVEL/lib/llvm/bin/clang++" \
    -DCMAKE_CXX_FLAGS="--rocm-path=$ROCM_DEVEL --rocm-device-lib-path=$ROCM_DEVEL/amdgcn/bitcode -isystem $ROCM_DEVEL/include" \
    -DCMAKE_PREFIX_PATH="$ROCM_DEVEL" -DROCM_PATH="$ROCM_DEVEL" \
    -DGPU_TARGETS=gfx1103 -DMIOPEN_BUILD_CK=ON -DMIOPEN_BUILD_DRIVER=OFF \
    -DMIOPEN_USE_MLIR=OFF -DMIOPEN_USE_SQLITE_PERFDB=OFF \
    -DMIOPEN_HIP_COMPILER="$ROCM_DEVEL/lib/llvm/bin/clang++" \
    -DMIOPEN_AMDGCN_ASSEMBLER="$ROCM_DEVEL/lib/llvm/bin/clang" \
    -DMIOPEN_OFFLOADBUNDLER_BIN="$ROCM_DEVEL/lib/llvm/bin/clang-offload-bundler"
cmake --build "$build_dir" --target MIOpenCKGroupedConv_gfx1103 --parallel "${BUILD_JOBS:-16}"
plugin_dir="$HOME/.local/lib/rocm-gfx1103-ck"
mkdir -p "$plugin_dir"
install -m 755 "$build_dir/lib/libMIOpenCKGroupedConv_gfx1103.so" "$plugin_dir/"
install -m 755 "$build_dir/_deps/composable_kernel-build/lib/libdevice_conv_operations.so.SOVERSION" "$plugin_dir/"
echo "Installed CK plugin to $plugin_dir"

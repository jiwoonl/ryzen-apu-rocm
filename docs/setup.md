# Setup

Tested on the machine and versions in the [README](../README.md). Run commands from this repository's root. Keep the Python ROCm 10.1 wheels together; Fedora's HIP tools are a separate installation.

## Fedora and conda

```sh
sudo dnf install cmake gcc-c++ make boost-devel openssl-devel protobuf-devel \
  rapidjson-devel libcurl-devel json-c-devel ncurses-devel libuuid-devel \
  libdrm-devel elfutils-devel opencl-headers OpenCL-ICD-Loader-devel \
  pybind11-devel systemtap-sdt-devel git-lfs \
  rocminfo rocm-hip rocm-hip-devel hipcc rocm-device-libs amd-gpu-firmware
```

Use `OpenCL-ICD-Loader-devel` with Fedora's existing loader. `ocl-icd-devel` conflicts with it; removing the loader is unnecessary.

For an existing Miniconda installation, add this to `~/.zshrc` if you want PATH access without automatic shell initialization:

```sh
export PATH="$HOME/miniconda3/bin:$PATH"
```

Then initialize only the current shell and create an environment. Pick any name; `apu` is an example. An existing environment with Python 3.13 also works.

```sh
eval "$(conda shell.zsh hook)"
conda env create -n apu -f environment.yml
conda activate apu
python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
  -r requirements-rocm.txt
```

`environment.yml` has no fixed environment name and installs only Python/build tooling. Install the ROCm wheels after activating your chosen environment. Hooks are installed under its `CONDA_PREFIX`; no environment named `rocm` is required.

The requirements select `device-gfx1103` for torch, torchvision, and the ROCm SDK. Do not substitute a `gfx1100` override. Wheel indexes change; if the pinned build leaves the stable index, check [TheRock releases](https://github.com/ROCm/TheRock/blob/main/RELEASES.md) or the [nightly index](https://nightly.repo.amd.com/rocm/whl-next/) for the same versions. The full ROCm/Torch pins must resolve together.

Check access to `/dev/kfd` and `/dev/dri/renderD128`. If necessary, add your user to the distribution's `render`/`video` groups and log in again. `rocminfo` should report `gfx1103`.

## MIOpen CK plugin for 780M

The wheels used here lacked the `gfx1103` MIOpen CK grouped-convolution plugin needed by some CNN paths. The included patch adds the target, guards XDL-only code, and bypasses a target-probe failure in this split SDK layout. Compiler warnings are suppressed for this experimental CK build.

```sh
mkdir -p vendor
git clone --filter=blob:none https://github.com/ROCm/rocm-libraries.git vendor/rocm-libraries
git -C vendor/rocm-libraries checkout 39d8d60365d2272a5366f76ccb3bfba24b219eaf
python scripts/prepare_rocm_devel.py
```

The last command installs the development wheel in `build/ck-env` and links its split SDK siblings to the active runtime environment. Export the printed `ROCM_DEVEL` path, then build:

```sh
export ROCM_DEVEL="$PWD/build/ck-env/lib/python3.13/site-packages/_rocm_sdk_devel"
BUILD_JOBS=16 bash scripts/build_ck.sh "$PWD/vendor/rocm-libraries"
python scripts/install_conda_hooks.py
conda deactivate
conda activate apu
python tests/gpu_ops.py
python tests/gpu_models.py
```

The build installs only `libMIOpenCKGroupedConv_gfx1103.so` and `libdevice_conv_operations.so.SOVERSION` under `~/.local/lib/rocm-gfx1103-ck`. It does not replace the wheel's `libMIOpen.so`. Hooks enable the plugin only inside this conda environment and use an isolated MIOpen database under `~/.cache/rocm-gfx1103-ck/miopen`. No wrapper command is needed. Reduce `BUILD_JOBS` if compilation exhausts RAM.

## XDNA1 and XRT

The Fedora kernel provides `amdxdna`. Check `/dev/accel/accel0`; avoid installing a second kernel driver over a working one. This setup built XRT userspace from the driver's pinned XRT submodule:

```sh
git clone --filter=blob:none https://github.com/amd/xdna-driver.git vendor/xdna-driver
git -C vendor/xdna-driver checkout 5599041841d8522e8f206b6fc55e0230f8cac915
git -C vendor/xdna-driver submodule update --init xrt
# XRT submodule: fdb373b1a43cd44201c799b80e2dd38df68e86b1
cmake -S vendor/xdna-driver/xrt/src -B build/xrt \
  -DXRT_NPU=1 -DXRT_ENABLE_WERROR=0 \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local/opt/xrt"
cmake --build build/xrt --parallel 16
DESTDIR="$PWD/build/xrt-stage" cmake --install build/xrt
mkdir -p "$HOME/.local/opt/xrt"
cp -a "build/xrt-stage$HOME/.local/opt/xrt/." "$HOME/.local/opt/xrt/"
export XILINX_XRT="$HOME/.local/opt/xrt"
source "$XILINX_XRT/setup.sh"
xrt-smi examine
```

The original successful validation reported 78 us latency and 38,511 ops/s. Those are XRT validation kernels, **not** the two-matmul benchmark.

An 8 MiB locked-memory limit caused a 64 MiB `mmap` failure (`err=-11`). Set a persistent limit for the user:

```sh
printf '%s soft memlock unlimited\n%s hard memlock unlimited\n' "$USER" "$USER" \
  | sudo tee /etc/security/limits.d/99-npu-memlock.conf
```

Log out and back in, then check `ulimit -l` reports `unlimited` before `xrt-smi validate`. The admin/performance-mode warning alone is not a validation failure. This project does not require forcing performance mode.

## MLIR-AIE, Peano, and Python bindings

```sh
python -m pip install 'mlir_aie==1.3.4' \
  -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.3.4
python scripts/prepare_peano.py
python -m pip install wheels/llvm_aie-21.0.0.2026062301+cb664e8c-*.whl
python scripts/build_pyxrt.py --source vendor/xdna-driver/xrt --install
python scripts/patch_mlir_loader.py
python scripts/install_conda_hooks.py
python -m pip install --no-deps -e .
python -m pip check
```

The pinned Peano wheel has `py3-none` in its filename but CPython-specific internal tags. `prepare_peano.py` fixes the tags and RECORD hash before installation; it leaves compiler binaries unchanged. Peano 22 was not compatible with this MLIR-AIE setup.

PyXRT must match both the installed XRT and active Python ABI. The build script uses PyTorch's bundled pybind11 headers. The MLIR loader patch temporarily enables Linux `RTLD_DEEPBIND` during initialization; this fixed an LLVM symbol collision when importing torch before AIE in one process. It keeps a backup and leaves the process's loader flags restored.

The activation hooks set `XILINX_XRT` and expose its tools. Deactivate/reactivate after installing them. Run:

```sh
python tests/npu.py
python tests/unified_api.py
python tests/unified_stress.py
python benchmarks/pipeline.py
```

`tests/npu.py` verifies actual PyXRT launch counts, both memory layouts, batches 1/2/4/8/16, changing inputs, and invalid-input rejection. GPU training and longer stability checks are separate:

```sh
python tests/gpu_training.py   # downloads MNIST
python tests/gpu_stability.py
python benchmarks/tune.py dram --k 64 --n 64 --depth 2 --dma 4
python benchmarks/tune.py sram --k 64 --n 64 --depth 1
python scripts/environment.py
```

The source patches and build recipes describe the reference setup. Hardware checks are rerun before publication; a complete installation from an empty Fedora system has not been rerun.

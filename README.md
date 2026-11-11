# Ryzen APU ROCm

ROCm/HIP on a Ryzen 7 PRO 8840U and Radeon 780M (`gfx1103`), with XDNA1 NPU experiments. Fedora setup notes, a MIOpen CK patch, and runnable tests.

## Results

`ReLU(A @ B) @ D`, 512×512 matrices, BF16 arithmetic, resident weights. Timings include host FP32 input/output conversion and transfers. Each row has 300 warm samples; compilation and initial device loading are excluded.

| Path | Batch | Mean (ms) | Median (ms) | p95 (ms) | Inputs/s |
|---|---:|---:|---:|---:|---:|
| 780M GPU | 1 | 0.643 | 0.413 | 1.591 | 1,555 |
| NPU, original tiles | 1 | 1.317 | 0.822 | 3.620 | 759 |
| NPU, tuned | 1 | 0.632 | 0.613 | 0.764 | 1,583 |
| NPU, SRAM intermediate | 1 | 0.781 | 0.763 | 0.941 | 1,281 |
| NPU, tuned | 2 | 1.130 | 1.094 | 1.343 | 1,770 |
| NPU, tuned | 4 | 2.217 | 2.161 | 2.579 | 1,804 |
| NPU, tuned | 8 | 4.582 | 4.560 | 4.945 | 1,746 |
| NPU, tuned | 16 | 12.441 | 11.611 | 17.289 | 1,286 |

Latency is for the whole batch; one input is one complete two-matmul pipeline. Desktop scheduling affected the tails. [Full results](results/reference/sram-batch-benchmark-results.json) include both NPU backends at every batch size.

- Both NPU matmuls and ReLU run in **one actual XRT launch**, including batched inputs.
- The default uses shared-memory scratch. The SRAM version keeps 32 KiB per compute core across 16 cores, but needs more input/weight traffic and was slower here.
- 34 tile/DMA/buffer configurations passed verification. The default uses `m=32, k=64, n=64`, FIFO depth 2. Batch 4 gave the highest measured throughput.
- NPU code is fixed-shape inference, without autograd. BF16 accumulation differs from FP32: relative L2 error was about 2.1% on the recorded random test. Same-tile independent NPU stages matched bit for bit.

GPU checks cover FP32/FP16 matmul, convolution/backprop, FFT, torchvision NMS, CNNs, attention, LSTM, MNIST training, and synthetic regression. Stress scripts exercise 2,000 inference calls and 200 training steps. These are smoke tests, not model accuracy benchmarks. [Publication checks](results/reference/publication-checks.json) passed on the environment below.

## Environment

ThinkPad T14 Gen 5 AMD (`21MCCTO1WW`), captured at publication:

| Component | Version |
|---|---|
| CPU | Ryzen 7 PRO 8840U, 8 cores / 16 threads |
| GPU | Radeon 780M, `gfx1103` |
| NPU | XDNA1 / `RyzenAI-npu1`, `aie2` |
| RAM | About 30 GiB usable, shared by CPU/GPU/NPU |
| OS | Fedora 45 KDE, x86_64; glibc 2.44 |
| Kernel | `7.3.3-300.fc45.x86_64` |
| Python | 3.13.18, conda environment `rocm` |
| Fedora HIP / rocminfo | 7.2.2 / 7.2.0 |
| Python ROCm SDK | 10.1.0, native `gfx1103` device packages |
| PyTorch | `2.14.0+rocm10.1.0` |
| PyTorch HIP runtime | `7.16.26385` |
| torchvision / torchaudio | `0.29.0a0+rocm10.1.0` / `2.11.0.3+rocm10.1.0` |
| XRT | 2.27.0, commit `fdb373b1a43c` |
| NPU firmware | 1.5.5.391 |
| MLIR-AIE / Peano | 1.3.4 / `21.0.0.2026062301+cb664e8c` |

The Fedora HIP packages provide host tools; PyTorch uses its own pinned ROCm SDK wheels. No `HSA_OVERRIDE_GFX_VERSION` is needed. Earlier XRT validation used kernel `7.2.11`; the original timing files did not capture the kernel per run. [Environment snapshot](results/reference/environment.json).

## Setup and run

See [setup](docs/setup.md) for the GPU stack, CK plugin build, conda hooks, XRT, PyXRT, and MLIR loader fix. This records one working machine; it is not a support matrix for every Ryzen APU.

After setup, from the repository root:

```sh
conda activate rocm
python -m pip install --no-deps -e .
python tests/gpu_ops.py
python tests/gpu_models.py
python tests/npu.py
python tests/unified_api.py
python benchmarks/pipeline.py
```

The benchmark writes to `results/local/`. Recorded results stay in `results/reference/`.

```python
import numpy as np
from npu_hybrid import FusedNpuMLP

rng = np.random.default_rng(42)
w1, w2 = [rng.normal(0, 0.04, (512, 512)).astype(np.float32) for _ in range(2)]
x = rng.normal(0, 0.04, (4, 512, 512)).astype(np.float32)
model = FusedNpuMLP(w1, w2)  # backend="sram" keeps the intermediate on-chip
model.prepare_batch(4)       # compile once, before timing
result = model(x)            # NumPy BF16, shape (4, 512, 512)
```

`FusedNpuGpuMLP` splits the stages between NPU and GPU. It uses a host intermediate and two device submissions; the one-launch claim applies to `FusedNpuMLP`. Instances reuse buffers and are not thread-safe.

## License

Original code and documentation: [MIT](LICENSE). Adapted MLIR-AIE files remain Apache-2.0 with LLVM exceptions. MIOpen/Composable Kernel patches retain their upstream MIT notices. See [NOTICE](NOTICE) and [third_party](third_party/). No ROCm, HIP, XRT, or model binaries are redistributed.

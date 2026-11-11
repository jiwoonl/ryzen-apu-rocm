# Full results

`ReLU(A @ B) @ D`, 512×512 BF16 matrices. Weights stay on device; timings include host FP32 input/output conversion and transfers. Each configuration has 300 warm samples, in five rounds of 60. Compilation and initial device loading are excluded. CPU thread count: 16.

Latency is measured for the whole batch. Throughput counts complete input matrices per second. Mean/input is amortized batch time, not the response time of an individual request.

## Pipeline timings

| Path | Batch | Mean (ms) | Median (ms) | p95 (ms) | Mean/input (ms) | Inputs/s |
|---|---:|---:|---:|---:|---:|---:|
| NPU, original tiles | 1 | 1.317 | 0.822 | 3.620 | 1.317 | 759 |
| 780M GPU | 1 | 0.643 | 0.413 | 1.591 | 0.643 | 1,555 |
| NPU, tuned | 1 | 0.632 | 0.613 | 0.764 | 0.632 | 1,583 |
| NPU, SRAM intermediate | 1 | 0.781 | 0.763 | 0.941 | 0.781 | 1,281 |
| NPU, tuned | 2 | 1.130 | 1.094 | 1.343 | 0.565 | 1,770 |
| NPU, SRAM intermediate | 2 | 1.367 | 1.336 | 1.580 | 0.684 | 1,463 |
| NPU, tuned | 4 | 2.217 | 2.161 | 2.579 | 0.554 | 1,804 |
| NPU, SRAM intermediate | 4 | 2.842 | 2.824 | 3.220 | 0.711 | 1,407 |
| NPU, tuned | 8 | 4.582 | 4.560 | 4.945 | 0.573 | 1,746 |
| NPU, SRAM intermediate | 8 | 5.559 | 5.538 | 5.901 | 0.695 | 1,439 |
| NPU, tuned | 16 | 12.441 | 11.611 | 17.289 | 0.778 | 1,286 |
| NPU, SRAM intermediate | 16 | 14.064 | 13.578 | 16.723 | 0.879 | 1,138 |

The tuned shared-memory path was faster than the SRAM path. Batch 4 had the highest measured throughput. GPU median latency was lower than NPU median latency; desktop scheduling affected the averages and tails.

## NPU configuration

| Setting | Tuned shared memory | SRAM intermediate |
|---|---|---|
| Tile dimensions `(m, k, n)` | `(32, 64, 64)` | `(32, 64, 64)` |
| FIFO depth | 2 | 1 |
| DMA row grouping | 4 | One row block per core |
| Intermediate storage | Shared-memory buffer | 32 KiB/core × 16 cores |
| Actual XRT launches per batch | 1 | 1 |

The SRAM implementation removes the intermediate write/read to external memory, but row partitioning increases input/weight traffic. It reuses the same SRAM buffers for successive batch items.

## Verification

- Batches 1, 2, 4, 8, and 16 matched individual executions bit for bit on both NPU backends.
- Actual PyXRT kernel calls were counted: one launch per batch.
- 100 changing-input runs passed, along with invalid-input and autograd rejection checks.
- The 34 validated tuning configurations cover tile dimensions, FIFO depth, and DMA row grouping. Two additional SRAM configurations exceeded resource limits.
- The NPU implementation is fixed-shape inference. On the recorded random test, BF16 output had about 2.1% relative L2 error against FP32. Independent NPU stages with the same reduction tile matched bit for bit.

## Data and reproduction

- [Pipeline data](../results/reference/sram-batch-benchmark-results.json)
- [NPU verification](../results/reference/sram-batch-verification.json)
- [Tuning report](../results/reference/npu-optimization-report.json)
- [Initial sweep](../results/reference/tuning-results.jsonl) and [fine sweep](../results/reference/fine-tuning-results.jsonl)
- [Publication hardware checks](../results/reference/publication-checks.json)
- [Environment](../results/reference/environment.json) and [setup](setup.md)

The environment snapshot was captured at publication. The original timing files did not capture the kernel per run. Earlier measurements in the reference directory belong to separate sessions.

From the repository root, after activating your environment:

```sh
python benchmarks/pipeline.py
```

New measurements go to `results/local/pipeline.json`; the reference data is unchanged.

# HTFT Parallelization & ROCm Migration Plan

This document outlines the concrete steps required to (1) make Hypertensor Field Transformer (HTFT) training fully ROCm-native and (2) introduce meaningful parallelism so Stage 1/Stage 2 work scales across multiple GPUs or accelerators.

## Goals
- Run `examples/train_htft.py` end-to-end on ROCm (MI2xx+/MI3xx) with feature parity relative to CUDA.
- Support faster wall-clock training via pipelined and/or data-parallel execution without rewriting the Hypertensor math core.
- Preserve reproducibility: existing checkpoint format, metrics JSONL, and tests must continue to work.

## Phase 0 — Baseline & Instrumentation
1. **Profile current training loop** on a single GPU (CUDA or ROCm via HIP shim) to capture:
   - Stage 1 vs Stage 2 time, data loading, projection stack cost.
   - Peak memory usage during teacher capture vs HTFT training.
2. **Add lightweight timers/counters** to `HTFTTrainer` so future runs emit per-stage throughput (samples/s), host↔device transfer volumes, Stage 1/Stage 2 loss deltas, and usage counts (min/max/avg) per batch. These diagnostics are required to debug stagnant loss runs.
3. **Confirm deterministic CPU fallbacks** (FAISS off, NumPy only) still pass `agents/test_project.sh`.

## Phase 1 — Native ROCm Enablement
1. **Environment updates**
   - Extend `agents/setup_env.sh` to install ROCm-specific wheels only when `ROCM_AVAILABLE=1`, guarding against accidental installs on unsupported hosts.
   - Document Radeon RX 7900 XTX driver prerequisites plus the existing `.env` variables (`ROCM_VISIBLE_DEVICES`, `HSA_OVERRIDE_GFX_VERSION`, etc.), emphasizing that the helper scripts already source `.env`.
2. **Torch device plumbing**
   - Audit `examples/train_htft.py` and `htfr/data/gemma_adapter.py` to ensure all tensors/models call `.to(device)` where `device` can be `hip`.
   - Replace `torch.cuda.is_available()` checks with `torch.backends.mps`/`hip` aware helper (e.g., `get_preferred_device()`).
3. **Kernel compatibility**
   - Validate that `numba` features used in `feature_ops.py` compile under ROCm; if not, add HIP-friendly fallbacks (pure NumPy, Triton, or ROCm-aware kernels).
   - Confirm FAISS GPU bindings exist for ROCm target; otherwise rely on current CPU fallback and note the limitation.
4. **Smoke tests on ROCm hardware**
- Run `agents/test_project.sh` and a truncated training job (`--max-train-examples 64`) on a Radeon RX 7900 XTX workstation to confirm correctness. Teacher capture remains on GPU, but HTFT updates still execute on CPU for now; capture the host CPU/GPU utilization in the logs.

## Phase 2 — Data & Projection Pipeline Parallelism
1. **Batch projection**
   - Refactor `ContextBuilder` + `ProjectionStack` to process mini-batches efficiently on CPU, minimizing per-sample overhead without moving the sketching kernels off-host. This work explicitly reuses NumPy/Numba; do **not** port CountSketch/SRHT to ROCm yet.
   - Introduce `--batch-size` CLI option; accumulate Stage 1 inputs, run CountSketch/SRHT kernels in batches, and feed Stage 1/Stage 2 updates in mini-batches (still online but grouped). Batches should be prepared by a producer pool (multiple CPU workers) and consumed by a single updater thread to keep tensor mutations serialized.
   - Instrument the batch path to log per-batch statistics: loss mean/std, active tensor counts, and worst-case `|distance|`. This data feeds the loss-debugging effort.
## Phase 3 — Stage-Level Parallelism
1. **Pipeline parallel mode**
   - Assign Stage 1 to device 0 and Stage 2 to device 1 (or shared queue) with overlap: while Stage 2 trains on batch n, Stage 1 processes batch n+1. Remember that Stage 1/Stage 2 updates remain inherently sequential per sample—this phase focuses on overlapping them across devices rather than rewriting the update rules.
   - Implement double-buffered embeddings/tail tensors, using PyTorch `cuda.streams` or ROCm stream equivalents, but keep the KNN/Hypertensor updates on CPU until later phases.
2. **Data parallel HTFR**
   - Wrap Stage 1/2 `HTFRModel` updates with gradient accumulation semantics to support distributed data parallel (DDP) over multiple ROCm GPUs, even though the tensor updates still originate on CPU. This likely requires sharding Hypertensors per rank to avoid write conflicts.
   - Because HTFR updates tensors in place, design a synchronization protocol (e.g., sharded tensors per rank with periodic all-reduce of control matrices). Explicitly call out that the FAISS/CPU KNN path stays in place; GPU KNN is out of scope.

## Phase 4 — Distribution & Checkpointing
1. **Sharded checkpoint support**
   - Extend `htfr.checkpoint.save_htft_checkpoint` to optionally write per-rank tensor shards or include device metadata indicating how tensors were partitioned.
2. **Resumable pipelines**
   - Store pipeline topology (`stage_assignments`, `batch_size`, `devices`, `producer_worker_count`) in checkpoint metadata so resumed runs reconstruct the same parallel layout and CPU worker configuration.

## Phase 5 — Validation & Tooling
1. **Functional tests**
   - Add ROCm-only CI job (self-hosted runner) executing: `agents/setup_env.sh` with ROCm extras, `agents/test_project.sh`, and a short pipelined training run (e.g., `--batch-size 32 --max-train-examples 256`). CI should capture both trainer diagnostics and ROCm system info.
2. **Performance regression harness**
   - Provide `agents/benchmark_htft.sh --profile profiles/latest.json` option that captures samples/s, GPU utilization, and loss curves for comparison across commits.
3. **Documentation**
   - Update README + AGENTS docs with new CLI flags (`--batch-size`, `--pipeline`), ROCm troubleshooting, and hardware recommendations.
4. **Human eval harness**
   - Add a `human_eval.sh` script (if missing) that enumerates the manual ROCm/GPU tests maintainers must run when sandboxed environments can’t execute GPU jobs. Include commands for: running `agents/setup_env.sh` with ROCm extras, launching a short `train_htft.py` job with `--batch-size` enabled, verifying diagnostics (loss trends, active tensor stats), and confirming teacher capture/device info on the Radeon RX 7900 XTX host.

## Deliverables & Ownership
| Deliverable | Owner | Success Criteria |
| --- | --- | --- |
| ROCm environment recipe | Infra | `agents/setup_env.sh` completes on 7900 XTX host, README updated |
| Device-aware trainer | Core | Stage 1/2 run on configurable devices, CLI docs in README |
| Batch/pipeline execution | Core | `--batch-size > 1` supported, throughput improves ≥1.5× on dual GPUs |
| Distributed checkpoints | Core+Infra | Multi-rank checkpoint can be saved/loaded without loss |
| ROCm CI job | Infra | Nightly ROCm pipeline green, captures metrics artifacts |

## Risks & Mitigations
- **FAISS GPU gap on ROCm:** fall back to CPU index; consider HNSWlib or custom ROCm kernel if perf suffers.
- **Numba incompatibility:** gate numba-accelerated SRHT behind feature flag; ship prebuilt kernels via Triton or HIP.
- **In-place HTFR updates vs DDP:** consider coarse-grained parameter server approach or limit DDP to Stage 2 where logits regression benefits most.
- **Memory pressure from batched projections:** expose knobs for `--stage*-srht-dim` and per-batch size to avoid OOM on smaller MI-series GPUs.

## Next Steps
1. Secure ROCm hardware access and create a branch dedicated to ROCm + parallelism.
2. Implement Phase 1 tasks, land incremental PRs with ROCm toggles guarded by feature flags.
3. Once ROCm parity is validated, proceed with Phase 2 batching and benchmark improvements before tackling pipeline/data parallelism.

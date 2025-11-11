# HyperTensor Field Regressor Update Plan

This document captures the architectural changes required to extend the
HyperTensor Field Regressor (HTFR) — the core building block of the
HyperField Transformer (HFT) — so it can regress over massively expanded
input vectors representing full LLM contexts.

## 1. Input Representation
- Treat each training example as the *entire* attention window feeding a
  single token prediction. Concatenate token IDs/embeddings, per-layer
  hidden states, per-head Q/K/V streams, ROPE phases, and any auxiliary
  metadata into one raw vector before projection.
- Apply a multi-stage projection stack to keep the working dimension
  tractable while preserving locality:
  1. Sparse CountSketch over the raw concatenation.
  2. Block-SRHT (`htfr.feature_ops.apply_block_srht`) down to the target
     dimension (set to 4k after aggressive compression).
  3. Optional lightweight PCA/whitening layer learned after the first
     epoch of training.
- Append sparse hashed indicators (n-gram Bloom filters, token-order
  hashes) ahead of the projection so distinct token permutations occupy
  different regions in feature space.
- Maintain multiple independent SRHT views (e.g., two separate 4k-dim
  projections) and concatenate them if the compression budget allows.
  This keeps each view small while modestly increasing information
  capacity.

## 2. Geometry & Locality
- Keep the existing HyperTensor primitive untouched—one normal vector,
  single offset, and per-band control columns. The expanded projected
  vector already captures the full context, so no multi-axis normals are
  needed.
- Set the working dimension to `d = 4096` after the projection stack and
  store every HyperTensor parameter, optimizer state, and feature cache
  in float16. Memory per tensor is roughly
  `2 × (d + 3 × vocab_limit)` bytes; with `vocab_limit = 4096`, an 8 GB
  budget supports ~200k tensors while still reserving slack for queues,
  Stage-1 caches, and diagnostics. Keep the active population near 120k
  to leave headroom for relocation and logging buffers.
- Use ANN or pre-filtered candidate sets when `N` grows past ~20k to
  keep top-`K` selection latency stable.

## 3. Initialization & Adaptation
- Allow purely random initialization (Gaussian normals + scaled control
  matrices) in addition to the existing k-means seeding so thousands of
  tensors can be provisioned instantly, mirroring NN weight init.
- Track usage/error statistics per tensor. Maintain a queue of high-loss
  contexts; periodically reassign “dead” tensors (low usage or low
  gradient norm) by reseeding them around queued contexts. This keeps
  coverage adaptive without storing the full dataset.
- Cache projected feature shards on disk if replay buffers are needed;
  HTFR itself continues to store only tensor parameters.

## 4. Training Loop Updates
- Extend data collection to emit the full-context vectors plus targets.
- Run the projection stack once per context and cache the result for
  overlapping sliding windows.
- During updates, log active tensor counts, loss contributions, and
  relocation events for diagnostics.
- For two-stage setups (compression + prediction), allow gradients to
  flow through both HTFR instances so the compression stage learns to
  highlight the most predictive context components.

## 6. Precision & Storage
- Enforce float16 everywhere: HyperTensor normals, control columns,
  SRHT/CountSketch matrices, relocation statistics, replay caches,
  optimizer momentum, and serialized checkpoints. Keep float32 scratch
  buffers only when algorithms (e.g., variance logs) require it, casting
  results back to float16 before storage.
- Update serialization to round-trip float16 tensors and include dtype
  metadata so checkpoints default to the new precision.
- Ensure preprocessing kernels (CountSketch, SRHT, PCA) accept float16
  inputs while performing accumulation in float32 to avoid numerical
  collapse, then downcast the final features before feeding them to the
  HyperField Transformer.

Implementing the above keeps the HTFR core simple (no new tensor
primitive) while scaling the feature space, initialization, and adaptive
coverage mechanisms to match LLM-sized inputs.

## 5. Information Flow & Verification
- **Embeddings/hashed indicators → SRHT vectors.** Verify CountSketch and
  SRHT preserve variance by logging per-dimension variance and cosine
  similarities between raw and projected vectors on a validation batch.
  A drop >10% signals the projection dimension or hash size must grow.
- **Projected vectors → tensor activations.** Track which inputs activate
  each HyperTensor (average |distance|, activation frequency). Use
  coverage histograms to confirm new contexts trigger previously
  inactive tensors; otherwise the relocation queue should reseed them.
- **HyperTensor controls → logits.** Reconstruct teacher logits for a
  held-out buffer and compare via KL divergence. Monitoring KL per input
  class shows whether the control matrices encode the necessary signal.
- **Two-stage coupling.** When a compression HTFR feeds the predictor,
  measure mutual information between Stage-1 outputs and teacher logits
  using held-out data (approximate via regression R²). If MI degrades,
  Stage 1 needs more tensors or widened projections.
- **Adaptive relocation efficacy.** After moving tensors from the
  high-error queue, track the post-relocation loss on the triggering
  contexts. A positive delta indicates the relocation succeeded in
  capturing the missing information.

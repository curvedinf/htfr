# HyperField Transformer for LLM Token Prediction

This note explains how to adapt HTFR into the HyperField Transformer
(HFT), a full transformer replacement stack that regresses directly on
the inputs and outputs of a language model.

## 1. Data Capture
- Run the teacher LLM with `output_hidden_states=True` and hook the
  per-layer, per-head attention inputs (queries, keys, values) plus the
  ROPE phases and token IDs for every position inside the causal window.
- For each prediction step, build a single raw context vector by
  concatenating:
  - Token embeddings (or one-hot codes) for the entire sliding window.
  - Layer-wise hidden states for that window.
  - Per-head Q/K/V tensors aligned to their source tokens.
  - Positional metadata (ROPE sin/cos pairs, attention masks, stride).
  - Optional hashed n-gram/Bloom-filter indicators describing token
    permutations.
- Associate every context vector with the teacher’s next-token logits
  (possibly truncated to a shortlist, as in the Gemma benchmark).

## 2. Feature Compression Pipeline
- Immediately project each raw vector through a multistage stack:
  1. CountSketch or other sparse hashing to mix heavy-tailed discrete
     features.
  2. Block-SRHT that reduces the raw Stage-1 input to a 16k-dimensional
     workspace (use two independent 16k views if memory allows). Stage 1
     consumes this 16k vector directly.
  3. Optional learned PCA/whitening layer once enough samples are
     available.
- Persist the projected vectors (not the raw contexts) if a replay
  buffer is required; HTFR checkpoints themselves only store tensor
  parameters.

## 3. Two-Stage HyperField Transformer
- Stage 1 (Attention/Compression):
  - Input: raw full-context vector.
  - Output: a compact 1k-dimensional context embedding expressed as
    float16 features plus auxiliary sparsity masks for diagnostics.
  - Supervision: teacher attention maps + token-prediction loss.
- Stage 2 (Token Prediction):
  - Input: the 1k Stage-1 embedding, the current token metadata, and a
    direct injection of the last 16 token embeddings (concatenated or
    block-hashed) to keep the immediate history uncompressed. A small
    CountSketch + SRHT projects this combined vector back to the 4k
    Stage-2 workspace referenced in the architecture guide.
  - Output: next-token logits.
  - Training: pretrain on teacher logits, then fine-tune end-to-end so
    gradients flow through Stage 1.
- Benefits: Stage 1 learns to drop low-importance tokens before the
  large HTFR sees them, reducing dimensionality while staying within the
  HTFR framework.

## 4. Initial HyperField Transformer Parameters
- **Information complexity.** Stage 1 must encode head-wise attention
  structure (~12 heads × 4 layers × 128 positions) plus hashed
  permutations; empirically this demands ~40 M effective degrees of
  freedom, so we allocate 8k HyperTensors with `top_k=16` and a learning
  rate pair (`eta=0.04`, `eta_g=0.004`). Stage 2 models the compressed
  embedding plus the 16-token tail; this mixture captures ~65 M
  degrees of freedom because it must resolve 4k-class logits, so we
  start with 16k HyperTensors, `top_k=24`, `eta=0.03`, and `eta_g=0.003`.
- **Perplexity contribution.** When staged against the Gemma teacher,
  Stage 1 alone reduces perplexity by ~30% (by reconstructing attention
  heat maps) while Stage 2 closes the remaining 70%. Initialize Stage 2
  with slightly lower taus (`tau=0.8`) to emphasize sharper locality for
  the logits, whereas Stage 1 can keep `tau=1.0` to explore a wider
  manifold. Track teacher-vs-student perplexity per stage to confirm
  these ratios hold; adjust tensor counts or taus if either stage stalls.

## 5. Model Capacity & Memory
- With projected dimension `d=4096` and a 4k-token compact vocab, one
  HyperTensor consumes roughly 65 kB. An 8 GB budget therefore supports
  about 120k tensors while still leaving room for projection matrices
  and optimizer state.
- Randomly initialize tensors (Gaussian normals + scaled control
  matrices) to mimic NN-style weight init, then rely on online updates to
  specialize them to observed contexts.
- Maintain a “high-error queue” of contexts; periodically reseed idle
  tensors around queued samples to maintain coverage across rare token
  patterns.

## 6. Training Procedure
1. Stream teacher data to build (context vector, next-token logits)
   pairs, applying the projection stack on the fly.
2. Train Stage 1 (if used) and Stage 2 HTFRs with online updates; log
   activation stats, loss, and relocation events for monitoring.
3. Periodically evaluate on held-out teacher outputs to ensure the
   compressed features retain enough information for accurate token
   regression.
4. For deployment, run inference by executing Stage 1 (if present) and
   Stage 2 per token; only HyperTensor parameters reside in memory, and
   no raw training data is retained.

## 7. Inference Workflow
- Maintain a rolling context buffer containing the latest `W` tokens,
  cached embeddings, layer states, and Q/K/V tensors. Update the buffer
  incrementally as new tokens arrive instead of recomputing the entire
  sliding window.
- When predicting token `t+1`, assemble the raw context vector by
  reading the buffer (plus the current token embedding) and appending
  hashed indicators; immediately pass it through the CountSketch →
  SRHT → (optional) PCA pipeline to reach the 4k-dimensional workspace.
- Feed the compressed vector into the Stage-1 HTFR (if used) to obtain
  attention masks or the compact embedding, then invoke the Stage-2 HTFR
  to emit logits. Because both stages are local mixtures, latency scales
  with the configured `top_k` rather than the original context length.
- After sampling or decoding the next token, update the buffer with the
  new token’s embeddings and any auxiliary signals so the next inference
  step reuses cached data.

## 8. Input Sources at Inference
1. **Token IDs.** Generated autoregressively; the tokenizer already
   exists locally (`htfr/data/gemma_adapter.py`). No teacher is needed—when a
   new token is sampled, its ID is known immediately.
2. **Token embeddings.** Reuse the frozen embedding matrix from the
   distilled vocabulary (`README.md:139-167`). Lookup is a simple table
   index on the token ID, so it can be computed on the fly.
3. **ROPE phases / positional encodings.** ROPE is deterministic given
   the token index and head dimension. Use the same sinusoidal formulas
   Gemma uses (documented in Hugging Face’s transformer stack) to
   synthesize the sin/cos pairs per head without querying the teacher.
4. **Attention masks / stride metadata.** These are fixed functions of
  the sliding window length (`seq_len`, `stride` in
  `examples/train_hft.py`). During inference, maintain the
   same stride and window so the mask can be regenerated every step.
5. **Hashed n-gram indicators.** Construct Bloom filters or hashed
   shingles directly from the current context buffer; the hash functions
   do not depend on the teacher outputs.
6. **Per-head Q/K/V approximations.**
   - During training, Stage 1 HTFR learns to reproduce the teacher’s
     attention signals (queries, keys, values or their compressed
     equivalents) from the raw concatenated context.
   - At inference, run Stage 1 on the buffered tokens/embeddings to
     produce synthetic Q/K/V tensors. Because Stage 1 holds only
     HyperTensor parameters, it replaces the teacher’s internal blocks.
7. **Layer-wise summaries.** If the deployment keeps a thin subset of
   the original transformer (e.g., embeddings + the first layer), those
   components generate the partial hidden states before HTFR consumes
   them. Otherwise, Stage 1’s outputs stand in for every layer.
8. **Projected context vectors.** After assembling the items above,
   apply the CountSketch→SRHT→PCA pipeline: Stage 1 consumes a 16k SRHT
   projection, emits a 1k embedding, and Stage 2 concatenates that
   embedding with the 16-token tail before a final 4k SRHT projection
   (`notes/htfr_architecture_update.md:9-27`). These projections are
   deterministic given the buffered features, so inference needs only
   the stored projection matrices.
9. **Offline caches.** For batch inference, precompute and store the
   projected context vectors generated by this process. Because they are
   derived solely from local token streams plus Stage 1, no teacher
   activations are required at runtime.

## 9. Information Flow Verification
- **Token IDs → embeddings/logits.** Run ablation tests that zero out
  specific tokens in the context buffer and confirm the predicted logits
  shift toward the UNK class. Stability here shows embeddings are
  flowing through the projection and tensor blend correctly.
- **ROPE phases / positional encodings.** Log the model’s perplexity as
  a function of absolute position. Flat curves after shuffling phases
  indicate positional information is being lost; in that case, increase
  the SRHT dimension or add dedicated positional indicators.
- **Attention masks / stride metadata.** Inject invalid tokens beyond
  the causal window during evaluation; the logits should remain
  unchanged if masks are respected. Any drift implies Stage 1 or Stage 2
  ignored the mask bits.
- **Hashed n-gram indicators.** Track collision rates by counting how
  often distinct n-grams map to identical hash buckets, then correlate
  with regression error. If high-collision buckets show elevated loss,
  expand the hash size or add a second hash function.
- **Stage 1 Q/K/V proxies.** Measure reconstruction error between Stage 1
  outputs and the teacher’s true attention tensors on a held-out set.
  Keep the mean cosine similarity above 0.9 so Stage 2 receives faithful
  signals; additionally, verify the 1k embedding retains ≥95% of the
  teacher perplexity reduction relative to raw SRHT features.
- **Stage coupling.** When training end-to-end, backpropagate a probe
  loss from Stage 2 to Stage 1 and ensure gradients remain non-zero. If
  they vanish, Stage 1 stops conveying useful information and needs more
  capacity. Monitor the KL improvement contributed by the appended 16-token
  tail by toggling it off during evaluation; a ≥10% KL rise confirms the
  tail is being used.
- **Projected vectors → logits.** Periodically decode using cached
  projected vectors and compare logits against fresh projections of the
  same contexts; mismatches reveal drift in the projection matrices or
  HyperTensor parameters.

Following this recipe lets HTFR operate directly on the full information
that feeds transformer token predictions while keeping inference fast
and memory usage bounded.

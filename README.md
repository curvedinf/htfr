# Hypertensor Field Regressor (HTFR)

HTFR is a research stack for building Hypertensor fields—collections of oriented, piecewise-linear regressors that emulate transformer-quality predictions with a fraction of the compute. The latest iteration ships a two-stage Hypertensor Field Transformer (HTFT) that consumes hidden states from a frozen teacher such as Gemma 3 270M, compresses them through fast sketching operators, and regresses truncated logits for next-token prediction.

## Highlights
- **Hypertensor primitives** (`htfr/hypertensor.py`): differentiable local regressors that blend geometric locality (signed distances to hyperplanes) with functional controls for smooth interpolation.
- **HTFR model core** (`htfr/model.py`): manages top‑K neighbor lookup (FAISS when available), adaptive interpolation mixes, online updates, and relocation logic for underused tensors.
- **HTFT pipeline** (`htfr/hypertensor_field_transformer.py`, `htfr/trainer.py`): Stage 1 learns a compact embedding supervised by the teacher; Stage 2 consumes that embedding plus a raw token tail to regress truncated logits.
- **Projection stack** (`htfr/feature_ops.py`): CountSketch + SRHT towers with optional block RMSNorm, enabling 100k+ dimensional inputs while keeping Stage 1/2 memory bounded.
- **Context builder + Gemma adapter** (`htfr/context.py`, `htfr/data/gemma_adapter.py`): streams Hugging Face datasets, captures teacher hidden states/logits, injects hashed n‑grams and ROPE phases, and produces weighted samples for the trainer.
- **Automation scripts** (`agents/*.sh`): reproducible environment bootstrap, end-to-end training helper, benchmark summarizer, and test runner.

## Architecture Overview
1. **Teacher capture.** `examples/train_htft.py` authenticates with Hugging Face (optional token), loads the specified causal LM, and streams dataset windows via `GemmaConfig`. Each window yields token IDs, final-layer hidden states, and optional logits.
2. **Context assembly.** `ContextBuilder` flattens the hidden-state window, appends hashed n‑gram indicators plus synthetic or teacher-provided ROPE phases, and produces (a) Stage 1 inputs, (b) tail embeddings for Stage 2, and (c) a projected Stage 1 regression target.
3. **Projection + regression.** `ProjectionStack` maps the raw context through CountSketch (if configured) and one or more SRHT blocks before feeding Hypertensor stages. Stage 1 outputs a 1 k‑dimensional embedding. Stage 2 concatenates that embedding with the raw tail (e.g., 16 tokens × hidden size) and predicts truncated next-token logits.
4. **Training loop.** `HTFTTrainer` iterates ContextSamples, computes cross-entropy on compact logits plus an MSE auxiliary loss for Stage 1, and logs perplexity/throughput every few seconds. Metrics can be streamed to JSONL for later benchmarking.
5. **Checkpointing.** `htfr.checkpoint` persists both stages (Hypertensor tensors, SRHT/CountSketch params, metadata) alongside the compact vocabulary map so training can resume or the model can be shared.

## Repository Layout
| Path | Description |
| --- | --- |
| `htfr/` | Core package (Hypertensor math, projection ops, trainer, checkpointing, FAISS helpers). |
| `htfr/data/gemma_adapter.py` | Teacher/dataset utilities for Gemma-class models and WikiText-style corpora. |
| `examples/train_htft.py` | Reference CLI for collecting teacher signals and training the two-stage HTFT. |
| `examples/benchmark_htft.py` | Summarize JSONL metrics emitted by the trainer (best/final perplexity, teacher ratios). |
| `agents/` | Automation helpers (`setup_env.sh`, `train_htft.sh`, `benchmark_htft.sh`, `test_project.sh`). |
| `tests/` | Pytest suite covering Hypertensor math, projection stacks, checkpoint I/O, and trainer glue. |

## Environment Setup
1. Use Python 3.10+ with system packages required by PyTorch/ROCm for your platform.
2. Populate `.env` with any secrets or GPU configuration you want inherited by the helper scripts (e.g., `HF_TOKEN`, `ROCM_VISIBLE_DEVICES`).
3. Run the provisioning script from the repo root:
   ```bash
   agents/setup_env.sh
   ```
   - Installs the project in editable mode with `benchmark`, `dev`, and `rocm` extras (override via `HTFR_SETUP_EXTRAS`).
   - Optionally reinstall the nightly ROCm wheel set when `HTFR_INSTALL_ROCM_TORCH=1` (default). Set `HTFR_ROCM_INDEX_URL` to point at a different PyTorch package mirror or disable the flag to keep CPU/CUDA wheels.
4. Activate `.venv/bin/activate` when working manually, or rely on the helper scripts which activate it for you.

## Training the Hypertensor Field Transformer
The main entry point is `examples/train_htft.py`. It orchestrates teacher capture, context generation, trainer loops, checkpoint saves, and optional metrics logging.

### Minimal launch
```bash
python examples/train_htft.py \
  --hf-token "$HF_TOKEN" \
  --model google/gemma-3-270m \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --train-tokens 200000 \
  --eval-tokens 50000 \
  --seq-len 128 \
  --stride 64 \
  --output checkpoints/htft_gemma270m.npz \
  --metrics-path logs/train_metrics.jsonl
```

### Key switches
- `--stage1-tensors / --stage2-tensors`: number of Hypertensors allocated per stage (defaults 8k / 16k).
- `--stage1-countsketch-dim` / `--stage2-countsketch-dim`: CountSketch width before SRHT; set to zero to bypass sketching.
- `--stage1-srht-dim` / `--stage2-srht-dim`: target dimensionality for SRHT projections.
- `--stage1-target-dim`: embedding dimension produced by Stage 1 and consumed by Stage 2.
- `--tail-tokens`: number of trailing tokens whose raw embeddings are concatenated onto the Stage 2 input.
- `--vocab-limit`: size of the compact shortlist used for truncated logits; the `build_vocab_mapping` helper maps full-token IDs into this range plus the `unk_index`.
- `--train-top-k / --stage*-tau / --stage*-eta / --stage*-eta-g`: optimizer knobs forwarded to each Hypertensor stage.
- `--max-train-examples` / `--max-eval-examples`: cap on teacher windows obtained from each stream to avoid exhausting VRAM.

All CLI options are documented via `python examples/train_htft.py --help`; run this command after editing the script to ensure argument parsing still works.

## Monitoring and Benchmarking
When `--metrics-path` is provided, every evaluation pass appends a JSON line describing step counts, train/eval perplexity, and (optionally) the truncated teacher perplexity baseline. Summaries are available via:
```bash
agents/benchmark_htft.sh logs/train_metrics.jsonl
```
The script reports best vs. final student perplexity and the student/teacher gap so you can compare runs without reloading massive checkpoints.

## Checkpoints and Resuming
Checkpoints saved by `save_htft_checkpoint` are self-contained `.npz` archives.

```python
from htfr.checkpoint import load_htft_checkpoint, HTFTCheckpoint

ckpt: HTFTCheckpoint = load_htft_checkpoint("checkpoints/htft_gemma270m.npz")
print(ckpt.stage1.model.output_dim, ckpt.stage2.metadata.get("vocab_limit"))
```

The checkpoint stores:
- Both stages’ Hypertensors, optimizer hyperparameters, SRHT/CountSketch parameters, and metadata.
- The compact vocabulary shortlist plus `unk_index`, so truncated logits can be mapped back to full token IDs.
- Tail configuration (token count, embedding size) ensuring inference scripts rebuild Stage 2 inputs correctly.

You can reconstruct an `HypertensorFieldTransformer` from a checkpoint by creating new `ProjectionStack` instances with the serialized SRHT/CountSketch parameters and reusing the saved `HTFRModel` instances.

## Testing and Quality Gates
- **Syntax check:** `python -m compileall htfr examples`
- **Pytest + coverage:** `agents/test_project.sh` (wraps compileall + `pytest --cov` and writes `coverage.xml`).
- **CLI smoke tests:** run `python examples/train_htft.py --help` and `python examples/benchmark_htft.py --help` after editing argument definitions to ensure argparse wiring is intact.
- **Style:** follow PEP 8 and keep logging concise—training output is streamed every few seconds, so favor short single-line messages.

## Extending the Project
- **Different teachers or datasets:** model-agnostic adapters can follow the `TeacherWindowLike` protocol in `htfr/context.py`. Start by duplicating `htfr/data/gemma_adapter.py`, swap in your tokenizer/model, and make sure to emit ROPE phases or allow the builder to synthesize them.
- **Alternate projection stacks:** `ProjectionStack` accepts arbitrary SRHT parameter lists and optional `CountSketchParameters`. Use `make_block_srht`/`make_count_sketch` to explore other dimensionalities or block sizes, then persist them through `StageState` metadata for reproducibility.
- **Inference-only deployments:** When training completes, call `HypertensorFieldTransformer.diagnostics()` to snapshot usage/loss stats before freezing the tensors.
- **FAISS tuning:** The neighbor search wrapper automatically falls back to NumPy when FAISS is unavailable. Adjust `FaissIndex` parameters (M, efConstruction/efSearch) through the `HTFRModel` constructor for larger tensor banks.

## Dependencies & Extras
Core requirements live in `pyproject.toml` (NumPy, Numba, SymPy, python-dotenv). Optional extras:
- `benchmark`: datasets, transformers, torch, huggingface_hub—needed for teacher capture and benchmarking.
- `dev`: pytest, pytest-cov.
- `rocm`: ROCm-flavored PyTorch/Accelerate/Fabric/Optimum stack plus utilities for AMD GPUs.

Use pip extras (e.g., `pip install -e .[benchmark,dev]`) to tailor installations for local vs. production environments.

## License
HTFR is released under the MIT License (see `LICENSE`). Contributions are welcome—please follow the guidelines in `AGENTS.md` for workflow details.

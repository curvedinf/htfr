# Hypertensor Field Regressor (HTFR)

## Abstract

The Hypertensor Field Regressor (HTFR) is a fast, differentiable regression framework for very high-dimensional inputs. Its primitive, the HyperTensor, couples a geometric component (an oriented hyperplane) with a functional component (a local interpolation operator) to form a smooth, piecewise-linear field. A prediction is a locality-weighted blend of a small set of HyperTensors. HTFR supports online, error-driven learning (backprop-style), KNN-style locality, and scales to hundreds of thousands of input dimensions with structured projections. It targets tasks where large neural networks are accurate but too slow or heavy, such as token-wise next-logit regression from LLM internal features.

---

## HyperField Transformer Pipeline

The repository now focuses on the two-stage HyperField Transformer
(HFT) described in `notes/htfr_llm_adaptation.md`. The new workflow
replaces the legacy single-stage scripts and provides a reusable trainer
plus a simple benchmarking harness.

### Train the HFT against Gemma 3 270M

```
python examples/train_hft.py \
    --hf-token "$HF_TOKEN" \
    --model google/gemma-3-270m \
    --train-tokens 200000 \
    --eval-tokens 50000 \
    --seq-len 128 \
    --stride 64 \
    --output checkpoints/hft_gemma270m.npz \
    --metrics-path logs/train_metrics.jsonl
```

Key features:

- **Full-context capture.** The `GemmaAdapter` streams sliding windows
  from WikiText (or your dataset of choice) while emitting the hidden
  states, logits, and next-token targets required by the HFT pipeline.
- **Context builder.** `htfr.context.ContextBuilder` concatenates the
  hidden states, hashed n-gram indicators, and synthetic ROPE phases
  before projecting them through CountSketch + SRHT stacks to reach the
  16k → 1k → 4k layout described in the notes.
- **Two-stage training.** Stage 1 regresses a 1k context embedding
  (supervised by a projected teacher hidden state) while Stage 2 predicts
  truncated next-token logits from that embedding plus the uncompressed
  16-token tail. Both stages default to `top_k=32` active HyperTensors to
  match the Gemma-3-270M capacity budget.
- **Checkpointing.** `htfr.checkpoint.save_hft_checkpoint` persists both
  stages, their projection stacks, and the compact vocabulary mapping so
  training can resume or the model can be shared.
- **Metrics.** The trainer logs student and teacher perplexity into the
  optional JSONL file so you can track convergence without rerunning the
  teacher. Use `examples/benchmark_hft.py logs/train_metrics.jsonl` to
  summarize the trend once training finishes.

The defaults allocate ~8k Stage‑1 tensors and ~16k Stage‑2 tensors
(float16 parameters) which keeps memory usage comparable to Gemma
3 × 270 M while providing enough locality to hit competitive perplexity
once trained.

---

## 1. HyperTensor Primitive

- **Domain/Range.** Input \(x\in\mathbb{R}^D\), output \(y\in\mathbb{R}^M\).
- **Geometry.**
  - Unit normal \(n\in\mathbb{R}^D\) (\(\lVert n\rVert = 1\)), offset \(\delta\).
  - Signed distance \(d(x) = n^\top x + \delta\).
  - Reference distances \(d_{\text{neg}} < 0 < d_{\text{pos}}\) define the interpolation band; clip \(d'(x) = \mathrm{clip}(d(x), d_{\text{neg}}, d_{\text{pos}})\).
- **Functional controls.**
  - Control matrix \(C \in \mathbb{R}^{M\times 3}\).
  - Piecewise-linear barycentric coefficients:
    - If \(d' \ge 0\): \(\alpha = (0, 1-a_{\text{pos}}, a_{\text{pos}})\) with \(a_{\text{pos}} = d' / d_{\text{pos}}\).
    - If \(d' < 0\): \(\alpha = (a_{\text{neg}}, 1-a_{\text{neg}}, 0)\) with \(a_{\text{neg}} = -d' / d_{\text{neg}}\).
  - Local interpolant: \(L(x) = C\,\alpha\).

**Definition.** A HyperTensor is \((n, \delta, d_{\text{neg}}, d_{\text{pos}}, C)\). It maps \(x\mapsto L(x)\) with continuous (C⁰) output and constant gradient within each half-band.

---

## 2. HTFR Model

Given HyperTensors \(\{T_i\}_{i=1}^N\), define distances \(d_i(x)\) and local interpolants \(L_i(x)\).

- **Locality weights.** Top-\(K\) selection by smallest \(|d_i(x)|\) (with \(K\)). Weights over the active set:
  - Softmax: \(w_i(x) = \frac{\exp(-|d_i(x)|/\tau_i)}{\sum_{j\in\mathcal{A}(x)} \exp(-|d_j(x)|/\tau_j)}\).
  - Inverse-distance: \(w_i(x) = \frac{(|d_i(x)|+\varepsilon)^{-1}}{\sum_{j\in\mathcal{A}(x)}(|d_j(x)|+\varepsilon)^{-1}}\).
- **Prediction.** \(\hat y = f(x) = \sum_{i\in\mathcal{A}(x)} w_i(x)\,L_i(x)\).

This is a HyperTensor field: a smooth blend of oriented, local, 1-D interpolators embedded in \(\mathbb{R}^D\).

---

## 3. Learning

### 3.1 Loss

- Regression: \(\mathcal{L}(y, \hat y) = \frac{1}{2}\lVert y-\hat y\rVert^2\).
- Classification (logits): cross-entropy over logits.

### 3.2 Backprop-style updates (per sample)

Let \(g = \partial \mathcal{L} / \partial \hat y\) for MSE, or derived from a general loss (e.g., softmax-CE). Write \(d_i'\) for clipped distances.

1. **Controls** \(C_i\) (columnwise):
   \[\boxed{\Delta C_i = -\eta\; w_i(x)\; g\; \alpha_i(d_i')^\top}\]
2. **Geometry via distance sensitivity.** Inside bands:
   \[\frac{\partial L_i}{\partial d_i} =
     \begin{cases}
       \frac{V_{\text{pos},i} - V_{0,i}}{d_{\text{pos},i}}, & d_i\in(0, d_{\text{pos},i}) \\
       \frac{V_{\text{neg},i} - V_{0,i}}{d_{\text{neg},i}}, & d_i\in(d_{\text{neg},i}, 0)
     \end{cases}
   \]
   \[\boxed{\Delta n_i = -\eta_g\; w_i(x)\; (g^\top \tfrac{\partial L_i}{\partial d_i})\; x,\quad
   \Delta \delta_i = -\eta_g\; w_i(x)\; (g^\top \tfrac{\partial L_i}{\partial d_i})}\]
3. **Weights** \(w_i\) (optional). Softmax over \(-|d|\) gives:
   \[\frac{\partial \hat y}{\partial w_i} = L_i,\quad
   \Delta w_i \propto -g^\top (L_i - \hat y).\]
4. **Reference bands** (optional slow adaptation).
   \[
     d_{\text{pos},i} \leftarrow (1-\gamma)d_{\text{pos},i} + \gamma\max(0, d_i),\quad
     d_{\text{neg},i} \leftarrow (1-\gamma)d_{\text{neg},i} + \gamma\min(0, d_i).
   \]

Learning rates: \(\eta, \eta_g\). Clip \(n_i\) and gradients for stability.

---

## 4. Initialization

- **Clustering:** k-means on a sample of inputs; for cluster \(i\), set \(n_i\) to the first principal direction, \(\delta_i\) to the hyperplane through the cluster mean.
- **Bands:** \(d_{\text{pos}}, d_{\text{neg}}\) from projected standard deviation.
- **Controls:** \(C_i\) as global or local cluster mean.
- **Locality temperature:** \(\tau\) tuned on a warmup buffer.

---

## 5. Complexity & Scale

Per query with top-\(K\):

- Distances: \(\mathcal{O}(K D)\) if using a preselected candidate set; \(\mathcal{O}(N D)\) naive.
- Interpolants & blend: \(\mathcal{O}(K M)\).
- Updates: same as inference plus \(\mathcal{O}(K D)\).

Large-\(D\) compatibility. Use a fast structured projection \(P\) (SRHT/CountSketch), learn HyperTensors in \(P x\)-space (\(d \ll D\)). All formulas hold with \(x\) replaced by \(P x\).

---

## 6. Properties

- **Continuity:** \(f(x)\) is continuous; gradients are piecewise constant inside bands; kinks only at band edges.
- **Locality:** Only the top-\(K\) closest HyperTensors are active; predictable latency.
- **Interpretability:** Each HyperTensor exposes a direction \(n\), a position \(\delta\), and a local response profile \(C\).
- **Expressivity:** A finite blend of oriented, piecewise-linear local maps can approximate continuous functions on compact sets to arbitrary precision (via sufficient coverage and control resolution).
- **Regularization:** L2 on \(n\), band width priors, sparsity on active counts.

---

## 7. Variants

- **KNN-HTFR:** select \(\mathcal{A}(x)\) by KNN in \(d\) (fast, simple).
- **Kernel-HTFR:** alternate weighting (e.g., triangular, Epanechnikov).
- **C¹ HyperTensors:** replace linear with cubic Hermite along \(d\) for continuous gradients.
- **Vector-valued geometry:** allow multiple normals per unit (multi-axis interpolation).
- **Shared decoder:** for very large \(M\) (e.g., vocab logits), map \(L_i\) then decode with shared head.

---

## 8. Online Algorithm (Inference + Update)

```python
import numpy as np
from dataclasses import dataclass


@dataclass
class LocalResult:
    value: np.ndarray
    distance: float
    weights: np.ndarray
    clipped_distance: float
    distance_derivative: np.ndarray

class HyperTensor:
    def __init__(self, n, delta, dneg, dpos, C, tau=1.0):
        self.n = n / (np.linalg.norm(n) + 1e-8)
        self.delta = float(delta)
        self.dneg, self.dpos = float(dneg), float(dpos)
        self.C = C.astype(np.float32)            # d x 3
        self.tau = float(tau)

    def _alpha(self, d):
        dcl = np.clip(d, self.dneg, self.dpos)
        if dcl >= 0:
            a_pos = dcl / (self.dpos + 1e-12)
            return np.array([0.0, 1.0 - a_pos, a_pos], dtype=np.float32), dcl
        else:
            a_neg = -dcl / (self.dneg - 1e-12)   # dneg<0
            return np.array([a_neg, 1.0 - a_neg, 0.0], dtype=np.float32), dcl

    def local(self, x):
        d = float(self.n @ x + self.delta)
        a, dcl = self._alpha(d)
        L = self.C @ a
        if 0.0 < d < self.dpos:
            dLd = (self.C[:, 2] - self.C[:, 1]) / (self.dpos + 1e-12)
        elif self.dneg < d < 0.0:
            dLd = (self.C[:, 1] - self.C[:, 0]) / (-self.dneg + 1e-12)
        else:
            dLd = np.zeros_like(L)
        return LocalResult(value=L, distance=d, weights=a, clipped_distance=dcl, distance_derivative=dLd)

def predict_and_update(x, y, tensors, K=4, eta=0.05, eta_g=0.005, train=True):
    # distances and local outputs
    loc = []
    for i, T in enumerate(tensors):
        res = T.local(x)
        loc.append((i, abs(res.distance), res))
    # select top-K by |d|
    idx = np.argpartition([t[1] for t in loc], K)[:K]
    active = [loc[i] for i in idx]
    # weights (softmax over -|d|/tau_i)
    ws = np.array([np.exp(-abs(res.distance)/tensors[i].tau) for i,_,res in active], dtype=np.float32)
    ws /= ws.sum() + 1e-12
    # blend
    yhat = sum(w * res.value for w, (_,_,res) in zip(ws, active))
    if not train:
        return yhat
    # gradient at output (MSE)
    g = (yhat - y)  # dL/dyhat
    # updates
    for w, (i, _, res) in zip(ws, active):
        Ti = tensors[i]
        Ti.C -= eta * w * np.outer(g, res.weights)
        coeff = float(g @ res.distance_derivative)
        Ti.n -= eta_g * w * coeff * x
        Ti.n /= (np.linalg.norm(Ti.n) + 1e-8)
        Ti.delta -= eta_g * w * coeff
    return yhat
```

Single block; 4-space indentation; vectorizable; easy to port to JAX/PyTorch (treat \(\alpha\) as differentiable piecewise ops).

---

## 9. Deployment Patterns

- **LLM token regression:** Feed per-token, per-head RoPE features; optionally sketch to \(d\); train HTFR to predict logits. Use shared decoder if \(M\) is large.
- **Streaming:** Top-\(K\) active updates per sample; periodic reseeding of low-usage HyperTensors.
- **Indexing:** For large \(N\), maintain an ANN index over signed distances (or normals) to shortlist candidates before exact top-\(K\).

---

## 10. Experimental Protocol

- **Datasets:** (i) synthetic smooth fields; (ii) tabular benchmarks; (iii) token-wise LLM logits (distillation).
- **Baselines:** ridge, random-features, RBFN, shallow MLP, linear MoE.
- **Metrics:** RMSE / CE, latency, memory, active-\(K\) fraction, ablations on \(K\), \(\tau\), band widths.
- **Ablations:** KNN vs softmax locality, linear vs Hermite interpolation, geometry updates on/off, sketch vs none.

---

## 11. Limitations & Mitigations

- Non-C¹ kinks at band edges → use Hermite interpolation if needed.
- Geometry drift instability if \(\eta_g\) too large → use orthonormality and gradient clipping.
- Coverage gaps if \(N\) too small → reseed low-usage HyperTensors and widen bands.
- Very large \(N\) cost → top-\(K\) with ANN prefilter + batching.

---

## 12. Summary (one-line)

HTFR replaces heavy deep stacks with a field of small, interpretable HyperTensors that interpolate along learned directions and are trained end-to-end with backprop-style updates, delivering strong accuracy at low latency on ultra-high-dimensional inputs.

---

## Python Implementation Overview

This repository provides a NumPy implementation of HTFR that closely follows the white paper.

- [`htfr.tensor.HyperTensor`](htfr/tensor.py) implements the geometric primitive with piecewise-linear interpolation.
- [`htfr.model.HTFRModel`](htfr/model.py) wraps multiple HyperTensors, performs top-\(K\) locality selection, computes softmax or inverse-distance weights, and applies online updates using MSE or softmax cross-entropy gradients.
- [`htfr.initialization`](htfr/initialization.py) contains lightweight k-means clustering and principal-direction routines to seed HyperTensors from data.

### Quickstart

```python
import numpy as np
from htfr import HyperTensor, HTFRModel

# build a toy model with two tensors
output_dim = 1
ht1 = HyperTensor(n=np.array([1.0, 0.0]), delta=0.0, dneg=-1.0, dpos=1.0,
                  C=np.array([[0.0, 0.5, 1.0]]))
ht2 = HyperTensor(n=np.array([-1.0, 0.5]), delta=0.2, dneg=-0.5, dpos=0.5,
                  C=np.array([[1.0, 0.3, -0.2]]), tau=0.5)
model = HTFRModel.from_tensors([ht1, ht2], top_k=2)

# online update
x = np.array([0.2, -0.1])
y = np.array([0.75])
yhat = model.predict_and_update(x, y)
print("prediction", float(yhat.item()))
```

### Dependencies

- Python 3.10+
- [NumPy](https://numpy.org/) (`pip install numpy`)

### License

MIT

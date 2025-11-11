# Interpolation Locality Update Analysis (Revised)

## 1. Impact of Locality Controls
- **Reference scaling.** Setting `interpolation_reference = 5 × max_knn_radius` reliably confines gradient updates to tensors within the existing KNN horizon. The reference radius now propagates through initialization, checkpoint load/save, and runtime creation so locality remains stable after serialization.
- **Update sparsity.** Empirically, the band clipping in each interpolator combined with the tighter radius reduces the number of tensors whose controls are updated per sample. This cuts cross-talk between unrelated regions and lowers gradient noise, especially on heterogeneous training buffers.

## 2. Module Quality Review
| Module | Strengths | Weaknesses | Keep? |
| --- | --- | --- | --- |
| **LERP** | Minimal compute, numerically robust baseline. | Lowest expressiveness. | ✅ Core fallback for diagnostics and ablation.
| **Hermite** | C¹ continuity when derivatives are available; smooth transitions. | Requires derivative estimates we currently fabricate from three samples, causing overshoot on noisy batches. | ⚠️ Keep for now but gate behind non-default weight until derivative bootstrapping is implemented.
| **Catmull–Rom** | Handles smooth data without derivatives, stable gradients. | Boundary extrapolation still linear; needs per-band tension control. | ✅ Good default for smooth regimes.
| **Local polynomial patches** | Captures gentle curvature, cheap to evaluate. | Susceptible to oscillations if band spans abrupt discontinuities. | ✅ Useful for mid-frequency trends.
| **Compact-support wavelets** | Detects sharp changes, cheap derivatives. | Hard clipping makes gradients sparse; interacts poorly with softmax weighting. | ❌ Recommend removing unless paired with multi-resolution controls; its binary weights destabilize ensemble updates.

**Removal recommendation.** Drop the compact-support wavelet module from default randomization. It contributes little expressiveness beyond local polynomials yet complicates training because its derivatives vanish outside a narrow core. Moving it to an optional experimental registry would simplify maintenance and avoid silent degradations on smooth data.

## 3. Suggested Enhancements
1. **Derivative bootstrapping for Hermite.** Fit finite-difference slopes from a wider neighborhood (e.g., 5-point stencil) during initialization so Hermite segments receive consistent tangents.
2. **Tension-aware Catmull–Rom.** Introduce a per-tensor tension parameter (Kochanek–Bartels style) to modulate curvature when extrapolation artifacts appear.
3. **Adaptive module weighting.** Track per-module loss deltas and anneal `interpolation_weights` so the randomizer favors modules with lower recent error.
4. **Local radius annealing.** Start with `locality_radius = 5 × max_knn_radius` but shrink it for tensors whose residual variance drops below a threshold; this tightens specialization over time.
5. **Diagnostics tooling.** Surface module usage histograms and average clipped distances per tensor in the training logs to highlight saturation or underutilization.

## 4. Simplifications for Maintainability
- Consolidate spline-based modules under a single implementation that toggles Hermite vs. Catmull–Rom basis functions via flags; this reduces duplicated algebra.
- Provide a small registry helper to register/unregister experimental modules so downstream researchers can safely trial removals (like the wavelet) without editing the core package.
- Document the recommended weight presets (e.g., bias towards Catmull–Rom and local polynomials, de-emphasize Hermite) directly in the README to guide practitioners.

Overall, the locality-aware modular interpolators behave as intended, but pruning the wavelet module from the default ensemble and refining derivative estimation will materially improve stability.

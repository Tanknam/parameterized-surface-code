# Technical Documentation: Per-Qubit Noise rMWPM Benchmarking

## 1. Repository Overview

```
/requirements.txt          Python dependencies (Stim, sinter, pymatching, numpy, matplotlib, etc.)
/scripts/
  circuit_pauli_override.py  Per-qubit noise injection and scaling helpers
  rmwpm_decoder.py           Reweighted MWPM decoder (edge reweighting: -log p)
per_qubit_noise_quickstart.ipynb  Notebook: synthetic data generation, benchmarking & plots
*.checkpoint                Sinter resume checkpoints for long Monte Carlo runs
idea.txt                    Project narrative / motivation (non-technical)
TECHNICAL_DOCUMENTATION.md  (this file)
```

## 2. Core Concepts

| Concept | Description |
|---------|-------------|
| i.i.d. noise | Homogeneous physical error probability across all qubits. |
| i.n.i.d. noise | Spatially heterogeneous noise derived from per-qubit T1/T2 values mapped to Pauli channels. |
| rMWPM | Reweighted Minimum-Weight Perfect Matching: assign each matching edge weight `w = -log(p)` of its originating error mechanism to bias decoding toward paths consistent with heterogeneous error likelihoods. |
| DEM | Stim Detector Error Model enumerating discrete error mechanisms, their probabilities, and the detectors/logical observables they affect. |
| Graphlike term | DEM error mechanism that flips at most two detectors (after Stim decomposition). Maps cleanly to a standard MWPM edge. |

## 3. Data Flow

1. Synthetic T1/T2 tables defined per target distance (d=3,5,7).
2. For each distance: obtain actual qubit indices from a Stim-generated unrotated surface code circuit (with noise placeholders) and randomly assign (T1,T2) pairs (with repetition if needed) to those qubits.
3. Convert (T1,T2) to Pauli error rates `(px,py,pz)` using `pauli_from_t1_t2` (Pauli-twirled amplitude/phase damping approximation). Enforce physical constraint `T2 ≤ 2*T1`.
4. Override noise: starting from a Stim generator circuit, replace uniform single-qubit noise instructions with explicit per-qubit `PAULI_CHANNEL_1` instructions (drop bulk depolarizing noise). Two-qubit noise currently not reinserted (placeholder area for future work).
5. Scale probabilities to a target mean error rate (preserve relative heterogeneity) via `CircuitWithQubitNoise.scale_to_target_error_rate`.
6. Extract a Detector Error Model (DEM) from the scaled circuit (Stim's `circuit.detector_error_model(...)`).
7. Build rMWPM matching graph: iterate over flattened DEM error instructions; for each graphlike group produce a PyMatching boundary or interior edge with weight `-log(max(p, min_probability))`.
8. Supply custom decoder to Sinter for Monte Carlo sampling tasks; Sinter packs the detection events, which are then decoded by the compiled rMWPM instance.
9. Aggregate logical error statistics; interpolate pseudo-thresholds (solve `P_L(p) = p`).

## 4. Important Modules

### 4.1 `circuit_pauli_override.py`
Key objects/functions:
- `T1T2Info`: simple container (currently not exploited fully due to omitted code fragments).
- `pauli_from_t1_t2(duration, T1, T2)`: returns `(px,py,pz)` using exponential decay formulas:
  - `px = py = (1 - e^{-duration/T1}) / 4`
  - `pz = (1 + e^{-duration/T1} - 2 e^{-duration/T2}) / 4`
  Clamped into `[0,1]`.
- `override_noise_with_pauli_channels(circuit, mapping, idle_duration_us)`: constructs a new circuit where the standard uniform noise instructions are replaced per qubit.
- `CircuitWithQubitNoise`: dataclass storing the transformed circuit and `qubit_noise_map` (qubit -> `(px,py,pz)`), plus utilities:
  - `get_mean_error_rate()`: average over `(px+py+pz)` per qubit.
  - `scale_to_target_error_rate(target)`: multiply all noise probabilities by a factor so that the mean matches `target`, preserving ratios; ensures physical constraints.
- `_transform_block` and `_scale_noise_block`: recursion over Stim circuit instructions and `REPEAT` blocks to either substitute or scale noise operations.
- Public creation helpers:
  - `create_circuit_from_t1_t2(rounds, distance, t1t2_mapping, idle_duration_us)`
  - `create_scaled_circuit_from_t1_t2(..., target_error_rate, ...)`
- Decoder convenience wrappers:
  - `build_rmwpm_decoder(circuit_with_noise, ...)`
  - `create_sinter_rmwpm_decoder(min_probability=1e-15)`

### 4.2 `rmwpm_decoder.py`
Main responsibilities:
- `RMWPMDecoder.from_circuit` / `.from_detector_error_model`: build a PyMatching matcher respecting rMWPM weighting.
- Edge processing loop:
  1. Iterate flattened DEM instructions; skip non-error types.
  2. For each error: get probability `p`, compute weight `w = -log(max(p, min_probability))`.
  3. Partition targets into `(detectors, logicals)` groups via `_extract_groups` (handles separators).
  4. Add boundary edge if one detector; interior edge if two; skip otherwise (non-graphlike).
  5. Track `detectors_used` and max logical/fault indices.
- Post-pass: ensure number of fault ids ≥ observables; pad unused detector indices (0 probability placeholders) so PyMatching dimension matches `dem.num_detectors` (prevents packed shape mismatch with Sinter).
- `SinterRMWPMDecoder`: implements Sinter's `Decoder` interface (`compile_decoder_for_dem`) returning a `_CompiledRMWPMDecoder`.
- Batch decoding uses `bit_packed_shots=True` for efficiency.

## 5. Algorithms & Rationale

### 5.1 Pauli Error Approximation
Given idle duration Δt and T1,T2:
\[
px = py = \frac{1 - e^{-\Delta t / T_1}}{4}, \quad pz = \frac{1 + e^{-\Delta t / T_1} - 2 e^{-\Delta t / T_2}}{4}
\]
This corresponds to Pauli-twirled amplitude + phase damping. Constraint `T2 ≤ 2 T1` ensures derived probabilities are non-negative.

### 5.2 Scaling Noise
Let `mean_raw = (1/N) Σ_i (px_i+py_i+pz_i)`. Target mean = `P_target`. Scale factor `α = P_target / mean_raw`. If any `(px_i+py_i+pz_i)*α > 1`, clamp by computing the maximum safe factor `α_safe = min_i (1 / (px_i+py_i+pz_i))` and using `min(α, α_safe)`.

### 5.3 rMWPM Edge Weights
Standard MWPM often uses uniform weights (or log-likelihood ratios when X/Z separated). Here, each error term’s probability directly influences matching cost: higher `p` ⇒ lower weight (since `-log p` decreases), encouraging corrections through more-likely fault locations. This is equivalent to maximizing product of edge probabilities (or minimizing sum of negative logs) consistent with a given boundary syndrome.

### 5.4 Detector Padding
If Stim’s DEM includes detectors that never appear in any processed edge (e.g. because their only associated errors were decomposed away or appear inside ignored constructs), PyMatching would otherwise treat the graph as having fewer detectors than Sinter’s packed data columns. Adding dummy boundary edges with probability `min_probability` makes these detectors explicit, preserving shape alignment.

## 6. Usage (Programmatic)

Example pattern (simplified):
```python
from scripts.circuit_pauli_override import create_scaled_circuit_from_t1_t2, create_sinter_rmwpm_decoder
import sinter

circuit = create_scaled_circuit_from_t1_t2(rounds=9, distance=3, t1t2_mapping=my_t1t2_map, target_error_rate=0.01, idle_duration_us=0.35)

stats = sinter.collect(
    tasks=[sinter.Task(circuit=circuit, json_metadata={'d':3,'r':9,'p':0.01})],
    decoders=['rmwpm'],
    custom_decoders={'rmwpm': create_sinter_rmwpm_decoder()},
    max_shots=1_000_000,
    max_errors=50_000,
)
```

## 7. Notebook Outputs
- Logical error rate vs physical error rate plots (log-log) for i.i.d. and i.n.i.d.
- Pseudo-threshold interpolation producing distance vs pseudo-threshold plot (log scale y-axis).

## 8. Configuration Parameters
| Parameter | Location | Effect |
|-----------|----------|--------|
| `idle_duration_us` | Circuit creation functions | Adjusts base (px,py,pz) via T1/T2 exponentials. |
| `target_error_rate` | `create_scaled_circuit_from_t1_t2` | Mean total single-qubit error probability after scaling. |
| `min_probability` | rMWPM decoder factory | Lower bound for `p` in `-log(p)`; also placeholder for padding edges. |
| `approximate_disjoint_errors` | `RMWPMDecoder.from_circuit` | Whether composite errors (`PAULI_CHANNEL`) are split into independent events by Stim. |
| `decompose_errors` | same | Enable Stim’s graphlike decomposition of hyperedges. |

## 9. Limitations / Known Gaps
- Two-qubit gate error modeling currently stripped; no custom per-pair noise injection implemented.
- Measurement and reset error asymmetry not individualized per qubit.
- rMWPM currently ignores hyperedges > 2 detectors (skipped); potential information loss for correlated events.
- Missing full unit tests; only informal notebook-based validation.
- Scaling logic fragments (some lines omitted in repository snippet) should be revisited to ensure physical constraint handling is complete in final version.

## 10. Potential Enhancements
1. Implement structured two-qubit error modeling with `PAULI_CHANNEL_2` probabilities derived from pairwise rates.
2. Add support for hypergraph matching (or transform higher-order terms with ancilla nodes) rather than skipping non-graphlike terms.
3. Integrate statistical error bars and automated regression plots.
4. Provide a CLI (`python -m scripts.run_benchmark --distance 5 --decoder rmwpm ...`).
5. Add caching of per-qubit probability tables and device calibration import (JSON/CSV loader).
6. Unit tests for: probability scaling, DEM to edge translation, detector padding, pseudo-threshold interpolation.
7. Profiling: measure runtime & memory deltas for rMWPM vs baseline PyMatching across distances.

## 11. Validation Steps Performed
- Manual sampling of small circuits; verified syndrome packing size matches `dem.num_detectors` (no shape mismatch after padding logic added).
- Decoding of a batch of packed syndromes returns predictions array with expected shape and plausible mismatch count.
- Visual pseudo-threshold comparison indicates modest improvement for rMWPM under synthetic heterogeneity (consistent with limited variance).

## 12. Quick Troubleshooting
| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| "column mismatch" between decoder and packed data | Unused detectors not represented in graph | Ensure padding logic executed (`rmwpm_decoder.py` latest). |
| All rMWPM results identical to baseline | Heterogeneity too small / scaling washed out differences | Increase variance of T1/T2 or disable scaling for an experiment. |
| NaN weights | Zero probabilities before `-log` | Confirm `min_probability` > 0. |

## 13. Glossary
- **DEM**: Detector Error Model
- **Graphlike**: Error involving ≤2 detectors
- **MWPM**: Minimum-Weight Perfect Matching
- **rMWPM**: Reweighted MWPM using per-edge `-log(p)` weights
- **Pseudo-threshold**: Physical error rate where logical-per-round equals physical rate.

## 14. Citation Notes
The rMWPM approach follows the general principle of using negative log probabilities as additive edge weights for MWPM; exact equation alignment should be cross-checked with the paper’s definitions when writing any publication or thesis chapter.

---
End of technical documentation.

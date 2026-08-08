# Self-Diagnosing Parameter Prediction

## Confidence-Gated Models for Sparse Tuning Data

LIBXS Predict

Note: Open with the deployment problem: a predictor is useful only if it
knows when a safe rule should stay in charge.

---

## The Problem

CP2K and DBCSR use tuned GPU kernels for known matrix shapes.  
However, deployment sees new shapes between tuned points.

| Choice | Risk |
| --- | --- |
| Fixed rules only | Miss local tuning opportunities |
| Predict everything | Silent slowdowns |
| Confidence-gated | Override only with evidence |

<span style="opacity: 0.4; font-size: 50%;">Prior work predicted offline
based on hardware-occupancy features using XGBoost [Jakobovits 2019].</span>

---

## Method in One Slide

Distance-weighted *k*NN voting plus polynomial fingerprint diagnostics.

The model returns:

- Predicted value.
- Per-output confidence.
- Override/defer signal.

Note: The main phrase is not just prediction, but deployment decision
support.

---

## GPU Kernel Dispatch

Small-matrix GPU kernel dispatch from inputs `M`, `N`, and `K`.

**Output**: batch size, block sizes, workgroup shape,  
loop unroll, layout, and access selectors.

**Training data**: tuned kernels parameters  
(device-agnostic, Intel PVC shown here).

---

## Why Ordinary Accuracy Is Not Enough

Some parameters encode hidden hardware constraints.

Nearby shapes can agree on a value that is wrong for the query.

| Shape | Predicted BK | Rule BK | Result |
| --- | ---: | ---: | ---: |
| 21 × 22 × 23 | 4 | 21 | 487 vs. 991 GF/s |

Average error is not the operational risk, e.g., Mean Absolute Error.

Note: This example motivates policy separation. The current full-rerun
evidence is summarized later.

---

## Deployment Policy

Separate ownership from prediction.

| Rule controlled | Confidence gated |
| --- | --- |
| `BS`, `BM`, `BN`, `BK`, `WS` | `WG`, `LU`, `AL`, `AA`, `AB` |
| structural safety | preference/access choices |
| source rules stay authoritative | override near-unanimously |

SMM kernel parameters: BS batch-size, BM/BN/BK block extents,  
WS work-sharing, WG workgroup shape, LU unroll,  
AL/AA/AB access modes.

---

## Confidence Signals

| Signal | Time | Used for |
| --- | --- | --- |
| Fingerprint decay | Build | constant, smooth, categorical, erratic |
| *k*NN vote fraction | Query | per-output deployment confidence |

Fingerprint behavior chooses the output mode.

Neighbor agreement decides whether a prediction may act.

---

## Override Rule

```text
if output is rule-owned:
    use safe rule
else if confidence ≥ threshold:
    use prediction
else:
    use safe rule
```

Abstention is part of LIBXS behavior. Learned tuning  
becomes compatible with hard-won domain rules.

---

## Tuned GPU Parameters

![PVC tuning impact by arithmetic-intensity bin](assets/pvc_ai_performance_slide.png)

1339 PVC kernels, three reruns per mode.  Tuning gives +1.3% over
handwritten rules; LOO prediction reaches +1.1%.  The gain
concentrates in compute-heavy shapes (AI 2–4: +6.8%, 41 distinct BK
values).  Other bins are near neutral — the rules are already strong.

---

## Confidence Projection

![Saved PVC predictor confidence over the M×N×K cube](assets/pvc_confidence_projection.png)

Over the M × N × K cube (739k queries), 42% fall below the 0.9
threshold (defer to rules).  58% sit at or above it, but the rest is
graded rather than split: 22% lands in [0.8, 0.9) — just short of the
gate.  The threshold is a policy choice on a continuum.

---

## What Confidence Gating Buys

It changes the failure mode.

| Without gating | With gating |
| --- | --- |
| Wrong values silently deploy | Low evidence defers |
| Average error hides risk | Per-output confidence is visible |
| Outliers look like bugs | Outliers identify missing data |

How to know if a parameter is confidently predicted?  
Well, if you know how to predict...

Note: confidence = (sum of weights voting for winner) / (sum of all weights).
A continuous output has no winner to count, so it reports 1.0 and callers read
info->variance instead; libxs_predict_set_dispersion optionally turns that
spread into a confidence (used by the earthquake sample).

---

## Beyond Kernel Dispatch

The same LIBXS machinery handles:

- Timeseries forecasting.
- Spatial prediction.
- Cross-series decomposition.
- Non-stationary series with auto-differencing.
- Materials classification.

The interface is still prediction plus confidence.

---

## Crystal System Prediction

<!-- .slide: data-background-image="assets/crystal_system_wheel_slide.png" data-background-size="contain" data-background-position="right center" style="text-align: left" -->

- 60 386 compositions
- 37 features
- 7 crystal systems

The sample is a mixed classification problem  
where confidence decides whether to act.

<span style="opacity: 0.4; font-size: 50%;">AFLOW: An Automatic Framework
for High-Throughput Materials Discovery [Curtarolo 2012].</span>

Note: This is the key slide for computational chemistry audience.
Structure initialization in CP2K/FHI-aims requires symmetry information;
a confidence-gated predictor can provide it or abstain.

---

## Secondary Evidence

| Domain | Ours | Literature | Confidence |
| --- | ---: | ---: | --- |
| ETT** (H=96) | MSE 0.244 | 0.370–0.449 | 0 parameters, 1 CPU core |
| Sunspots | MAE 17.4 | MAE 19.8–45.5 | 1.0 (dense cycles) |
| Discharge | 0.22 err/σ | 0.10–0.47 err/σ | 1.0 (seasonal) |
| SOI* | nRMSE 0.11 (0.07 hKNN) | 0.23–0.55 | 1.0 (spread modes) |
| Earthquakes | MAE 0.259 (0.254 hKNN) | 0.184–0.283 | 0.462 (ambiguous) |
| Crystals | 79.6% → 95.0% (conf ≥ 0.9) | ≈75–80% | 54% gated coverage |

Confidence separates dense-coverage domains from genuinely ambiguous
ones.  Literature comparisons are orienting — different features, splits,
metrics.  ETT is the exception: same dataset, same split, same horizon.

<span style="opacity: 0.4; font-size: 50%;">Results for comparison from
[Dang2022], [Akkala2025], [Kratzert2018], [Kratzert2019], [Simatupang2025],
[Ahmed2024], [Kaftan2025], [Nie2023], [Zeng2023], [Zhou2022], [Wu2021]</span>

<span style="opacity: 0.4; font-size: 50%;">* SOI: Southern Oscillation Index —
** ETT: Electricity Transformer Temperature (ETTh1), the standard timeseries
benchmark; baselines are PatchTST, DLinear, FEDformer, Autoformer</span>

Note: The Southern Oscillation Index (SOI) measures the difference in air pressure between
Tahiti and Darwin, Australia, serving as a key indicator of El Niño and La Niña events.

---

## Why This Matters for Atomistic Codes

Simulation setup often needs plausible structure or  
kernel choices before expensive computation begins.

A confidence-gated predictor can say:

- This guess is supported enough to use.
- This case is ambiguous; keep the conservative path.
- This regime deserves new measurements or another feature.

---

## Fortran-First Feedback Loop

No Python, no framework dependency — links into your Fortran binary.

| Running application moment | LIBXS call | Effect |
| --- | --- | --- |
| Load existing knowledge | `libxs_predict_load_csv` | seed model from file |
| New measured case | `libxs_predict_push` | append evidence 𝒪(1) |
| Checkpoint or idle point | `libxs_predict_build` | rebuild model cheaply |
| Next query | `libxs_predict_eval` | value + confidence |

Start from a CSV of prior runs or start empty — learn from completed  
work, and let later decisions use the stronger local evidence.

---

## Takeaways

- Sparse tuning spaces reward abstention.
- Confidence must be per output.
- Running jobs can add evidence  
  and rebuild at checkpoints.
- Fingerprints diagnose mode choice.
- *k*NN votes expose local evidence.
- Rule deferral turns uncertainty  
  into safe behavior.

---

## Closing Thought

The useful model is not the one that *always* has an answer.

It is the one that knows when its answer should not be in charge.

<span style="opacity: 0.3;">This slide set: https://libxs.readthedocs.io/predict/  
LIBXS: https://libxs.readthedocs.io/
</span>

---

## What Else is in LIBXS?

| Domain          | Summary                                                        |
|-----------------|----------------------------------------------------------------|
| Permutation     | Co-prime shuffling, smooth row permutations, stratification    |
| Histogram       | Thread-safe histogram with running statistics                  |
| Registry        | Thread-safe key-value store with per-thread caching            |
| Hashing         | CRC32-based hashing, Adler-32, string hashing                  |
| **Predict**     | Fingerprint-guided parameter prediction with model persistence |
| Malloc          | Pool-based allocator (steady-state, no system calls)           |
| Memory          | Byte comparison, matrix copy/transpose, alignment queries      |
| String          | Edit distance, substring search, word similarity, formatting   |
| Timer           | High-resolution timing via calibrated TSC                      |
| CPUID           | CPU feature detection (SSE to AVX-512, AArch64, RISC-V)        |
| GEMM            | Batched dense GEMM (strided, pointer-array, grouped)           |
| Math            | Matrix comparison, GCD/LCM, coprime, BF16 conversion           |
| MHD             | Read/write MetaImage (MHD/MHA) files                           |

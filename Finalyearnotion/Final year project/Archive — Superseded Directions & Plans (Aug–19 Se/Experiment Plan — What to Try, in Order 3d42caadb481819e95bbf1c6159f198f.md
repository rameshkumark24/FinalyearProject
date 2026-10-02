# Experiment Plan — What to Try, in Order

<aside>
🎯

Purpose: a ranked list of things to try, cheapest and most likely first, with a stopping rule for each. This is the trial-and-error plan — written down **before** the experiments run, so that negative results stay reportable instead of looking like failure.

</aside>

---

# 0. The strategy — read this before choosing anything

<aside>
⚠️

**Do not try to beat 19.65 dB on scans.** That number comes from a funded lab with real clinical data and an actively maintained codebase. A three-person student team on free Colab will lose that race, and losing it consumes the whole timeline.

</aside>

<aside>
✅

**Compete where no numbers exist.** Nobody has published:

- per-image-type results for photographs on a common benchmark
- digitization error in **clinical units** (ms, BPM, mV)
- a **per-factor** robustness study isolating what actually breaks the pipeline

**You cannot lose a race that nobody has run.** Every item below is chosen on that principle.

</aside>

---

# 1. Where the gap physically is

The five-stage pipeline is on the Existing Methodology page. The failure is concentrated in **Stage 4, grid size extraction**, for two separate reasons:

| Mechanism | Why it breaks on photographs | What it suggests trying |
| --- | --- | --- |
| **M1 — Global scale, local distortion** | Grid spacing is recovered by autocorrelation over the page, producing **one** scale factor. Under perspective, true grid spacing **varies across the page**. A single global number is wrong by construction — near-correct in the middle, increasingly wrong toward the edges | Estimate scale **locally**, per lead region, instead of globally |
| **M2 — Total dependence on the grid** | If gridlines are faint, low-resolution, colour-shifted or absent, the only carrier of physical units disappears and the pipeline has no fallback | Use the **calibration pulse** as an independent scale reference |

## The calibration pulse

Standard 12-lead printouts carry a **1 mV calibration pulse: 10 mm tall, 0.2 s wide** — a physical ruler printed on the page itself, independent of the grid, and present in every lead row.

<aside>
🔑

Leading methods scale from **gridline autocorrelation**. The calibration pulse is a second, independent measurement of the same quantity — and being a large, high-contrast, local feature, it should survive perspective distortion and low resolution far better than faint 1 mm gridlines.

**Honesty check:** using reference pulses for scaling is a *known* idea and appears in the literature. What appears **not** to have been done is a controlled comparison of pulse-based versus grid-based scaling **as a function of image degradation**. That comparison is the contribution — not the idea. Confirm this positioning against the literature before submission.

</aside>

---

# 2. The experimental lever — ECG-Image-Kit is a factor generator

The generator exposes parameters that let **one degradation factor be varied at a time** while everything else is held fixed. This is what makes a controlled robustness study possible at all.

| Flag | What it controls | Why it matters here |
| --- | --- | --- |
| `--calibration_pulse` | Probability of adding the 1 mV pulse | **Enables the pulse-versus-grid experiment directly** |
| `--random_grid_present` | Probability of a grid at all; 0 = no grid | Tests whether anything survives without gridlines |
| `--standard_grid_color` (1–5) | brown, pink, blue, green, red | **Proxy for machine and manufacturer variation — supports the geography claim with no physical printing** |
| `-r`, `--random_resolution` | Resolution, default 200; random draws from [50, r] | Low resolution is a documented failure mode of existing tools |
| `--wrinkles`, `-ca` | Creases and crease angle | Physical paper damage |
| `--random_bw` | Black-and-white conversion | Photocopy and fax-like degradation |
| `--hw_text`, `--print_header` | Handwriting and printed header text | Occlusion of the trace |
| `--num_columns` | Layout (3x4, 6x2, 12x1) | Layout generalisation |
| `--store_config --lead_bbox --lead_name_bbox` | Writes ground-truth bounding boxes to JSON | **Free ground truth for the error decomposition in A1** |

---

# 3. Tier A — try these first (CPU only, days not weeks)

## A1. Error decomposition — segmentation error vs calibration error

**Do this before anything else.** It tests whether the project's core premise is true.

**Procedure:** run a baseline on matched scan and photo versions of the same records. For each output compute (a) trace overlap against the ground-truth mask, and (b) estimated grid scale versus true scale. Regress final SNR on each.

**Success:** calibration error explains substantially more SNR variance than segmentation error on photos.

**If it fails:** the premise is wrong — the problem is segmentation, not scaling. **That is still a publishable finding**, and it redirects everything below toward Tier C. Do not push on regardless.

**Stop after:** 3 days.

## A2. Calibration pulse vs grid scaling

**Procedure:** generate four matched sets — grid on/off crossed with pulse on/off — at several distortion levels. Implement a pulse detector (the pulse is a large rectangular step at a known position). Compare scale-factor error from the pulse against from the grid, as degradation increases.

**Success:** pulse-derived scale error grows more slowly than grid-derived error as distortion rises.

**Why it is worth trying:** the pulse is 10 mm tall and high-contrast; gridlines are 1 mm and faint. Under blur and downsampling, the small periodic structure disappears long before the large step does.

**Stop after:** 5 days without a measurable crossover.

## A3. Local vs global grid estimation

**Procedure:** replace the single page-level scale with a per-lead-region estimate; optionally fit a smooth scale field across the page. Compare on photos.

**Success:** measurable SNR gain on photo categories, with no loss on scans.

**Cheap diagnostic first:** measure how much true grid spacing actually varies across a photographed page. If variation is negligible, skip A3 entirely — half a day saves a week.

## A4. Physiology-constrained scale selection

**Procedure:** rather than trusting one scale estimate, generate several candidates, reconstruct under each, and select the one whose output is most physiologically plausible — heart rate in a sane range, QRS duration in a sane range, amplitudes not absurd.

**Why it is attractive:** it exploits domain knowledge that no current method uses, and it needs no training. A reconstruction implying a heart rate of 400 BPM is certainly mis-scaled and can be rejected without ground truth.

**Risk:** plausibility bands must come from a cited clinical source, not invented. Blocked on obtaining the AHA/ACCF/HRS reference ranges.

## A5. Clinical-parameter error metric

The headline contribution, already decided. NeuroKit2 delineation on ground truth and digitized output; report error in BPM, ms and mV; correlate against SNR.

**Report the delineation failure rate as a first-class result** — "digitization degraded X% of signals past the point of measurability" is itself a finding.

---

# 4. Tier B — moderate cost

| Experiment | Procedure | Payoff / risk |
| --- | --- | --- |
| **B1. Factorial robustness study** | Vary one generator factor at a time; measure SNR and clinical error per factor | **High payoff, low risk.** Produces a per-factor degradation table nobody has published. Works even if every other experiment fails |
| **B2. Homography from grid intersections** | Detect grid intersections, fit a homography, dewarp before digitizing | Principled; but the strongest baseline already does something similar — confirm it is not duplication |
| **B3. Ensemble of open digitizers** | Run 2–3 tools, combine per lead by agreement or median | Often works; **low novelty**. Good safety net, weak as a headline |
| **B4. Test-time augmentation** | Digitize several augmented copies, aggregate | Cheap, usually a small gain; not a contribution on its own |
| **B5. Grid-colour generalisation** | Evaluate across all 5 grid colours | **Directly supports the geography claim with zero printing and zero ethics exposure** |

---

# 5. Tier C — Phase II, needs GPU

| Experiment | Note |
| --- | --- |
| **C1. Fine-tune segmentation with photo-heavy augmentation** | Standard domain adaptation. Likely to work, but needs GPU and is the most contested ground |
| **C2. Three-arm classification comparison** | Answers Gap 2. Either outcome publishes |
| **C3. Learned scale regressor** | Predict scale directly from the image. Elegant; needs the most data and compute |

---

# 6. Procedure — how to run trial and error without wasting the term

<aside>
📏

**Rules. Agree these once and hold to them.**

1. **Freeze a dev set and a test set at the start.** All tuning happens on dev. The test set is touched **once**, at the end. Patient-independent — no record may appear in both
2. **One variable per run.** Two changes at once means neither is attributable
3. **Log every run in the Progress Log, including failures** — with the parameter changed and the number obtained
4. **Write the expected result before running.** If the outcome surprises you, that is the interesting part; if you did not write it down, you will not notice
5. **Obey the stopping rules.** An experiment that has not shown an effect in its budgeted days is finished. Record it and move on
6. **Always report per-image-type**, never a single average — the average is exactly what hides the photo failure
</aside>

## Order of execution

- [ ]  **Week 1** — baseline runs end to end on one image (Review II gate)
- [ ]  **Week 1** — A1 error decomposition → decides everything downstream
- [ ]  **Week 2** — A5 clinical metric implemented; B1 factorial study started
- [ ]  **Week 3** — A2 calibration pulse; A3 local grid (only if A1 supports them)
- [ ]  **Week 4** — B1 completed, B5 grid colour; results consolidated (Review III gate)
- [ ]  **Phase II** — Tier C

---

# 7. How to know there is a paper

<aside>
📄

A submittable paper exists as soon as **all** of these hold — note that none of them requires beating anyone:

- [ ]  Per-image-type results for at least two baselines on a public benchmark
- [ ]  Digitization error reported in clinical units, with its relationship to SNR
- [ ]  A per-factor degradation table from the factorial study
- [ ]  At least one mechanism explaining *why* the failure occurs, supported by the error decomposition
- [ ]  Code and evaluation scripts released
- [ ]  Limitations stated honestly, including everything synthetic

**A method that improves on the baseline is a bonus, not a requirement.** The measurement and the mechanism are the contribution.

</aside>
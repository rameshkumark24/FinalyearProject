# Existing Methodology — How Current Methods Work

<aside>
🎯

Purpose: understand **exactly how the existing systems work internally**, so the project can identify which stage breaks on photographs and improve that stage specifically — rather than replacing a whole pipeline it does not understand.

Every stage description below is taken from a published method's own description of itself.

</aside>

---

# 1. The canonical five-stage pipeline

The clearest published decomposition is from the npj Digital Medicine open-source algorithm (Open-ECG-Digitizer). Almost every other method is a variation on these five stages, sometimes merging or skipping some.

| Stage | What it does | Technique used | Output |
| --- | --- | --- | --- |
| **1. Semantic segmentation** | Classifies every pixel as ECG trace, gridline, text, or background | Residual U-Net, 4-class pixel classification | Class mask |
| **2. Perspective correction** | Removes camera distortion and rotation | Hough transform applied twice — to angle-radius space, then angle-angle space — to recover the camera perspective from parallel gridlines | Dewarped image |
| **3. Layout identification** | Finds which lead is where on the page | Lightweight U-Net segments lead labels (aVR, III...), positions estimated by weighted centroids, matched to standard 6x2 / 12x1 / 3x4 layouts | Lead map |
| **4. Grid size extraction** | Recovers the pixel-to-physical-unit conversion | **Autocorrelation on gridline spacing**, plus adaptive grid search against templates | **mm/s and mm/mV scale factors** |
| **5. Segmentation-to-trace** | Turns the 2-D mask into 1-D signals | Connected-component analysis; a "snipping" algorithm for overlapping signals; Jonker-Volgenant linear sum assignment to match segments to leads | 12-lead time-series |

<aside>
🔑

**Stage 4 is the project's target.** Stages 1, 3 and 5 decide *where the ink is*. Stage 4 alone decides *what the ink means in millivolts and milliseconds*. A pipeline can segment a trace perfectly and still emit a clinically wrong signal if Stage 4 mis-estimates the grid — and Stage 4 is the stage most damaged by perspective, because a photograph makes grid spacing vary across the page.

This is the mechanism behind hypothesis **H2** and the reason the project is tractable without a large GPU.

</aside>

---

# 2. How each published method implements the pipeline

| Method | Geometry / rotation | Segmentation | Trace extraction | Reported result |
| --- | --- | --- | --- | --- |
| **Krones et al.** (Challenge winner, BSD-2) | Hough transform — **rotation only** | U-Net | Mask vectorisation | CV SNR 17.02; hidden-test 12.15. **Negative on all phone-photo categories** |
| **Open-ECG-Digitizer** (npj Dig Med) | **Full perspective dewarping** via double Hough | Residual U-Net, 4-class | Connected components + Jonker-Volgenant assignment | **19.65 dB — stated on scanned papers** |
| **Yu et al.** (USST_Med, CinC 2024) | YOLOv8-Tiny estimates tilt angle from lead-name positions | ResUNet with CBAM attention | Column-by-column scan of binary mask | SNR 2.202 (5/16); macro F 0.393 |
| **Shang et al.** (mins-eth, ETH Zurich) | Handles rotation, cropping, creases, text | Faster R-CNN detection then U-Net segmentation | Per-region extraction | SNR 0.893 (6/16) |
| **Karbasi et al.** (2025) | Adaptive grid detection module | Two-stage U-Net, augmented for overlap | Mask to time-series | IoU 0.87; rho 0.964 |
| **ECGtizer** | Automated lead detection | Three pixel-based extraction algorithms | Deep-learning signal reconstruction module | Reports outperforming ECGminer and PaperECG |
| **PMcardio** (commercial, closed) | Grid detection and distortion correction | Deep-learning lead extraction | Not disclosed | PCC above 0.91, SNR above 12.5 dB, RMSE below 0.10 mV; under 7 s |

<aside>
⚖️

**Be precise about what is and is not solved.** Open-ECG-Digitizer explicitly implements full perspective dewarping for phone photos, and PMcardio explicitly targets smartphone images. The project must **not** claim that photographs are unhandled.

The accurate claim: **no published work reports per-image-type results for photographs on a common benchmark.** The winner's negative photo scores are the only public per-category evidence, and the strongest method reports its headline number on scans. That is a measurement gap, and measuring it is a legitimate contribution.

</aside>

---

# 3. Classification methodology

Two competing paradigms, which is what makes hypothesis **H4** a real question.

| Approach | How it works | Strength | Weakness |
| --- | --- | --- | --- |
| **Direct image classification** | CNN consumes the ECG image, outputs diagnosis | Immune to grid-calibration error; challenge best macro F 0.817 | Not interpretable in clinical units; cannot feed signal-based tools |
| **Digitize then classify** | Image to time-series, then a 1-D signal model | Produces a reusable signal; interoperable; archivable | Inherits every calibration error from Stage 4 |
| **Fusion** | Both streams combined | Potentially best of both | Untested in this setting — the project's third arm |

---

# 4. Software stack the field actually uses

| Tool | Purpose | Licence | Status |
| --- | --- | --- | --- |
| **ECG-Image-Kit** | Generate synthetic ECG images with distortions | BSD-3-Clause | Last commit Oct 2024 — expect dependency rot |
| **ECG-Digitiser** (Krones) | Challenge-winning baseline | BSD-2-Clause | Last push Jun 2025 |
| **Open-ECG-Digitizer** | Strongest open baseline; the 5-stage pipeline | **"Other" — read the LICENSE file before use** | Active, last push Jun 2026 |
| **NeuroKit2** | ECG delineation — P/QRS/T onsets and offsets, giving PR, QRS, QT | MIT | Very active, last push Sep 2026 |
| **wfdb-python** | Read PTB-XL and other PhysioNet WFDB records | MIT | Active |

<aside>
💡

**NeuroKit2 is the key enabler.** It delineates P onsets/offsets, QRS onsets/offsets and T onsets/offsets, which is precisely what is needed to compute PR, QRS and QT intervals from both the ground-truth signal and the digitized signal — and therefore to compute the **clinical parameter error** that Objective 2 proposes. MIT licence, actively maintained, runs on CPU.

</aside>
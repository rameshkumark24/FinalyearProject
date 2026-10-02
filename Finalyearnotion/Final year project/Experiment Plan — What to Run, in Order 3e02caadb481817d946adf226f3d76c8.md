# Experiment Plan — What to Run, in Order

<aside>
🧪

**Purpose.** The trial-and-error runs for the Structure–Function–Rhythm framework (see the Methodology Research page), in the order they run. Each run card has what to compare, how to measure it, the expected result written **before** running, and a keep or stop rule. Negative results stay reportable because the question is set in advance.

</aside>

---

# Rules

1. **Subject-level splits** — no fetus or mother appears in two splits. NIFEADB: leave one subject out. NInFEA: group the 60 entries by woman (39 women). Heartbeat: by patient.
2. **Freeze each test set** and touch it once.
3. **One change per run.**
4. **Write the expected result first.**
5. **Log every run in the Progress Log**, including failures.
6. **Report 95% bootstrap confidence intervals** — every dataset here is small.
7. **Time-box each run to 2 days.** If it overruns, record why and take the fallback.

---

# Stage 0 — Data (19–24 Sep, no forms needed)

- **D1.** Download NInFEA, NIFEADB and CinC 2013 set A from PhysioNet (Python `wfdb`). Write a one-page data card for each: channels, sampling rate, durations, labels, licence.
- **D2.** Write the split files (rule 1) and commit them before any model is trained.
- **D3.** Submit the Heartbeat and CARDIUM request forms; ask the authors about shared patients.
- **D4. Overlap check (on access).** Perceptual-hash every Heartbeat and CARDIUM image and count near-duplicates across the two datasets. Until this or the authors rule out overlap, never test one dataset with a model trained on the other.
- **D5.** Download CARDIUM's clinical JSON and trimester file now — they are public in its repository — and write the data card (1,103 patients; 74 CHD in the public files; 321 patients in both 2nd and 3rd trimester).

---

# Tier A — open data: start now

## R — Rhythm expert (fetal ECG)

| Run | Compare | Measure | Expected (write before running) | Keep / stop rule |
| --- | --- | --- | --- | --- |
| **R1** Maternal ECG removal | Template subtraction · FastICA · template subtraction + PCA | Fetal QRS F1 (±50 ms) against CinC 2013 set A references | Template subtraction ≥ ICA on 4-channel recordings | Keep the best; no new extraction network |
| **R2** Fetal QRS detection | Best single channel · multichannel voting | F1 as in R1 | Voting is more robust on noisy recordings | Keep the best |
| **R3** Rhythm baselines (NIFEADB, leave-one-subject-out) | RR / heart-rate-variability features + logistic regression · random forest · 1D-CNN from scratch | Subject-level AUROC, sensitivity, specificity with CIs | Simple features ≈ 1D-CNN — 26 subjects is too few for a network to shine | Record all three |
| **R4** Leakage check | R3's 1D-CNN with a random segment-level split vs the subject-level split | Accuracy gap | Segment-level accuracy is much higher — a warning about near-perfect figures reported on small fetal-ECG data | Report both; never use segment-level results as ours |
| **R5** Foundation-model probes | ECGFounder · ECG-FM · HuBERT-ECG — frozen embeddings + logistic regression; resample to each model's rate (check each model card) | As R3 | **H-R:** at least one foundation model matches or beats the best R3 baseline | A negative result is still reported — "adult ECG models do not transfer to fetal ECG" is a finding |
| **R6** Fine-tune | Best R5 model: last blocks or LoRA | As R3 | Small gain at most | Run only if R5 beats R3 |
| **R7** Calibration | None · temperature scaling | Expected calibration error, Brier score | Temperature scaling lowers the error | Convert the output into an opinion (belief, disbelief, uncertainty) for fusion |

## F — Function expert (Doppler ultrasound, NInFEA)

| Run | Compare | Measure | Expected | Keep / stop rule |
| --- | --- | --- | --- | --- |
| **F1** Envelope | NInFEA's MATLAB envelope tool · our Python port (Otsu threshold, area opening, edge detection) | Agreement between the two envelopes | Near-identical | Use the Python port if it agrees |
| **F2** Cycle detection | Peak-based · template matching · autocorrelation | Cycle F1 and cycle-length error (ms), with fetal-QRS times from R1–R2 as the reference | Template matching is best | F1 ≥ 0.9, otherwise keep only heart rate from Doppler |
| **F3** Timing measures | Cycle length · E/A ratio · mechanical AV interval · ejection time · QRS-to-ejection delay | Distributions by gestational week; sanity check against the published HR-IQS reference ranges (Heart Rhythm 2024) | Values within published ranges | Drop any measure that cannot be read in most cycles |
| **F4** Healthy range | Gestational-age regression on NInFEA (weeks 21–27) → z-score flag | Flag rate on held-out healthy women; response to synthetically altered envelopes (e.g. a lengthened AV interval) | Few flags on healthy; altered envelopes flagged | Label the altered-envelope test as synthetic everywhere |

## C — Cross-modal trust check (NInFEA)

| Run | Compare | Measure | Expected | Keep / stop rule |
| --- | --- | --- | --- | --- |
| **C1** Heart-rate agreement | Beat-level fetal heart rate from fetal ECG vs Doppler | Bland–Altman bias and limits of agreement | Close agreement on clean recordings | Baseline for C2 |
| **C2** Agreement score | Heart-rate difference · beat-train cross-correlation · small self-supervised model trained on matched vs time-shifted pairs | AUROC for spotting deliberately corrupted pairs: maternal rate substituted, time shifted, noise added | Cross-correlation ≈ the learned model; both beat plain heart-rate difference | Keep the simplest within the CI of the best |
| **C3** Trust gate | Rhythm and function experts with vs without the trust check | False-alarm rate on healthy NInFEA | **H-C:** fewer false alarms with the check | If no reduction, report it and keep the check only as a quality flag |

---

# Tier B — echo data: when access arrives

## S — Structure expert (Heartbeat; CARDIUM if granted)

| Run | Compare | Measure | Expected |
| --- | --- | --- | --- |
| **S1** Baselines | ImageNet ResNet-50 and ViT-B per view, mean over views | Patient-level sensitivity, specificity, AUROC, AUPRC with CIs | Below Heart-ViT |
| **S2** Reference | Heart-ViT **released weights** (both trimesters, 4 folds each) — no training needed | As S1, next to the published figures; pooled out-of-fold results as well as the 61-patient test (only 6 CHD) | Matches the published numbers |
| **S3** FetalCLIP probe | Frozen FetalCLIP + linear head per view | As S1 | Beats ImageNet baselines |
| **S4** FetalCLIP + LoRA | LoRA adapter | As S1 | Beats S3 |
| **S5** Control foundation models | DINOv2 · BiomedCLIP | As S1 | Below FetalCLIP — fetal-specific pretraining matters |
| **S6** View aggregation | Mean · max · attention; drop each view at inference | As S1; drop in sensitivity per missing view | Attention is most robust to a missing view |
| **S7** Metadata | Image-only vs + 4 metadata fields | As S1 | Small gain, as the Heartbeat authors found |
| **S8** Trimester shift | First, with no training: the released 2nd-trimester Heart-ViT weights scored on all 690 3rd-trimester patients, and the 3rd-trimester weights on the 2nd-trimester test. Then our models trained on one trimester and tested on the other, FetalCLIP vs ImageNet. On CARDIUM, leave the 321 patients seen in both trimesters out of every cross-trimester test | Drop from in-trimester results | **H-S:** FetalCLIP shrinks the drop |
| **S9** Calibration and uncertainty | Temperature scaling · evidential head · 3-model ensemble | ECE, Brier; uncertainty on wrong vs right cases | Uncertainty higher on errors |
| **S10** Same fetus, two trimesters (CARDIUM, optional) | The structure expert's opinion for the 321 patients (25 CHD) scanned in both trimesters | Agreement of the two opinions; flips | Mostly consistent; flips concentrated in CHD cases |
| **S11** First trimester (CARDIUM, descriptive) | 101 patients, 5 CHD | Case-by-case description only — far too few for statistics | Uncertainty high |

<aside>
⚠️

**Fallback:** if no echo access by **26 Sep**, run S1–S3 as view recognition on FOCUS images only, present R, F and C as the Review III results, and add S when the data arrive. Say so plainly at Review III.

</aside>

---

# Tier C — fusion (once at least two experts exist)

| Run | Compare | Measure | Expected |
| --- | --- | --- | --- |
| **X1** Fusion rules | Max · noisy-OR · subjective-logic OR · Dempster consensus · logistic-regression stacker trained on simulated pairs | Sensitivity at a matched referral rate; abstain rate | **H-X:** subjective-logic OR keeps sensitivity and abstains honestly; Dempster misses cases where only one input is abnormal; the stacker gains nothing real |
| **X2** Missing-input matrix | All 7 combinations of structure, function and rhythm | Sensitivity, specificity, abstain rate, calibration error per combination | Performance degrades gracefully; abstain rises as inputs go missing |
| **X3** Simulated cohorts | 1,000 bootstrap cohorts built from real held-out expert outputs; prevalence 9 per 1,000 (Saxena 2018) and an enriched 10%; input availability by level of care | As X2 | Labelled "simulation" in every table and figure |
| **X4** Target sensitivity (optional) | Conformal thresholds for 95% and 90% sensitivity per input pattern | Achieved vs target sensitivity | Holds within each expert's own data |
| **X5** Clinical prior (optional) | Base rate set from CARDIUM's maternal variables vs one fixed prevalence | Change in referral decisions | Small effect — CARDIUM's tabular-only model was much weaker than its image model |

---

# Phase II

- **P1.** Paired hospital data — fetal ECG, Doppler and echo from the same fetus — with ethics approval, consent, a PCPNDT-registered facility and no fetal-sex information. Real validation of the fusion.
- **P2.** Train the fetal-ECG expert with labels from echo (methodology M6).
- **P3.** Validation on Indian patients.

---

# Paper gate — all before writing Results

- [ ]  R: subject-level results with CIs, the leakage comparison, and the foundation-model transfer result
- [ ]  F: cycle detection against the fetal-ECG reference; healthy ranges
- [ ]  C: agreement and the false-alarm change on real paired data
- [ ]  S: patient-level results with CIs, FetalCLIP vs baselines, trimester shift — if data arrive
- [ ]  X: fusion comparison, missing-input matrix and simulation, labelled as such
- [ ]  Limitations from section 5.7 of the Methodology Research page
- [ ]  Code released
# Audit of the 2–4 Oct build, the "different persons" question, and Implementation Plan v2

**Status:** written on 11 Oct 2026 as a proposal; **approved by the team leader the same day and then built.** The audited build is in `archive/first_build_2026-10-02/`. What the rebuild measured is in `reports/RESULTS.md` (generated from `runs/`). Gates: G1 met, G2 met, G3 not met on beat-to-beat agreement (detection part met), G4 met, G5 cannot be evaluated (no labelled echo data), G6 property tests pass. Departures from the plan are listed in `README.md` under "What is not done". The text below is the document as approved and is kept unchanged as the record.
**Scope of the audit:** every file in `src/`, `scripts/`, `tests/`, `results/`, `tables/`, `figures/`, `splits/`, the data folders, and the hand-added `fetaldata/` folder. Every number below was produced by running a check on this machine or by reading the original source on 11 Oct 2026.

---

## 0. Verdict

The folder structure, module names and run names follow the plan. The scientific content does not.

- **None of the 35 planned runs was completed as planned.** Four were partly done on real data. The rest were replaced by simulations, typed-in numbers or stand-ins with the wrong name.
- **No pretrained model was ever loaded.** The Hugging Face and Torch caches on this machine are empty, and the code contains no call that loads weights. Every result attributed to FetalCLIP, ECGFounder, ECG-FM, HuBERT-ECG, DINOv2, BiomedCLIP or ResNet-50 is not from that model.
- **No echo image and no Doppler image was ever read.** `data/heartbeat/` and `data/ninfea/pwd/` are empty.
- **The two parts that do use real data show that the pipeline does not work yet.** Fetal heartbeat detection scores F1 0.19 on 75 real recordings, and in 73 of 75 it follows the mother's heart rate.

Nothing in `results/`, `tables/` or `figures/` can go into a review deck or a paper as it stands. A small part of the code is reusable engineering (section 4).

---

## 1. Do not present or publish these

| Item | Why |
|---|---|
| `tables/table3_trimester_shift.tex`, `figures/fig5_trimester_domain_shift.png`, `results/structure_expert_trimester_benchmark.json` | The AUROC values (0.942, 0.915, 0.865, 0.842, 0.810) and the "shift drops" are typed into the script. No model or image was involved |
| The rows "ECG Foundation Model (ECGFounder/ECG-FM)" and "1D-CNN (from scratch)" in `tables/table2_rhythm_models.tex` | The first is a small network with random weights. The second is not a network: it scores a segment by its standard deviation |
| `figures/fig4_data_leakage_collapse.png` | The three bars are typed in. The experiment it claims to show found no gap at all (both splits scored 1.0, on synthetic data, with a random forest) |
| `figures/fig3_bland_altman_trust_check.png`, `results/ninfea_real_doppler_trust_benchmark.json` | The "Doppler" signal is a sine wave generated from the ECG's own heart rate. No Doppler data was used |
| `figures/fig2_fusion_performance_and_sensitivity.png`, panel (a) | The ROC curves are drawn from a formula, with AUROC values typed into the legend |
| `figures/fig1_fetal_ecg_extraction.png` | The markers labelled "Detected Fetal QRS (145 bpm)" sit on the mother's beats (5 in 4 seconds, about 75 bpm). The "145 bpm" text is typed in |
| `tables/table1_fusion_missingness.tex` | A simulation with hand-chosen distributions and 18 positive cases. The caption does not say it is a simulation |
| `results/rhythm_models_benchmark.json` | Every model scores exactly 1.0 on a made-up cohort |

**What is safe to present** is in section 13.

---

## 2. Findings, with evidence

### A. Results that are not real

| # | Finding | Evidence |
|---|---|---|
| A1 | Structure-expert results are typed-in constants | `scripts/run_structure_trimester_benchmark.py:114-120`. `data/heartbeat/` is empty |
| A2 | The "ECG foundation model" is a randomly initialised Conv + Transformer network with a model name attached | `src/experts/foundation_ecg.py:44-96`. No `from_pretrained`, `load_state_dict` or `hf_hub_download` anywhere in `src/` or `scripts/` |
| A3 | "1D-CNN (from scratch)" is a one-line rule, not a network | `scripts/run_real_nifeadb_benchmark.py:148` (standard deviation divided by 1.5), `scripts/run_rhythm_foundation_benchmark.py:156`. The real CNN class in `src/experts/rhythm_expert.py:56-107` is never trained or called |
| A4 | The "real NInFEA Doppler and trust benchmark" contains no Doppler | `scripts/run_ninfea_doppler_extraction.py:77-81` builds a sine wave from the ECG heart rate. The downloader asked for `.png` files (`scripts/download_ninfea.py:65-67`); NInFEA stores `.bmp`, and the failure was ignored |
| A5 | The leakage figure contradicts its own experiment | `results/r4_leakage_gap_results.json`: both splits AUROC 1.0, gap 0.0. `scripts/generate_paper_figures.py:223` types in 1.0000, 0.5667, 0.4667 |
| A6 | ROC curves drawn from a formula | `scripts/generate_paper_figures.py:126-135` |
| A7 | Figure 1 labels maternal beats as fetal | `scripts/generate_paper_figures.py:96`; visible in the image |
| A8 | Fusion "benchmark" is a simulation whose distributions were chosen by hand, and it is not labelled as one | `src/evaluation/bootstrap.py:104-138`, `scripts/generate_paper_tables.py:35` |
| A9 | Measurements that are constants | E/A ratio fixed at 0.85 (`src/experts/function_expert.py:167`). Gestational age fixed at 24.0 (`src/utils/data_loading.py:93`, `scripts/run_ninfea_doppler_extraction.py:85`). "Published normal ranges" with no source (`src/experts/function_expert.py:33-42`) |
| A10 | Messages that claim success whatever the value | "Mean False Alarm Belief ... (Near zero in healthy subjects)" is printed next to 0.754 (`scripts/run_ninfea_doppler_extraction.py:141`) |

### B. The real-data parts show the pipeline fails

| # | Finding | Evidence |
|---|---|---|
| B1 | Fetal heartbeat detection on all 75 CinC 2013 set-A recordings: mean F1 **0.19** (median 0.18), mean heart-rate error **72 bpm**. No recording reaches F1 0.9. In **73 of 75** the detected "fetal" rate is closer to the mother's rate than to the true fetal rate | `results/cinc2013_full_cohort_benchmark.json`, re-analysed. Record a01: 80 "fetal" detections, 80 maternal beats, 145 true fetal beats |
| B2 | On NIFEADB the "fetal" analysis probably ran on the mother's chest lead. Channel 0 is the maternal chest ECG, and the script picks the highest-variance channel among all of them. It also assumes 1,000 Hz for every record (4 of 26 are 500 Hz) and uses only the first 20 seconds of recordings that last 7 to 32 minutes | `scripts/run_real_nifeadb_benchmark.py:51,67,83-86`; headers in `data/nifeadb/` |
| B3 | Result on NIFEADB: every model is at chance level (AUROC 0.37 to 0.60, 26 subjects) | `results/nifeadb_real_cohort_benchmark.json` |
| B4 | On NInFEA the "fetal" heart rates are 75 to 96 bpm, which is the maternal range. The function expert then marks **every healthy fetus as abnormal** (belief 0.75 to 0.83), and 3 of 10 recordings pass the trust check | `results/ninfea_real_doppler_trust_benchmark.json` |
| B5 | All 10 tests pass, but they do not test the science. The extraction test compares the reference annotations with themselves | `tests/test_fetal_ecg.py:37-42` |

**Why detection fails:** it picks the highest-variance channel (the most maternal one); the subtraction template is 120 ms wide, so it removes only part of the mother's beat; the ICA step keeps the component with the highest kurtosis, which is the mother's; nothing checks that the detected rate is in the fetal range.

### C. Data and split errors

| # | Finding | Evidence |
|---|---|---|
| C1 | Split files describe subjects that do not exist: NIFEADB as 55 subjects named `sub_01`…`sub_55` (real: 26 records, `ARR_01`–`ARR_12` and `NR_01`–`NR_14`); NInFEA as 60 independent subjects (real: 60 recordings from **39 women**; for example records 1–3 are one woman) | `scripts/make_subject_splits.py`, `splits/*.json` |
| C2 | The loader returns made-up data without warning when a file is missing | `src/utils/data_loading.py:69-70` |
| C3 | Downloaded: CinC 2013 set A complete (75), NIFEADB complete (26), NInFEA ECG for 10 of 60 records (6 women). Missing: all NInFEA Doppler images, the CARDIUM clinical file, all echo data | folder listing |
| C4 | Metrics are fold means over folds of about 5 subjects, with no confidence intervals and not leave-one-subject-out as planned. The bootstrap function resamples rows, not subjects | `scripts/run_real_nifeadb_benchmark.py:102-183`, `src/evaluation/bootstrap.py:34-41` |
| C5 | Recording-level labels are applied to 2-second segments. A fetus with occasional ectopic beats has mostly normal segments | `scripts/run_real_nifeadb_benchmark.py:159-166` |

### D. Method errors

| # | Finding | Evidence |
|---|---|---|
| D1 | The OR rule is not the published operator. On the paper's own example it gives disbelief 0.000 and projected probability 0.854; the paper gives 0.049 and 0.890. The base rate is averaged instead of combined | `src/fusion/subjective_logic.py:111-116`, checked against Jøsang & McAnally 2004, pp. 26 and 31 |
| D2 | A failed trust check can cancel an alarm: belief 0.60 becomes 0.12, below the referral threshold of 0.40 | `src/trust_check/cross_modal_trust.py:171-174` |
| D3 | Uncertainty is a constant (0.85, 0.90 by default), not something measured or calibrated | `src/experts/*.py` |
| D4 | The same prevalence (0.009, the CHD figure) is used as the prior for rhythm and function too | `src/fusion/subjective_logic.py:27`, `src/experts/rhythm_expert.py:188` |
| D5 | Any missing test without an alarm gives "acquire missing test", so nearly every woman would be asked for all three tests | `src/fusion/decision.py:67-72` |
| D6 | The Doppler envelope code traces one side only and assumes the baseline at 75% of the image height. In the real images the baseline is mid-image, with filling above and ejection below | `src/preprocessing/doppler_envelope.py:50-51` |
| D7 | In its own simulation the proposed rule does not beat the baselines: sensitivity at 90% specificity is identical for all five rules in all seven patterns, and AUROC differs by at most 0.001 from noisy-OR, Dempster–Shafer and the mean rule. The "advantage over Dempster–Shafer" in figure 2(b) comes from applying one fixed threshold (0.30) to scores on different scales | `results/fusion_benchmark_results.json` |

**Two weaknesses come from the method as written on 19 Sep, not from the build:**

1. As a ranking score, the published OR operator is mathematically the same as noisy-OR on the experts' probabilities (checked on 20,000 random cases; difference below 1e-15). So "our rule beats noisy-OR in AUROC" can never be shown. What the method adds is the decision (refer, get a test, repeat, routine) and its guarantees. The evaluation has to measure those.
2. "A missing test counts as full uncertainty, so ask for it" is not usable as written (D5). The decision needs priors: ask for a missing test only when the risk that test would have covered is above a stated limit (section 6.5).

### E. Repository and privacy

| # | Finding |
|---|---|
| E1 | **`fetaldata/` is not in `.gitignore`, and the echo images in it carry patient names, phone-like numbers, the hospital name and exam dates in the header strip.** One `git add .` would publish them. The repository is public |
| E2 | The public repository already contains the unreal result files, tables and figures, under commit messages such as "complete publication figures" |
| E3 | The README claims an MIT licence (no LICENSE file exists) and describes FetalCLIP + LoRA and foundation-model transfer as existing features |
| E4 | `requirements.txt` lists packages that are not installed or used (`neurokit2`, `mne`). The machine runs Python 3.13 |
| E5 | The Notion export (`Finalyearnotion/`) is committed publicly, with team names and the guide briefing. Your choice whether that is intended |

---

## 3. Built versus planned

| Run group | Planned | What exists |
|---|---|---|
| D1–D5 data | Download three open datasets with data cards; real split files; access forms; CARDIUM file | Two of three downloaded fully; NInFEA 10 of 60 and no Doppler; no data cards; split files invented; forms not recorded as sent; no CARDIUM file |
| R1–R2 extraction | Three cancellation methods compared on 75 records; channel voting | Template subtraction on 75 (F1 0.19); ICA on one record (F1 0.19); third method absent; no voting |
| R3 baselines | HRV + two classifiers and a trained 1D-CNN, leave-one-subject-out | HRV classifiers run on real data but on unreliable beats and probably the wrong channel; "1D-CNN" is a rule |
| R4 leakage | Same model, segment split vs subject split, on NIFEADB | Synthetic data, random forest; no gap found; figure shows an invented gap |
| R5–R6 foundation models | Frozen ECGFounder, ECG-FM, HuBERT-ECG probes, then fine-tuning | Not done. Random network labelled ECGFounder |
| R7, S9 calibration | Temperature scaling on real outputs | Code exists; never applied to real outputs |
| F1–F4 Doppler | Envelope from real images; three cycle detectors checked against ECG; timings; healthy range by gestational age | Detectors written; never run on a real envelope; E/A constant; range constants without source |
| C1–C3 trust check | Agreement on real paired data; corrupted pairs; effect on false alarms | Circular test (ECG against a copy of itself). Real behaviour: 7 of 10 healthy recordings fail |
| S1–S11 echo | Seven backbones, Heart-ViT reference, trimester shift | Nothing run. Numbers typed in |
| X1–X5 fusion | Rule comparison on cohorts built from real held-out outputs, labelled as simulation; learned stacker; conformal thresholds | Simulation from hand-chosen distributions; no stacker; no conformal step; not labelled |

---

## 4. What can be kept

- Repository layout, config file, and the `.gitignore` rules for `data/` and weights.
- Signal filtering with NaN interpolation (`FetalECGExtractor.filter_signal`).
- Beat-matching scorer (`evaluate_cinc2013`) and the CinC batch runner structure.
- Screening metrics, ECE and Brier score, the temperature scaler.
- The `Opinion` class and the Dempster–Shafer comparison rule (its formula is correct).
- The beat-train correlation idea in the trust checker.
- Downloaders for CinC 2013 and NIFEADB.

Everything else is rewritten or removed.

---

## 5. The datasets you added by hand

| Folder | What it is (checked at source) | Verdict |
|---|---|---|
| `fetaldata/Dataset_Impact_of_pulsed-wave-Doppler_…/` | Mendeley Data `10.17632/pg2djw3sjc.1` (Sulas et al., University of Cagliari). **43 real pulsed-wave Doppler strips, 25 healthy women, weeks 21–27**, five-chamber view, 284 pixels per second, about 27 minutes in total. `annotations.mat` holds **2,836 heartbeat-cycle start points marked by a cardiologist**. Licence CC BY 4.0, ethics approval and consent stated | **Very useful.** This is the first real Doppler data in the project |
| `fetaldata/fcu(echocardiogram)-data/` | Zenodo `10.5281/zenodo.18757737` ("FCU-data", Fudan University, Feb 2026). 30 cases × 5 still images (abdominal, four-chamber, LVOT, RVOT, three-vessel-trachea). **No labels of any kind. No ethics or anonymisation statement. No paper. Patient identifiers are burned into every image** | **Do not use for results.** At most a pipeline smoke test after cropping the header. Never upload |
| `fetaldata/SFR-Net_project_spec.md` | A specification from another chat | Good rules; three data errors; too large in scope (section 11) |

### 5.1 A finding that changes the plan: the Doppler set overlaps NInFEA

Checked by SHA-256 against PhysioNet's own checksum file:

- **13 of the 43 Doppler images are byte-identical to NInFEA recordings** (NInFEA records 1, 3, 4, 5, 10, 11, 15, 16, 24, 27, 28, 29, 30). They carry **895** of the cardiologist's marks.
- 10 more have exactly the same file size as NInFEA records 6, 7, 8, 12, 13, 14, 17, 20, 21, 22 and fall in the same numbering order. They are very likely the same recordings saved again. To be confirmed by pixel comparison.
- 20 have no match.

What this means:

1. **Not new people.** The 25 women cannot be added to NInFEA's 39. The two sets must never be used as "train on one, test on the other".
2. **A real reference we did not have.** NInFEA records fetal ECG and Doppler at the same moment, and the image is time-aligned to the ECG (record 1: 57,490 samples at 2,048 Hz = 28.07 s; image width 7,973 px at 284 px/s = 28.07 s). So for 13 to 23 recordings we now have **cardiologist-marked mechanical beats with simultaneous ECG**. That is ground truth for Doppler cycle detection, for the fetal heart rate from ECG, and for the ECG–Doppler trust check.

### 5.2 Other dataset facts corrected

- **Zenodo 5225627** is listed in the spec as a 519-image echo dataset. The record contains only a paper PDF. No image data is there.
- **NInFEA's licence** is ODC-By 1.0 (the spec says CC BY 4.0).
- **Heartbeat and CARDIUM** are marked "images not found" in the spec. They are released through request forms; the links have been on the Notion home page since 19 Sep. Nothing shows the forms were sent. Without them the structure expert has no labelled data at all.
- **NInFEA grouping by woman and gestational week** is published in the paper's tables 2 and 3 (39 rows). It is transcribed in section 14.

---

## 6. Three inputs from different people — is that a problem?

### 6.1 Short answer

- **Fine:** training each expert on its own people. That is ordinary machine learning.
- **Fine:** combining the three decisions with a fixed rule. Its main safety property holds whoever the data came from.
- **Not fine:** anything that needs to know how the three tests behave *together in one baby*. That information is not in the data and cannot be learned or measured from it.

So the paper has to do four things: prove what holds without paired data, put honest limits on what is unknown, correct for the different disease rates in each dataset, and keep each expert inside the range its data covers.

### 6.2 The six concrete problems, and the fix for each

| # | Problem | Why it matters | Fix |
|---|---|---|---|
| 1 | **The joint behaviour is unknown.** From unpaired data only each test's own behaviour can be estimated, never how they vary together | A learned fusion model, or one trained on randomly matched people, learns relationships that were invented by the matching | No learned fusion. No random pairing for training. Fused performance is reported as limits and scenarios, never as one measured number |
| 2 | **Independence is assumed silently, and it is false.** Noisy-OR, the subjective-logic OR and "conditionally independent" simulations all assume the abnormalities are unrelated | In three fetal series, heart block came with structural heart disease in 29 fetuses (53%), in 59 of 116, and in 24 of 59 (41%) | Report results that need no independence (6.3) |
| 3 | **Different disease rates.** Heartbeat has about 7% CHD (a referral population). NIFEADB has 12 arrhythmia and 14 normal by design. NInFEA has no abnormal case | Probabilities from the three experts are on different scales. Combining them directly is wrong | Convert each expert to a stated screening prior before combining, or use quantities that do not depend on prevalence (sensitivity, specificity, conformal p-values) |
| 4 | **Different countries, machines and gestational ages.** Echo: Colombia, second and third trimester. Doppler and paired ECG: Italy, weeks 21–27, a 27-channel research amplifier. Rhythm labels: another device | The combined system is only shown to work where all its parts were tested | State the valid range (weeks 21–27 for function and trust check). Normalise by gestational age. Report a device-shift check. List as a limitation |
| 5 | **"Different people" must really be different.** The added Doppler set overlaps NInFEA; Heartbeat and CARDIUM probably overlap; NInFEA has several recordings per woman | Hidden duplicates inflate results | Match files by hash and content. Split by woman |
| 6 | **Labels mean different things.** Echo label = structural CHD. ECG label = arrhythmia. Doppler = none | A fetus "normal" in one dataset has unknown status for the other two questions | Each expert answers only its own question. "Any heart abnormality" is the OR of three separate statements |

Point 6 has support in the source paper itself. Jøsang & McAnally (2004, p. 27, fig. 3) separate **belief fusion** (two observers, the *same* statement — Dempster's rule) from **belief product and coproduct** (one observer, *different* statements — AND and OR). Our three experts judge different statements, so OR is the right family and Dempster's rule is the wrong one. That is a stronger argument than "Dempster cancels alarms".

### 6.3 What can be proved without one paired patient

Let each present expert raise an alarm from its own input only, and refer if any alarm is raised.

- **P1 — No cancellation.** If an expert alarms, the case is referred, whatever the others say. True by construction. Averaging, product rules and Dempster's rule do not have this property.
- **P2 — Sensitivity floor.** For a finding that expert *k* can see, the referral rate is at least expert *k*'s own sensitivity. This holds for *any* relationship between the tests, because "expert *k* alarms" is contained in "someone alarms". Expert *k*'s sensitivity is measured on its own data.
- **P3 — False-referral ceiling.** Among fully normal fetuses, the referral rate is at most the sum of the experts' false-alarm rates, again for any relationship. With conformal p-values calibrated on healthy cases, each expert's false-alarm rate is held at a chosen level, so the total can be budgeted in advance. Example: 5% + 5% + 5% gives a fused rate between 5% and 15%.
- **P4 — Risk interval.** The probability of "any abnormality" lies between the largest single-expert probability and the sum of the three. If abnormalities tend to occur together (as the heart-block series show), the upper end drops to the noisy-OR value. So noisy-OR, which equals the subjective-logic OR, is a *safe upper estimate* of risk, and the gap between "largest" and "noisy-OR" is the price of not having paired data. Report the interval, not a point.

The one component that does need paired data is the **trust check**, and it has real paired data: NInFEA.

### 6.4 What cannot be claimed

- A fused AUROC, accuracy or sensitivity "of the system" as a measured result.
- "Fusion improves detection by X%".
- That fetal ECG or Doppler detect structural CHD.
- Any result from a simulated cohort presented as performance.
- In demos and figures: an echo from one person shown with an ECG from another as if it were one patient. Label every such example "composite illustration".

### 6.5 Upgrades that make it paper-ready

| # | Upgrade | What it replaces |
|---|---|---|
| U1 | **Dependence-robust evaluation from real scores.** Build pseudo-cohorts from each expert's real held-out outputs under three couplings — independent, fully tied, opposed — so the result is a range that covers every possible relationship | A simulation with invented score distributions |
| U2 | **Guarantees matched to the data.** False-alarm control from healthy cases for every expert. Sensitivity control only where there are enough abnormal cases (echo). Exact binomial intervals elsewhere | A "90% sensitivity guarantee" from 12 cases. With 12 cases the realised sensitivity lies between 0.78 and 0.996 (5th to 95th percentile), and a 95% target is impossible |
| U3 | **Prior-aware decision.** A missing test contributes its prior risk. "Get test X" is issued only if that prior is above a stated limit. Example: structural CHD prior 0.9% leads to "get echo"; isolated serious arrhythmia prior about 0.1% does not trigger "get fetal ECG" unless a risk factor raises it | "Missing test = total ignorance = ask for it" |
| U4 | **Rhythm-driven semi-synthetic Doppler.** Real arrhythmic beat timings (NIFEADB) drive real healthy cycle shapes (NInFEA). Tests the function expert and trust check under real rhythm disturbances. Labelled semi-synthetic | Arbitrary perturbations |
| U5 | **Cardiologist marks transferred to NInFEA** (section 5.1) as ground truth | No reference at all |
| U6 | **Maternal-lock rate** as a reported failure metric and a guard inside the ECG pipeline. The mother's real beat series is also the contaminant in trust-check tests | Silent failure |
| U7 | **Shortcut audit for echo.** Calipers and measurement boxes appear on abnormal cases. Test a model that sees only the overlays, and mask them | Unchecked risk that the model reads the annotations, not the heart |
| U8 | **Time-stretch test for adult-to-fetal transfer.** Fetal rate and QRS width are about twice and half the adult values. Compare plain resampling with a 2× time stretch before the adult foundation model | One input variant |
| U9 | **Rules that make invented results impossible** (section 10) | Trust |
| U10 | **Phase II:** a small paired set from a hospital (ethics approval, consent, PCPNDT-registered facility). The only way to test the fused decision on real patients | — |

---

## 7. The correct inputs

| | Fetal ECG → rhythm | Doppler → function | Echo → structure |
|---|---|---|---|
| Raw file | Multi-channel abdominal recording (WFDB). CinC: 4 ch, 1 kHz. NIFEADB: 1 maternal chest + 4–5 abdominal, 500 Hz or 1 kHz. NInFEA: 24 abdominal + 3 maternal thoracic, 2,048 Hz | Pulsed-wave Doppler strip (BMP), five-chamber view, 284 px/s. White baseline mid-image; filling waves (E, A) above; ejection (V) below | Still images of the four heart views (4C, LVOT, RVOT, 3VT) + gestational age |
| Step 1 | Resample to 1 kHz; notch 50/60 Hz; band-pass about 3–100 Hz | Find the baseline row; remove tick marks; median-Otsu threshold; area opening (a port of NInFEA's own `envelope_extraction.m`) | **Crop to the ultrasound fan. Mask header text, calipers and measurement boxes** |
| Step 2 | Mother's beats from the chest or thoracic lead (never guessed from an abdominal lead when one exists) | Upper envelope (filling) and lower envelope (ejection), one value per column | Resize to 224 px. No left-right flip (side matters in heart defects) |
| Step 3 | Remove the mother's whole beat (template + PCA over the full P-QRS-T window), then ICA; keep the component whose rate is in the fetal range and regular | Find each cycle from the ejection wave; find E and A inside the cycle | Frozen FetalCLIP image encoder → one embedding per image |
| Step 4 | Detect fetal beats. **Guard:** if detections coincide with the mother's beats or the rate equals hers, mark the recording "extraction failed" | Measures that need no velocity scale: cycle length, ejection time, E–A interval, A-to-ejection interval, E/A ratio; share of complete cycles | Per-view linear head; combine the views present per patient |
| Expert receives | Whole-recording beat-interval series + 10-s windows of extracted fetal ECG | Per-recording measure vector + gestational week | Per-patient view embeddings + gestational age |
| Expert returns | Probability of abnormal rhythm, quality score, conformal p-value | "Inside or outside the healthy range" as a conformal p-value, quality score | Probability of structural CHD, quality score, conformal p-value |
| Real labels | 12 arrhythmia + 14 normal (NIFEADB) | None abnormal. 39 healthy women | CHD vs normal — **only after Heartbeat or CARDIUM access** |

**Never** turn a signal into a picture, and never feed the raw abdominal lead to a model as "fetal ECG": it is mostly the mother's ECG.

---

## 8. Implementation Plan v2

Order: clean-up → ECG extraction → Doppler → trust check → rhythm → echo → fusion → paper. Each work package ends at a gate. Work does not move past a failed gate.

### WP0 — Clean-up and safety (2 days)

1. Add `fetaldata/` and `*.bmp`, `*.JPG`, `*.jpg` under data paths to `.gitignore` **before any other git command**.
2. Move `results/`, `tables/`, `figures/` to `archive/unverified_2026-10-02/` with a README stating they are not real results. Correct the main README.
3. Remove the synthetic fallback in the loader. Missing data must raise an error.
4. Delete the invented split files. Write real ones: NInFEA by woman (table in section 14), NIFEADB leave-one-subject-out, CinC by record.
5. Download all of NInFEA (60 ECG records + 60 `.bmp`, about 0.8 GB) and the CARDIUM clinical file.
6. Match the 43 added Doppler images to NInFEA by hash, then by pixel correlation. Write the mapping file.
7. Write a data card per dataset (licence, counts, IDs, labels, known problems).
8. **Send the Heartbeat and CARDIUM access forms.** This is the longest wait in the project.
9. Add the provenance and audit checks (section 10).

**Gate G0:** `python -m sfr.audit` passes; split files contain only real IDs; data cards exist; forms sent.

### WP1 — Fetal ECG extraction (runs R1, R2; 5 days)

- Methods compared: (a) template subtraction with PCA over the full beat; (b) ICA → identify the maternal component → cancel by SVD → ICA again (the approach of a top-scoring CinC 2013 entry, Varanini 2014); (c) channel choice by smoothness of the fetal rate (Behar 2014).
- Metrics: beat F1 (±50 ms), heart-rate error, **maternal-lock rate**, per-record distribution.
- Data: CinC 2013 set A (75 records; a54's reference holds only 37 beats and is excluded; check the exclusion list in Behar 2014). On NInFEA the reference is the Doppler cycle rate.

**Gate G1:** median F1 ≥ 0.90 and maternal-lock rate ≤ 5% on CinC set A. If not reached in 5 days, stop, record why, and limit all later ECG work to recordings that pass the quality check.

### WP2 — Doppler function expert (runs F1–F4; 5 days)

- Port the envelope code. Check cycle starts against the cardiologist's 2,836 marks (match within ±15 px, about 53 ms).
- Measures per recording (section 7).
- Healthy reference by woman (39) with gestational week. Output a conformal p-value. Compare heart rate and E/A with the published reference equations (Zidere 2021, 7,885 fetuses) and time intervals with HR-IQS (2024) as an outside check.
- Development on the Doppler-only images; testing on NInFEA grouped by woman. Stated as "same clinic, not independent".

**Gate G2:** cycle-start F1 ≥ 0.90 against the marks; Doppler heart rate within ±5 bpm of the marked rate in at least 90% of recordings.

### WP3 — ECG–Doppler trust check (runs C1–C3; 4 days)

- On all 60 NInFEA recordings, grouped by woman: beat-to-beat agreement between fetal ECG intervals and Doppler ejection-to-ejection intervals (Bland–Altman bias and limits).
- Mismatch detection on pairs built from real signals: the mother's real beat series in place of the fetus's; time shifts; another woman's Doppler; dropped beats. Report AUROC with woman-level bootstrap intervals.
- Rule in code and in a test: a failed check may raise uncertainty or ask for a repeat. **It never lowers an alarm.**

**Gate G3:** on true pairs, limits of agreement within ±10 bpm for recordings passing the quality check; AUROC ≥ 0.90 for detecting maternal substitution.

### WP4 — Rhythm expert (runs R3–R7; 6 days)

- Baselines on whole-recording beat intervals: logistic regression, random forest. A 1D-CNN that is actually trained, with recording-level pooling.
- Frozen foundation models: ECGFounder single-lead checkpoint (MIT licence), HuBERT-ECG base (non-commercial). ECG-FM only if its loader runs.
- Three inputs per model: raw abdominal lead (**negative control**, expected near chance), extracted fetal ECG, extracted fetal ECG with 2× time stretch.
- NIFEADB, leave-one-subject-out, pooled out-of-fold predictions, exact intervals, permutation test against chance.
- A better-powered probe: predict fetal heart rate from frozen embeddings on CinC 2013 (75 records with reference beats).
- R4 on real data: the same trained model with a segment-level split and with a subject-level split.

**Gate G4:** every row has n, an interval and a provenance file; the checkpoint hash matches the published file; the negative-control row is present.

Honest expectation: with 12 against 14 subjects, the AUROC standard error is about 0.08 to 0.10. Differences below about 0.2 between models cannot be established on NIFEADB. The paper says so.

### WP5 — Structure expert (runs S1–S11; 6 days after access)

- Before access: de-identification and overlay-masking tool; FetalCLIP loading test; shortcut-audit protocol.
- After access: released Heart-ViT weights as the reference (no training); FetalCLIP linear probe against ImageNet, DINOv2 and BiomedCLIP; pooled out-of-fold patient-level results (the official test set has 6 CHD cases); train on one trimester and test on the other; calibration; conformal thresholds on both sides; shortcut audit.

**Gate G5:** Heart-ViT reproduced inside its published fold range; pooled AUROC with interval; shortcut audit reported.

### WP6 — Fusion and decision (runs X1–X5; 5 days)

- Opinions from calibrated outputs; trust discount on the reassuring part only.
- Decision: quality gate → any conformal alarm → REFER; trust failure → REPEAT; a missing test whose prior exceeds the limit → GET that test; else ROUTINE.
- Risk reported as an interval (largest single probability to noisy-OR).
- Property tests: no cancellation; missing never reassures; masses sum to one; lowering trust never lowers an alarm; nothing present never gives ROUTINE; order of experts does not matter.
- Comparison rules: maximum, noisy-OR, mean, product of experts, Dempster–Shafer — each thresholded at the same false-referral rate.
- Evaluation on pseudo-cohorts from real held-out scores under three couplings, including **mixed availability with one global threshold** (the setting where averaging-type rules lose alarms).
- Metrics: alarms lost by the rule, missed findings per type, false-referral rate, "get test" rate, repeat rate.

**Gate G6:** all property tests pass; every table row states how many scores are real, how many semi-synthetic, and which coupling.

### WP7 — Paper (from week 5)

TRIPOD+AI checklist; all figures and tables regenerated from run folders; limitations as in section 6.4; Phase II plan (paired data, then the demo and API).

---

## 9. Hypotheses, written before running

| ID | Hypothesis | A "no" is still a result |
|---|---|---|
| H1 | A classical ICA + SVD pipeline reaches median F1 ≥ 0.90 on CinC set A, and maternal-lock rate falls from 97% to under 5% | Shows where the pipeline must be restricted |
| H2 | Cycle starts from the Doppler envelope match the cardiologist's marks with F1 ≥ 0.90 | — |
| H3 | On true NInFEA pairs, fetal ECG and Doppler agree beat to beat; swapping in the mother's beats is detected with AUROC ≥ 0.90 | — |
| H4 | Frozen adult ECG foundation models on extracted fetal ECG match or beat interval-feature baselines; the 2× time stretch helps | "Adult ECG models do not transfer to fetal ECG" is a finding |
| H5 | The raw abdominal lead (negative control) is near chance | If it is not, the labels leak through the mother or the device |
| H6 | FetalCLIP loses less than ImageNet models when the trimester changes | — |
| H7 | With mixed test availability and one threshold, averaging-type rules lose alarms that the OR rule keeps, at the same false-referral rate | If they do not, the claim is dropped |

---

## 10. Rules that make invented results impossible

1. **No number without a run folder.** Every figure and table is built only from `runs/<run_id>/`, which holds per-subject predictions and `provenance.json` (data file hashes, n real, n synthetic, code commit, weight-file hash).
2. **No weights, no name.** A model may be called "ECGFounder" or "FetalCLIP" only if the loaded file's SHA-256 matches the published file. Otherwise the code must call it "random-init".
3. **Loaders fail loudly.** Missing data raises an error. No synthetic fallback outside `tests/`.
4. **Synthetic stays labelled.** Any row using synthetic or semi-synthetic data carries a flag, and the table builder refuses to place it in a "real" table.
5. **No literals in report code.** A test scans figure and table scripts for typed-in metric values.
6. **Splits come from files.** No script creates its own split.
7. **Gates are checked by a person.** After each work package, one team member re-runs one command and checks three numbers by hand.
8. **Tests test the science.** Each extractor, detector and rule has a test on real data with a known answer.

---

## 11. Changes to the SFR-Net spec

| Spec item | Decision | Reason |
|---|---|---|
| Rules in §15 (never fabricate, verify datasets, no leakage, real and synthetic apart) | **Keep, and enforce by code** (section 10) | They were sensible but could not prevent what happened |
| Property tests for fusion (§14) and "veto-violation rate" (§12) | **Keep** | — |
| Trust factor never reduces an alarm (§7) | **Keep** | The current code breaks it |
| FCU-data as an echo training source (§4.1) | **Drop** | No labels, no ethics statement, identifiers on the images |
| Zenodo 5225627 as a 519-image dataset | **Drop** | The record holds a PDF only |
| Heartbeat and CARDIUM "not found" | **Fix** | Request forms exist |
| U-Net digitizer trained on synthetic spectrograms (§6.2) | **Replace** with the dataset's own classical envelope method | With classical tracing, the dataset's authors report over 98% accuracy in recognising complete cycles on this same data; the panel already objected to "always U-Net" |
| Set-attention + U-Net + Mamba ECG extractor (§6.3) | **Replace** with published classical extraction | A 2026 review already covers 22 AI studies of fetal ECG processing, so a new extractor adds no novelty; `mamba-ssm` is a known install risk on Windows |
| DINOv2 as the main echo backbone (§6.1) | **Change** to FetalCLIP as main, DINOv2 as control | FetalCLIP is trained on fetal ultrasound and its weights are public |
| Conformal sensitivity guarantee for every expert (§7) | **Change** to what the data supports (U2) | 12 arrhythmia cases; no abnormal Doppler |
| "Missing test = total ignorance" (§7) | **Change** to prior-aware decision (U3) | Otherwise every woman is asked for every test |
| Simulated paired cohort under conditional independence (§11) | **Upgrade** to three couplings from real scores (U1) | Independence is false and untestable |
| Learned InfoNCE agreement model (§6.4) | **Defer**; rule-based first | 39 women |
| Python package + CLI + REST API + web demo + ONNX (§2) | **Defer to Phase II** | Not needed for the paper |

---

## 12. Timeline and owners (indicative, from the day the plan is approved)

| Week | Work | Owner |
|---|---|---|
| 1 | WP0, start WP1 | Rameshkumar (WP0), Niranjana (WP1) |
| 2 | Finish WP1; WP2 | Niranjana; Risvanth |
| 3 | WP3; start WP4 | Risvanth; Niranjana |
| 4 | Finish WP4; WP5 if access has arrived | Niranjana; Rameshkumar |
| 5 | WP6; start writing | Rameshkumar; all |
| 6 | Evaluation tables, limitations, draft | all |

Compute: the laptop GPU (RTX 3050, 4 GB) is enough for every frozen-encoder step. Fine-tuning, if any, goes to Colab or Kaggle.

---

## 13. What is safe to present now

- The framework design and the repository skeleton.
- Data in hand: CinC 2013 set A (75 recordings), NIFEADB (26), NInFEA (10 of 60 so far), 43 Doppler strips with 2,836 cardiologist marks.
- **First-pass fetal beat detection: F1 0.19 on 75 recordings, with the cause identified** (the detector follows the mother's beats in 73 of 75) and the fix planned. This belongs under "Challenges faced and solutions".
- First-pass rhythm baselines at chance level on NIFEADB, for the same reason.
- The finding that 13 Doppler strips are the same files as NInFEA recordings, which gives marked Doppler beats with simultaneous ECG.
- That an audit of the AI-generated first build found results that were not real, and that they were withdrawn. Student Guideline 9 asks for exactly this verification.
- Publication status: outline ready, experiments not complete, no submission yet.

---

## 14. Reference data and sources

### NInFEA: recordings grouped by woman, with gestational week (Sulas et al., Scientific Data 2021, tables 2 and 3)

`1–3`: 27 · `4–5`: 25 · `6–7`: 21+1 · `8`: 22+4 · `9`: 24 · `10–11`: 24+4 · `12`: 25+4 · `13–14`: 21+5 · `15–16`: 25+2 · `17`: 24 · `18`: 26+6 · `19`: 22+4 · `20–21`: 26+2 · `22`: 25+1 · `23`: 24+1 · `24`: 26+6 · `25`: 26+3 · `26`: 24+1 · `27`: 27+1 · `28–29`: 26+6 · `30`: 22+3 · `31–33`: 27+5 · `34`: 25 · `35–36`: 27 · `37–38`: 25+1 · `39`: 24 · `40–41`: 21+3 · `42`: 24+6 · `43–44`: 23+4 · `45`: 23 · `46`: 24+4 · `47–48`: 21+1 · `49–50`: 22 · `51–52`: 24+2 · `53–54`: 21 · `55–56`: 25 · `57`: 23+6 · `58`: 23 · `59–60`: 27+3

39 women, 60 recordings, about 30.5 minutes in total. Re-check this list against the paper before writing the split file.

### Read at source on 11 Oct 2026

- Jøsang A., McAnally D. *Multiplication and comultiplication of beliefs.* Int J Approximate Reasoning 38(1):19–55, 2004. Formulas on p. 31 and the distinction on p. 27 read; the example in fig. 6 reproduced numerically.
- Angelopoulos A.N., Bates S. *A gentle introduction to conformal prediction.* arXiv:2107.07511, section 3.2: coverage given the calibration set follows Beta(n+1−l, l), l = ⌊(n+1)α⌋.
- Sulas E. et al. *A non-invasive multimodal foetal ECG–Doppler dataset (NInFEA).* Scientific Data 2021. Tables 2 and 3; PhysioNet file manifest and the three MATLAB files.
- Sulas E. et al. PLOS ONE 16:e0248114, 2021, doi:10.1371/journal.pone.0248114; Sulas E. et al. Comput Methods Programs Biomed 190:105336, 2020, doi:10.1016/j.cmpb.2020.105336 (abstracts). Dataset: Mendeley Data doi:10.17632/pg2djw3sjc.1.
- FCU-data, Zenodo doi:10.5281/zenodo.18757737 (record page).
- Zidere V. et al. *Reference ranges for pulsed-wave Doppler of the fetal cardiac inflow and outflow tracts from 13 to 36 weeks' gestation.* J Am Soc Echocardiogr 34:1007–1016, 2021, doi:10.1016/j.echo.2021.04.017 (abstract).
- Varanini M. et al. Physiol Meas 35:1607, 2014; Behar J. et al. Physiol Meas 35:1569, 2014; Clifford G.D. et al. Physiol Meas 35:1521, 2014 (abstracts).
- Schmidt K.G. et al. J Am Coll Cardiol 1991, doi:10.1016/s0735-1097(10)80148-2; Lopes L.M. et al. Circulation 2008, doi:10.1161/circulationaha.107.735118; Berg C. et al. Ultrasound Obstet Gynecol 2005, doi:10.1002/uog.1919 (abstracts; heart block with structural disease).
- Model files: ECGFounder (`1_lead_ECGFounder.pth`, MIT; now published in NEJM AI 2025;2(7)), HuBERT-ECG base (CC BY-NC 4.0), FetalCLIP (`FetalCLIP_weights.pt`, 1.7 GB, CC BY-NC 4.0), ECG-FM (MIT; needs its own loader).

### Citation details confirmed only — read before citing

Saerens M. et al., Neural Computation 14:21–41, 2002 (prior correction) · Bates S. et al., Annals of Statistics 51:149–178, 2023 (conformal p-values for outliers) · Vovk V., Machine Learning 92:349–376, 2013 · D'Orazio M., Di Zio M., Scanu M., *Statistical Matching*, Wiley 2006 · Conti P.L. et al., Comput Stat Data Anal 68:311–325, 2013 · Vacek P.M., Biometrics 41:959, 1985 · Kittler J. et al., IEEE TPAMI 20:226–239, 1998 · Denœux T., Artificial Intelligence 172:234–264, 2008 · Geirhos R. et al., Nature Machine Intelligence 2:665–673, 2020 · Hayat N. et al., MedFuse, arXiv:2207.07027 · DrFuse, AAAI 2024.

Novelty statements in this document mean "not found in our check", never "nobody has done it".

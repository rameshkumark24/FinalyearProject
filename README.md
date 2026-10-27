# Screening the fetal heart from unpaired tests (Structure–Function–Rhythm)

Final-year project, Department of Computer Science and Engineering (Artificial Intelligence and Machine Learning), KPR Institute of Engineering and Technology.
Team: Rameshkumar K, Niranjana J, Risvanth V. Guide: Dr. Karthick Panneerselvam.

Three tests show three different things about a fetal heart:

| Test | Question | Expert | Labelled public data used |
|---|---|---|---|
| Fetal echocardiography | Is the structure normal? | structure | 752 cases, 133 with a heart defect (DELTA: open, one centre, released Oct 2026) |
| Pulsed-wave Doppler | Is the timing of filling and ejection normal? | function | healthy fetuses only (NInFEA) |
| Non-invasive fetal ECG | Is the rhythm normal? | rhythm | 12 arrhythmia, 14 normal rhythm (NIFEADB) |

No public dataset holds all three tests for the same fetus. Each test therefore gets its own expert, built on its own public data, and the experts are joined by a standard any-alarm rule: refer if any expert alarms. Its false-referral ceiling, the sum of the alarm levels, is the textbook union (Bonferroni) bound and needs no paired patient; what this project adds is evidence about when that ceiling holds in practice (`docs/positioning.md`). Where ECG and Doppler were recorded together (NInFEA), a trust check verifies beat by beat that both follow the same heart.

## Limits first

Every number below comes from a run folder; `reports/RESULTS.md` opens with a claim ledger that lists every claim, how it was tested, and what would overturn it.

- **Most referrals would be false.** At the decision rule's priors (structure 0.009; the function and rhythm priors are assumptions), the design levels give about 4.9 true referrals (structural defects and arrhythmias; the function expert's sensitivity is unknown) and up to 165 false referrals per 1000 people screened (64 to 97 at the measured healthy rates), roughly 34 false referrals for each true one at the ceiling (run `wp10_allocation`; `docs/clinical_context.md`).
- **The echo result is an upper bound.** Inputs with little or no heart anatomy also separate the classes, and among the 642x506 images the structure expert's AUROC is 0.527 [0.283, 0.761] with 0 of 11 defects caught (runs `wp5b`, `wp5c`). Fitted inside Heartbeat, a FetalCLIP probe reaches 0.808, but the parts removed by the cleaning step reach 0.804 too (run `wp5_structure_heartbeat_2T`).
- **Calibration does not travel between devices.** Calibrated on NIFEADB, the rhythm expert flagged 52.6% of the readable healthy NInFEA recordings; calibrated on healthy NInFEA recordings of other women, the ECG-Doppler pair referred 10.0%. Local calibration held from about 10 women (15 recordings) on this one device pair (runs `wp6_fusion`, `wp13_calibration_transfer`). The same holds for the echo expert: DELTA's calibration flagged 12.8% to 21.2% of normal fetuses in Heartbeat and CARDIUM at a 5% level, and local calibration repaired the false alarms but then caught only 3.1% to 6.9% of defects (run `wp17_structure_calibration`).
- **The echo expert failed its external test.** Applied unchanged to 784 second-trimester Heartbeat patients from Colombia (a test fixed in advance), it reached AUROC 0.509 [0.435, 0.583], chance level, and its DELTA calibration flagged 12.8% of normal fetuses at a 5% level; 3rd trimester 0.543, CARDIUM 0.499 (report section 13). The data were read correctly: the authors' own Heart-ViT and five baselines, run from their released checkpoints with their code, reproduce every published test metric within 0.007 percentage points (Heart-ViT AUROC 0.886 on the 2nd-trimester test set; run `wp15_reference_models`). The rhythm and function experts have not been tested on a second dataset.
- **The rhythm expert is a feasibility result on 26 recordings.** With every recording brought to 500 Hz it reaches 0.810 [0.581, 0.994], and 0.714 [0.375, 0.981] on the 22 recordings from one setting; the mother's chest lead, which holds no fetal signal, still scores 0.702, so the four recordings sampled at 500 Hz (all arrhythmia) differ in another way (run `wp4e_rhythm_500hz`).
- **The trust check is a consistency check, and simple baselines nearly match it.** Heart rate alone separates the mother's beats with AUROC 0.966 (locked share 0.984); on swapped pairs a cross-correlation of the beat trains reaches 0.977 against 0.919 (run `wp3b_trust_baselines`). Its agreement gate G3 is not met.
- **Re-splitting the ceiling between the experts gains little.** Levels chosen on half the people and valued on the other half changed true referrals by +0.23 per 1000 [-0.26, +0.71]; the method is an established one (Rubin, Dudoit and van der Laan 2006).
- **Nothing is measured on patients who had all three tests.** That needs paired hospital data.

## Component results

Held-out data unless stated. Every number comes from a run folder; the full tables are in `reports/RESULTS.md`.

| Part | Result | Gate |
|---|---|---|
| Fetal beat detection (CinC 2013, 50 test recordings) | median F1 0.995; the mother's rhythm returned in none | G1 met |
| Doppler cardiac cycles (24 test strips, 1,595 cardiologist marks) | mean F1 0.963 | G2 met |
| ECG–Doppler trust check (NInFEA, 26 test women) | AUROC 0.984 against the mother's beats, 0.919 against another woman's ECG; beat-to-beat limits ±13.3 bpm against a target of ±10 | G3 **not met** (agreement part) |
| Function expert (60 healthy recordings, 39 women) | 2 of 60 flagged at level 0.05; only the heart rate is clearly repeatable | no gate; sensitivity cannot be measured |
| Rhythm expert (NIFEADB, leave one subject out) | one interval index AUROC 0.938; frozen ECGFounder 0.952; the same network with random weights up to 0.881, and the two cannot be ranked; the logistic expert 0.798, but 0.696 [0.347, 0.981] on the 22 recordings sampled at 1 kHz | G4 met |
| Structure expert (DELTA, case-level folds) | AUROC 0.828 (0.878 on the authors' test split); inputs with little or no cardiac anatomy also reach 0.62 to 0.77, so this is an upper bound; defects were caught in 60 of 114 cases imaged at 900x752 but in 0 of 11 at 642x506; **on Heartbeat (external, preregistered) AUROC 0.509** | see the report's gates |
| Heartbeat's released models (authors' code and checkpoints) | all six reproduce their published 2nd-trimester test results within 0.007 points (Heart-ViT AUROC 0.886, sensitivity 79.2%, specificity 93.2%), so the external data are read as the authors intended | G5: reference model reproduced |
| Decision rule | false referrals 10.0% on 60 real healthy ECG–Doppler pairs (ceiling 15%); 7.1% to 16.9% with all three experts' healthy alarm rates combined (ceiling 20%) | G6 met (property tests) |
| Standard bounds without paired data (added after seeing the results) | false referrals 7.1% to 16.9% for any dependence between the tests; 4.5% to 57.4% once the uncertainty of each expert's rate is included; on the 60 real ECG–Doppler pairs the measured 10.0% lies inside its predicted range (8.3% to 11.7%) | G7 met at the measured rates, NOT met with uncertainty |

"Not met" and "partly met" are results. They are reported, not hidden.

## What the standard bounds give without paired data

Two properties of the any-alarm rule hold for any dependence between the three tests, so they need no patient who had all three. Both are standard, not contributions of this project: the first is the union (Bonferroni) bound, the second follows from the rule itself. Property tests check them for independent, tied and opposed experts, for alarms that never come together, for random correlations and for conformal p-values from small calibration sets (`tests/test_guarantees.py`, `tests/test_fusion.py`).

1. **False referrals.** The share of healthy fetuses referred is at most the sum of the alarm levels of the experts used: 20% with three tests, 15% with the ECG and the Doppler. This holds only when every expert is calibrated on healthy cases from the setting where it is used: the rhythm expert calibrated on NIFEADB referred 51.7% of the healthy NInFEA fetuses.
2. **No alarm is cancelled.** A fetus whose finding makes its expert alarm is always referred. This only restates each expert's own sensitivity: combining tests widens which conditions are checked and adds false referrals, but it does not make any single condition easier to detect.

With each expert's alarm rate measured on its own healthy data, the false-referral rate lies between 7.1% and 16.9% for any dependence (run `wp7_dependence_bounds`). These rates rest on small samples: with their uncertainty the range widens to 4.5% to 57.4%, mostly because the rhythm expert is calibrated on 14 healthy recordings. With so few, its p-values are multiples of 1/15, so its level of 0.10 acts as 1/15 and the design's ceiling is in fact 16.7% (run `wp8_tradeoff`).

Any rate on patients who had all three tests is not measured.

## Read this first

- **Results live in `reports/RESULTS.md`**, which `scripts/build_report.py` generates from the run folders. Nothing in it is typed by hand.
- **The heart-defect result rests on one open, single-centre dataset (DELTA), and its controls show that part of the label can be read from images without the heart** (`reports/RESULTS.md`, section 6). Read the structure expert's AUROC as an upper bound, not as screening performance.
- **There is no measured performance of the fused system on patients.** That needs paired data, which does not exist publicly.
- **Sections 9 to 12 of the report were added after the results were known** (16 and 20 October 2026, in answer to two reviews): bounds for any dependence, alarm levels against sensitivity, subgroups, baselines, the rhythm check at 500 Hz, group-conditional echo calibration, trust-check baselines, calibration transfer, budget allocation, and threats to validity. They refit nothing but linear probes, are labelled as checks made after seeing the results, and their intervals are not corrected for the number of checks: read them as exploratory only. **Section 13** (external echo data, 27 October 2026) holds one test fixed in advance (13.2, `docs/preregistration_external_echo.md`); the rest of it is exploratory in the same sense.
- **The code was rebuilt from 11 October 2026.** An earlier build (2–4 October) reported numbers that did not come from real computation. It is no longer part of this repository; it remains in the git history, and none of its numbers is valid.

## Rules that keep results honest

1. **No number without a run folder.** Every figure and table is built from `runs/<run>/`, which holds per-case outputs, `metrics.json` and `provenance.json` (hashes of the data files and of the code, how many cases were real, how many simulated).
2. **No weights, no name.** A model is called "ECGFounder" or "FetalCLIP" only if the loaded file matches the published SHA-256. Otherwise it is called "random-init".
3. **Loaders fail loudly.** Missing data raises an error. There is no synthetic fallback.
4. **Simulated cases stay labelled.** A run that alters real signals or pairs scores of different people says so in its provenance and in every table row.
5. **No typed-in metrics in report code.** `python -m sfr.audit` checks it.
6. **Splits come from files** in `splits/`, fixed before the held-out data were analysed.
7. **Tests test the science**: `tests/test_real_data.py` runs the pipeline on real recordings with known answers.

## Layout

```
sfr/            the package (its hash is recorded in every run)
  ecg/          fetal ECG extraction (maternal cancellation, ICA, beat tracking, guards against the mother's rhythm)
  doppler/      envelope from the strip image, cardiac cycles, timing measures, healthy reference, orientation
  trust/        ECG–Doppler trust check
  rhythm/       interval features, leave-one-subject-out evaluation, ECGFounder loader
  echo/         image cleaning for echo frames, FetalCLIP loader
  fusion/       opinions and their OR, comparison rules, conformal p-values, the decision
  data/         loaders for CinC 2013, NInFEA, NIFEADB and the annotated Doppler strips
  provenance.py, audit.py
sfrtool/       the one-patient research tool: artifacts, scoring of one case, what was and was not checked
               (outside sfr/, so the code hash recorded by every run still holds)
artifacts/     small files the tool needs: refitted linear models, healthy calibration scores, a card with hashes
scripts/        one script per run, run_all.py, build_report.py, export_artifacts.py, predict_patient.py, downloaders, dev_*.py
splits/         development and test definitions, and the DELTA image index
runs/           one folder per run (large arrays are not committed)
reports/        RESULTS.md, tables/, figures/, manifest.json - all generated
docs/           positioning (what is standard, what is new), related work, clinical context, preregistration of the external
                echo test, theory notes, data cards, inference notes, analysis notes
tests/          property tests of the decision rule and its guarantees, tests of the evaluation code and of the tool, tests on real recordings
```

## Reproduce

```bash
pip install -r requirements.txt               # or requirements-lock.txt for the exact versions of the committed runs
python scripts/download_ninfea.py             # NInFEA, every file checked against PhysioNet's SHA-256 list
python scripts/download_models.py             # ECGFounder and FetalCLIP weights, hash checked
# CinC 2013 set A and NIFEADB go to data/cinc2013/set-a and data/nifeadb (PhysioNet)
# DELTA (echo): download DELTA_Dataset.zip in a browser from doi:10.6084/m9.figshare.32358771, check its MD5
# (865d952adc0d8fe2be9df326e46cb763), extract the images and metadata.csv to data/delta/extracted/, then:
python scripts/make_echo_index_delta.py       # splits/echo_delta_index.csv from the authors' metadata.csv
python scripts/run_all.py --list              # every step and what it needs (data, weights, GPU)
python scripts/run_all.py                     # every run, in order, about three hours on a laptop with a small GPU
python scripts/export_artifacts.py            # artifacts/ for the one-patient tool, each refit checked against its run
python scripts/build_report.py                # reports/RESULTS.md, tables and figures
python -m sfr.audit                           # provenance and labelling checks
python -m pytest tests -q
```

The annotated Doppler strips (Mendeley Data, doi:10.17632/pg2djw3sjc.1) are expected in `fetaldata/`. That folder is ignored by git and must never be committed: it also holds unlabelled echo frames with patient identifiers in the pixels, which no analysis uses.

## Using the tool on one patient

`scripts/predict_patient.py` scores one patient's tests with the experts of the runs and applies the decision rule unchanged.
It is a research prototype: its performance on patients who had all three tests is not measured. Not for clinical use.

```bash
python scripts/export_artifacts.py            # once: writes artifacts/ and checks every refit against its run
python scripts/predict_patient.py --ecg data/ninfea/ecg/1 --doppler data/ninfea/pwd/1.bmp --gestation 27+0 --out tool_results/case1.json
python scripts/predict_patient.py --echo path/to/case_folder --cpu
```

- **Inputs** (at least one; only those experts run): `--ecg`, a WFDB record in the NInFEA layout (or `--ecg-layout nifeadb`);
  `--doppler`, a strip image at 284 columns per second, with `--gestation` as weeks+days; `--echo`, a folder with one image of each
  plane (Abdomen, 4CH, LVOT, RVOT, 3VT). An ECG and a Doppler recorded together are checked against each other (trust check)
  unless `--not-simultaneous` is given.
- **Calibration of the rhythm expert** (`--calibration`): `published` (14 normal NIFEADB recordings), `ninfea` (57 healthy NInFEA
  recordings), or a CSV with a column `score` holding the tool's rhythm scores of healthy recordings from your own device and
  setting. The tool warns whenever the calibration comes from another setting (calibrated on NIFEADB, the expert referred 51.7% of
  healthy NInFEA fetuses), and refuses a set too small for the alarm level: below 9 scores, no p-value can reach 0.10.
- **Output** (printed, and as JSON with `--out`): for each test its p-value, alarm and calibration (`published`, or `local (n=...)`);
  the trust check; the action under its unchanged name (`REFER`, `REPEAT`, `GET_TEST`, `ROUTINE`); `covered` and `not_covered`
  (structural heart defect, Doppler timing, rhythm disturbance); a `display_label` such as "No alarm on the tests done (structural
  heart defect (fetal echo)). NOT checked: Doppler timing (pulsed-wave Doppler: not done); rhythm disturbance (fetal ECG: not
  done). This is not a statement that the heart is normal."; the tool version, code and artifact hashes, the alarm levels, the
  note that the function and rhythm priors are assumptions, and the sentence "Research prototype. Not validated on patients with
  all three tests. Not for clinical use."
- A test that cannot be read gives `REPEAT` with the reason, never a guess. Echo images pass through `sfr.echo.deidentify`
  before anything is computed and are never saved. Results may be written only outside the repository or into `tool_results/`,
  which git ignores; the tool refuses any other place inside the repository.
- The structure expert needs the FetalCLIP weights (`python scripts/download_models.py`, CC BY-NC 4.0); on a laptop CPU one
  case takes under a minute.
- On real data the tool reproduces the runs: NInFEA recording 1 (trust check of wp3, Doppler measures of wp2b, rhythm score and
  decision of wp6) and one DELTA case (embedding of wp5), in `tests/test_predict.py`. `docs/inference_notes.md` lists what each
  expert needs and where it came from.

## Data and licences

See `docs/data_cards/`. Heartbeat (academic research only) and CARDIUM (CC BY-NC 4.0; academic, non-commercial research) are obtained through their authors' request forms and may not be passed on: their images, index files and per-patient outputs stay out of this repository (`tests/test_privacy.py`). Their terms ask every user to cite the dataset papers: Rodríguez et al., *Heartbeat*, Front Cardiovasc Med 2026, doi:10.3389/fcvm.2026.1726484; Vega et al., *CARDIUM: Congenital Anomaly Recognition with Diagnostic Images and Unified Medical records*, CVAMD workshop at ICCV 2025, arXiv:2510.15208. Run `wp15_reference_models` also needs the authors' Heartbeat code (github.com/BCV-Uniandes/Heartbeat, cloned to the git-ignored `data/third_party/Heartbeat`; it has no licence, so it is never copied here). CinC 2013, NInFEA and NIFEADB: Open Data Commons Attribution License v1.0. Annotated Doppler strips: CC BY 4.0. DELTA echo images: CC BY 4.0 (Zhang, Yao, Qian, Ye, figshare, doi:10.6084/m9.figshare.32358771). ECGFounder weights: MIT (model card). FetalCLIP weights: CC BY-NC 4.0 (research use only). No data or weights are in this repository.

## Paper

The paper (final draft of 28 October 2026) was written from these runs. It is kept outside this public repository until it is submitted.

## Not done

- A structure expert that works across hospitals: the DELTA expert does not (report section 13). The DELTA sweep videos are not used.
- Other ECG foundation models (HuBERT-ECG, ECG-FM); only ECGFounder was run.
- A labelled semi-synthetic test of the Doppler measures, and a comparison with published Doppler reference ranges.
- End-to-end clinical validation: any test of the fused decision, or of the one-patient research tool, on patients who had all three tests. It needs paired hospital data (Phase II).
- A cited prevalence for the priors of function and rhythm findings; they are assumptions (`docs/analysis_notes.md` lists where a cited value would go).
- A licence for the code and a citation file (CITATION.cff): both are decisions for the team (licence, authorship and order).
- Review by a clinician; the questions are in `docs/clinical_context.md`.

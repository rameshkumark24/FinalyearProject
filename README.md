# Screening the fetal heart from unpaired tests (Structure–Function–Rhythm)

Final-year project, Department of Computer Science and Engineering (Artificial Intelligence and Machine Learning), KPR Institute of Engineering and Technology.
Team: Rameshkumar K, Niranjana J, Risvanth V. Guide: Dr. Karthick Panneerselvam.

Three tests show three different things about a fetal heart:

| Test | Question | Expert | Labelled public data used |
|---|---|---|---|
| Fetal echocardiography | Is the structure normal? | structure | 752 cases, 133 with a heart defect (DELTA: open, one centre, released Oct 2026) |
| Pulsed-wave Doppler | Is the timing of filling and ejection normal? | function | healthy fetuses only (NInFEA) |
| Non-invasive fetal ECG | Is the rhythm normal? | rhythm | 12 arrhythmia, 14 normal rhythm (NIFEADB) |

No public dataset holds all three tests for the same fetus. Each test therefore gets its own expert, built on its own public data, and the experts are joined by a decision rule whose false-referral ceiling holds without any paired patient. Where ECG and Doppler were recorded together (NInFEA), a trust check verifies beat by beat that both show the same heart.

## Main results

Held-out data unless stated. Every number comes from a run folder; the full tables are in `reports/RESULTS.md`.

| Part | Result | Gate |
|---|---|---|
| Fetal beat detection (CinC 2013, 50 test recordings) | median F1 0.995; the mother's rhythm returned in none | G1 met |
| Doppler cardiac cycles (24 test strips, 1,595 cardiologist marks) | mean F1 0.963 | G2 met |
| ECG–Doppler trust check (NInFEA, 26 test women) | AUROC 0.984 against the mother's beats, 0.919 against another woman's ECG; beat-to-beat limits ±13.3 bpm against a target of ±10 | G3 **not met** (agreement part) |
| Function expert (60 healthy recordings, 39 women) | 2 of 60 flagged at level 0.05; only the heart rate is clearly repeatable | no gate; sensitivity cannot be measured |
| Rhythm expert (NIFEADB, leave one subject out) | one interval index AUROC 0.938; frozen ECGFounder 0.952; the same network with random weights up to 0.881 | G4 met |
| Structure expert (DELTA, case-level folds) | AUROC 0.828 (0.878 on the authors' test split); inputs without the heart also reach 0.62 to 0.77, so this is an upper bound | G5 partly met |
| Decision rule | false referrals 10.0% on 60 real healthy ECG–Doppler pairs (ceiling 15%); 7.1% to 16.9% with all three experts' healthy alarm rates combined (ceiling 20%) | G6 met (property tests) |

"Not met" and "partly met" are results. They are reported, not hidden.

## Read this first

- **Results live in `reports/RESULTS.md`**, which `scripts/build_report.py` generates from the run folders. Nothing in it is typed by hand.
- **The heart-defect result rests on one open, single-centre dataset (DELTA), and its controls show that part of the label can be read from images without the heart** (`reports/RESULTS.md`, section 6). Read the structure expert's AUROC as an upper bound, not as screening performance.
- **There is no measured performance of the fused system on patients.** That needs paired data, which does not exist publicly.
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
scripts/        one script per run, run_all.py, build_report.py, downloaders, development helpers (dev_*.py)
splits/         development and test definitions, and the DELTA image index
runs/           one folder per run (large arrays are not committed)
reports/        RESULTS.md, tables/, figures/, manifest.json - all generated
docs/data_cards one card per dataset: licence, counts, known problems
tests/          property tests of the decision rule, tests of the evaluation code, tests on real recordings
```

## Reproduce

```bash
pip install -r requirements.txt
python scripts/download_ninfea.py             # NInFEA, every file checked against PhysioNet's SHA-256 list
python scripts/download_models.py             # ECGFounder and FetalCLIP weights, hash checked
# CinC 2013 set A and NIFEADB go to data/cinc2013/set-a and data/nifeadb (PhysioNet)
# DELTA (echo): download DELTA_Dataset.zip in a browser from doi:10.6084/m9.figshare.32358771, check its MD5
# (865d952adc0d8fe2be9df326e46cb763), extract the images and metadata.csv to data/delta/extracted/, then:
python scripts/make_echo_index_delta.py       # splits/echo_delta_index.csv from the authors' metadata.csv
python scripts/run_all.py                     # every run, in order, about two and a half hours on a laptop with a small GPU
python scripts/build_report.py                # reports/RESULTS.md, tables and figures
python -m sfr.audit                           # provenance and labelling checks
python -m pytest tests -q
```

The annotated Doppler strips (Mendeley Data, doi:10.17632/pg2djw3sjc.1) are expected in `fetaldata/`. That folder is ignored by git and must never be committed: it also holds unlabelled echo frames with patient identifiers in the pixels, which no analysis uses.

## Data and licences

See `docs/data_cards/`. CinC 2013, NInFEA and NIFEADB: Open Data Commons Attribution License v1.0. Annotated Doppler strips: CC BY 4.0. DELTA echo images: CC BY 4.0 (Zhang, Yao, Qian, Ye, figshare, doi:10.6084/m9.figshare.32358771). ECGFounder weights: MIT (model card). FetalCLIP weights: CC BY-NC 4.0 (research use only). No data or weights are in this repository.

## Paper

The paper (draft 2, 13 October 2026) was written from these runs. It is kept outside this public repository until it is submitted.

## Not done

- The structure expert on a second, independent echo dataset. Heartbeat and CARDIUM are released only through their authors' request forms and were not requested; the DELTA sweep videos are not used.
- Other ECG foundation models (HuBERT-ECG, ECG-FM); only ECGFounder was run.
- A labelled semi-synthetic test of the Doppler measures, and a comparison with published Doppler reference ranges.
- Any test of the fused decision on patients with all three tests, and a screening tool for a single patient; both need paired hospital data (Phase II).

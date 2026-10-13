# Structure–Function–Rhythm (SFR): prenatal heart screening from three tests without paired patients

Final-year project. Three tests look at three different things in a fetal heart:

| Test | Question | Expert | Real labelled data |
|---|---|---|---|
| Fetal echocardiography | Is the structure normal? | structure | 752 cases, 133 with a heart defect (DELTA: open, one centre, released Oct 2026) |
| Pulsed-wave Doppler | Is the timing of filling and ejection normal? | function | healthy fetuses only (NInFEA) |
| Non-invasive fetal ECG | Is the rhythm normal? | rhythm | 12 arrhythmia, 14 normal (NIFEADB) |

No public dataset holds all three tests for the same fetus. So the experts are built separately, and their results are combined by a fixed rule whose safety properties do not depend on how the tests behave together. Where two tests *were* recorded together (fetal ECG and Doppler in NInFEA), a trust check verifies beat by beat that both show the same heart.

## Read this first

- **The repository was rebuilt from 11 October 2026.** The first build (2–4 October) reported numbers that did not come from real computation. It is kept in `archive/first_build_2026-10-02/` with a note; none of its results is valid.
- **Results live in `reports/RESULTS.md`**, which is generated from the run folders. Nothing in it is typed by hand.
- **The heart-defect result rests on one open, single-centre dataset (DELTA), and its controls show that part of the label can be read without looking at the heart** (`reports/RESULTS.md`, section 6). Read the structure expert's AUROC as an upper bound, not as screening performance.
- **There is no measured performance of the fused system on patients.** That needs paired data, which does not exist publicly.

## Where things stand

| Work package | What was done on real data | Gate |
|---|---|---|
| WP1 Fetal beat detection | CinC 2013 set A, 25 recordings for tuning, 50 held out; four ablations | G1 met |
| WP2 Doppler cycles | 43 strips with cardiologist marks, 19 for tuning, 24 held out (16 women) | G2 met |
| WP3 ECG–Doppler trust check | NInFEA, 13 women for tuning, 26 held out; mismatches built from real signals | G3 **not met** on beat-to-beat agreement; detection of the mother's beats met |
| WP2b Function expert | Healthy reference from 39 women; false-alarm control; repeatability | no gate; sensitivity cannot be measured |
| WP4 Rhythm expert | NIFEADB, leave-one-subject-out; interval features, a trained CNN, frozen ECGFounder with its published weights; controls; leakage comparison | G4 met |
| WP5 Structure expert | DELTA, 752 cases: frozen FetalCLIP and a linear probe, case-level folds and the authors' own split; shortcut audit; random-weight control | G5 partly met; the audit finds label information outside the heart |
| WP6 Fusion and decision | Property tests; decision rule on the real paired healthy cohort; rule comparison on virtual patients built from real scores; with the structure scores, false referrals with three tests | G6: tests pass |

"Not met" and "cannot be evaluated" are results. They are reported, not hidden.

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
sfr/            the package
  ecg/          fetal ECG extraction (maternal cancellation, ICA, beat tracking, guards against the mother's rhythm)
  doppler/      envelope from the strip image, cardiac cycles, timing measures, healthy reference, orientation
  trust/        ECG–Doppler trust check
  rhythm/       interval features, leave-one-subject-out evaluation, ECGFounder loader
  echo/         de-identification, FetalCLIP loader (no labelled data yet)
  fusion/       opinions and their OR, comparison rules, conformal p-values, the decision
  data/         loaders for CinC 2013, NInFEA, NIFEADB and the annotated Doppler strips
  provenance.py, audit.py
scripts/        one script per run, run_all.py, build_report.py, downloaders
splits/         development and test definitions
runs/           one folder per run (large arrays are not committed)
reports/        RESULTS.md, tables/, figures/, manifest.json - all generated
docs/data_cards one card per dataset: licence, counts, known problems
tests/          property tests of the fusion step and tests on real recordings
archive/        the first build, kept as a record
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

The annotated Doppler strips (Mendeley Data, doi:10.17632/pg2djw3sjc.1) are expected in `fetaldata/`. That folder also holds unlabelled echo frames with patient identifiers in the pixels; it is ignored by git and must never be committed.

## Data and licences

See `docs/data_cards/`. CinC 2013, NInFEA and NIFEADB: Open Data Commons Attribution License v1.0. Annotated Doppler strips: CC BY 4.0. DELTA echo images: CC BY 4.0 (Zhang, Yao, Qian, Ye, figshare, doi:10.6084/m9.figshare.32358771). ECGFounder weights: MIT (model card). FetalCLIP weights: CC BY-NC 4.0 (research use only). No data or weights are in this repository.

## Paper

A full paper draft was written from these runs on 12 Oct 2026. It is kept outside this public repository until submission; every number in it is read from `reports/RESULTS.md` and the run folders here.

## What is not done

- Structure expert on a second, independent dataset. It is run on DELTA only (one centre, no external test set). Heartbeat and CARDIUM are released only through their authors' request forms (re-checked 12 Oct 2026) and were not requested; see `docs/data_cards/echo.md`.
- The sweep videos of DELTA are not used; only the still images are.
- HuBERT-ECG and ECG-FM (only ECGFounder was run).
- Arrhythmia timings driving healthy Doppler shapes (planned as a labelled semi-synthetic test).
- Comparison of the Doppler measures with published reference equations (our measures are read from envelope peaks and clicks and are not the clinical intervals).
- Any test of the fused decision on patients with all three tests.

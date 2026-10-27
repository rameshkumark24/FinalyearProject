# Preregistration: the structure expert on a second echo dataset

Fixed on 20 October 2026, before any external echo image has been requested, received or seen.
Changes after that date must be listed at the end with their date and reason.

## Question

Does the frozen structure expert (FetalCLIP features, linear probe fitted on DELTA Train,
calibrated on the normal cases of DELTA Val; `artifacts/structure`) separate fetuses with a
structural heart defect from normal fetuses in a dataset collected elsewhere, and does its alarm
level hold there?

## Candidate datasets

Heartbeat (Front Cardiovasc Med 2026) and CARDIUM (ICCV 2025 Workshops), both behind access
forms; any other labelled fetal echo dataset with at least the planes the artifact expects. The
team requests access; nothing is downloaded by a script from a request page.

## Fixed analysis (`scripts/run_wp11_external_echo.py`)

1. No refitting of anything: the probe, its scaler and PCA come from the artifact, checked by
   hash. The script refuses any refit option and holds no fitting code (`tests/test_external_echo.py`).
2. Every image is cleaned by `sfr.echo.deidentify` and embedded by the verified FetalCLIP
   weights; one mean embedding per case.
3. Primary outcome: case-level AUROC with a 95% bootstrap interval (2000 resamples, seed 0).
4. Secondary outcomes at level 0.05:
   - transported calibration (the artifact's DELTA calibration scores): sensitivity and false-alarm
     rate with Clopper-Pearson intervals;
   - local calibration: half of the external normal cases (seed 0) calibrate, the other half and
     every abnormal case are judged.
5. Shortcut checks, nothing fitted: AUROC of image width, height and mean brightness used as
   scores; results by image size.

## How the result will be read (fixed now)

- Supports "reads structure": AUROC interval above 0.5 and above every shortcut score's AUROC.
- Supports "shortcut": AUROC not distinguishable from 0.5, or a shortcut score at least as high.
- Calibration holds if the transported false-alarm rate's interval includes 0.05; otherwise only
  local calibration can be used, as for the rhythm expert.
- If no external dataset arrives before the paper is submitted, the structure expert is removed
  from the headline claims and reported as an upper-bound component result only.

## Sample size

Whatever the dataset holds; the interval width will be reported. No subset will be chosen after
looking at results.

## Changes after 20 October 2026

**27 October 2026 — data received; clarifications fixed before any model was run on it.**
Heartbeat and CARDIUM arrived through the authors' request forms (on the development machine from
27 and 24 October). Before this entry only counts, file names, view names, folds and the names of
the metadata columns were read (`docs/data_cards/heartbeat.md`, `docs/data_cards/cardium.md`); no
model has seen an image and no score exists.

1. **Datasets.** Primary: Heartbeat, 2nd trimester, all 784 patients (the authors'
   cross-validation and test sets pooled; the frozen expert was fitted on none of them). Secondary,
   each reported on its own: the Heartbeat 2nd-trimester test set alone (61 patients, 6 with CHD;
   comparable with the authors' published numbers); Heartbeat 3rd trimester (690 patients);
   CARDIUM, all images (1,103 patients); CARDIUM, grey images only (1,023 patients).
2. **Case.** One patient in one trimester for Heartbeat, one patient for CARDIUM; the case
   embedding is the mean over all its usable images, as fixed. Heartbeat has no abdominal view and
   CARDIUM's views are not named and include colour stills, so the frozen expert meets other planes
   than DELTA's five. That is part of what this test tests; no plane is selected after seeing results.
3. **Unusable images.** An image in which `sfr.echo.deidentify` finds no scan region is skipped and
   counted; a case without a usable image is left out and counted.
4. **Image size of a case** is the most common width x height among its images.
5. **Reading rule, made precise.** "Above every shortcut score" compares the expert's AUROC point
   estimate with max(AUROC, 1 - AUROC) of each single-number shortcut score, because a shortcut can
   separate the classes in either direction. The rule decides the headline on the primary dataset;
   the secondary datasets are read with the same rule and reported.
6. **Calibration** as fixed: transported = the artifact's 62 DELTA calibration scores; local = half of
   the dataset's normal cases (seed 0).
7. **Software changes that do not change the analysis:** images are cleaned in parallel processes;
   images are resized with the helper of the one-patient tool (`sfrtool.experts.model_resizer`), which
   gives the same pixels as before; per-case outputs go to the run's `private/` folder because the
   access terms forbid redistribution.
8. **Everything else is exploratory** and labelled so: the authors' released models, probes fitted on
   each new dataset, calibration transfer of the structure expert, the trimester shift and the
   comparison of the two scans of the same fetus. These run only after the runs of item 1.

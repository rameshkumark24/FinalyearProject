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

None yet.

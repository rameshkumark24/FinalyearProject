# Positioning: what is standard, what this project adds, and what it does not show

Written on 20 October 2026 in answer to a review. The numbers referred to here are in
`reports/RESULTS.md` (generated from the run folders); this note holds no numbers of its own.

## Standard methods used here (not contributions)

| Element | Status | Source (see `related_work.md`) |
|---|---|---|
| Refer when any test alarms (the "believe the positive" or OR rule) | standard way to combine tests | Marshall 1989; Pepe and Thompson 2000 |
| False referrals of healthy people at most the sum of the alarm levels, for any dependence | the union (Bonferroni) bound | Dunn 1961 |
| The range between the largest single rate and the sum | Fréchet bounds; textbook probability | - |
| Conformal p-values against healthy calibration cases; smallest p-value 1/(n+1) | standard | Vovk 2013; Angelopoulos and Bates 2023; Bates et al. 2023 |
| Beta law of the false-alarm rate for one fixed calibration set | standard | Vovk 2013; Angelopoulos and Bates 2023 |
| Calibrating inside groups (Mondrian / group-conditional) | standard | Vovk 2013; Romano et al. 2020 |
| Validity lost under a shift between calibration and use, bounded by a distance between the distributions | standard | Barber et al. 2023 |
| Holm step-down; Simes (valid under positive dependence) | standard | Holm 1979; Simes 1986; Sarkar 1998; Benjamini and Yekutieli 2001 |
| Choosing cut-offs to maximise expected true positives under a cap on false positives, estimated on one part of the data and tested on another | standard | Rubin, Dudoit and van der Laan 2006; Roeder and Wasserman 2009 |
| Net benefit and decision curves | standard | Vickers and Elkin 2006 |
| Shortcut learning; controls built from inputs that cannot hold the signal | known problem and known check | Geirhos et al. 2020; DeGrave et al. 2021; Zech et al. 2018 |
| Doppler as the reference for fetal ECG timing | established use | Sulas et al. 2021 (NInFEA); Nii et al. 2006; Stampalija et al. 2012 |

Earlier drafts and the README called the false-referral ceiling and "the rule is at least as
sensitive as each expert" guarantees of this project. They are properties of a standard rule.
The second one only restates each expert's own sensitivity (below).

## What combining tests does and does not do

- **Widens coverage.** Each test looks for a different kind of problem (structure, timing,
  rhythm). Adding a test adds a condition that is checked at all.
- **Adds false referrals.** Each added test adds its own false alarms; the ceiling is the sum.
- **Does not improve detection of any one condition.** A fetus with a structural defect is
  referred by the structure expert at its own level; other experts only add chance alarms. In
  this project's numbers, the "sensitivity of the rule" for a condition equals the sensitivity of
  its own expert.
- **Splitting a fixed ceiling differently** (run `wp10_allocation`) trades detection of one
  condition against another. With these sample sizes the gain over the design levels is small and
  uncertain, and the method is that of Rubin et al. (2006) applied to discrete conformal levels.

## What the project adds (evidence, not methods)

1. **When the standard guarantee holds in practice.** The rhythm expert's false-alarm rate held
   on healthy NInFEA recordings only when it was calibrated on healthy recordings from the same
   setting; carried over from another device it referred about half of the healthy fetuses
   (run `wp6_fusion`). Run `wp13_calibration_transfer` turns this into a learning curve: how many
   local healthy recordings bring false alarms back near the level, alone or pooled with the
   original calibration set (one device pair, healthy fetuses only). The structure expert repeats it on data with defects: carried
   from DELTA to Heartbeat and CARDIUM its calibration fails, and local calibration repairs the false alarms but not the
   detection (runs `wp11_external_echo_*`, `wp17_structure_calibration`).
2. **Small-sample consequences, stated.** With 14 calibration recordings the rhythm level 0.10
   acts as 1/15, so part of the stated ceiling can never be spent (run `wp8_tradeoff`).
3. **A trust check for simultaneous ECG and Doppler, with its baselines.** It is a consistency
   (substitution) check between two recordings of the same heart. Heart rate alone nearly
   matches it against the mother's beats, and a plain cross-correlation of the beat trains beats
   it on swapped pairs (run `wp3b_trust_baselines`). Gate G3 (agreement) is not met.
4. **Honest component results with controls.** Each expert is reported with its controls,
   which show where the data allow a shortcut (DELTA image size; the four NIFEADB recordings
   sampled at 500 Hz).

## What the project does not show

- Screening performance of the combined system on patients: no dataset has all three tests.
- That the structure expert reads heart structure: inputs without anatomy also separate the
  classes, and among the 642x506 images it caught none of the defects, with an AUROC not
  distinguishable from chance (runs `wp5b`, `wp5c`).
- That the rhythm expert generalises: 26 recordings from one database; four of the arrhythmia
  recordings were sampled at 500 Hz and the rest at 1 kHz, and with every recording brought to
  500 Hz the mother's chest lead still scores above chance, which points to other differences of
  those four recordings (runs `wp4e`, `wp9`). It is a feasibility result.
- Any sensitivity of the function expert: NInFEA holds healthy fetuses only.
- External validity: the structure expert failed its preregistered external test (AUROC 0.509 on Heartbeat); the rhythm and function experts were not tested on a second dataset.

## The claim proposed for a paper (to be agreed with the guide)

> An any-alarm conformal rule keeps its false-referral guarantee only when each test is
> calibrated on healthy cases from the setting of use; we show how far it breaks when
> calibration is carried over between devices and how many local healthy recordings repair it.

Its evidence is runs `wp6_fusion` and `wp13_calibration_transfer`, with the standard theory
simulated in `wp12_theory_simulation`. It is a case study on one device pair; it would be
strengthened by a second device pair or a second echo dataset (`run_wp11_external_echo.py`).

# Analysis notes: error analysis, subgroups, baselines, priors

Written on 16 October 2026 in answer to a review that asked for error analysis, subgroups and
baselines. Everything listed under "added" is a check made after seeing the results; it
re-summarises saved per-case outputs and fits nothing new. The numbers are in
`reports/RESULTS.md` (sections 3, 5 and 9 to 11), built from the runs named here.

## What already existed before the review

| Question | Where it is answered |
|---|---|
| Structure expert by plane | `runs/wp5_structure_delta`, `by_plane` (AUROC of each plane alone; RESULTS section 6) |
| Structure expert by defect type | `runs/wp5_structure_delta`, `alarm.by_subtype` (share caught with an interval; types with fewer than 20 cases marked) |
| Structure expert by image size | `runs/wp5_structure_delta`, `image_size_by_label` and `same_image_size` (probe inside the 900x752 images only) |
| Which structural cases were missed | `runs/wp5_structure_delta/per_case.csv` (alarm and label of every case) |
| Image statistics as a shortcut baseline | `no_anatomy_numbers` probe (six numbers that describe the image, not the heart) and the removed-parts probe, with paired differences in `runs/wp5b_structure_controls_delta` |
| Heart rate alone as a rhythm baseline | `runs/wp4b_rhythm_intervals`, single index `hr_bpm_median` |
| Averaging rules against the OR rule | `runs/wp6_fusion` and `runs/wp6b_fusion_structure`, rule comparison and hypothesis H7 |
| Rhythm results by sampling rate | `runs/wp4c_foundation`, `sampling_rate_check_added_after_first_run` (point values only) |

## Added on 16 October 2026

| Run | What it adds |
|---|---|
| `wp7_dependence_bounds` | False-referral range for any dependence between the experts, at the measured rates and with their uncertainty; the same range checked against the joint rate measured on 60 real ECG-Doppler pairs; a dependence dial (figure f8); the sensitivity floor of the rule for each finding; precision of a referral over a grid of prevalences; gate G7 |
| `wp8_tradeoff` | Sensitivity and healthy alarm rate of each expert at a grid of alarm levels, and the false-referral ceiling of every combination of levels (figure f9). Shows that the rhythm expert, with 14 healthy calibration recordings, cannot alarm below 1/15 and that its level 0.10 acts as 1/15 |
| `wp9_checks_after_results` | Rhythm AUROC on the 22 recordings sampled at 1 kHz, with intervals; paired AUROC differences of rhythm scores (ECGFounder against the random-weight network, the expert against the best single index and against heart rate alone); robust ECG-Doppler agreement (percentiles of the beat-to-beat difference); structure alarms by image size; Doppler alarms by gestational age and the correlation of the Doppler score with gestational age; the identifiers of missed structural defects |

The threats to validity are generated with their numbers in `reports/RESULTS.md`, section 11.

## What could not be done, and why

- **Results by device.** DELTA records neither the device nor the gestational age, and NInFEA was
  recorded with one device. Image size is the only acquisition variable available in DELTA, and
  it is used.
- **Structure sensitivity by plane.** The alarm uses the mean embedding of the five planes of a
  case; there is no p-value per plane. The AUROC of each plane alone exists (run wp5).
- **Baselines for the function expert's sensitivity.** There is no Doppler recording of a fetus
  with abnormal timing, so no sensitivity, of the expert or of a baseline, can be measured.
- **Two rhythm scores in the paired comparisons.** The saved per-record files hold scores rounded
  to 6 decimals. For the random-weight network on the plain fetal input, rounding ties scores of
  the two classes, so its AUROC cannot be recomputed exactly from the file; it is listed, not
  re-analysed.
- **Correction for multiple checks.** The checks added after seeing the results report plain
  95% intervals; none is corrected for the number of checks.

## Priors are assumptions

The decision rule (`sfr/fusion/decision.py`, `DecisionConfig.prior`) uses priors of 0.009 for a
structural defect, 0.002 for abnormal Doppler timing and 0.001 for a rhythm disturbance. The
last two are assumptions without a cited source. They decide whether a missing test is asked for
(limit 0.005) and enter the risk interval reported with a decision; they never change an alarm.
Precision of a referral is therefore shown over a grid of prevalences (run wp7), never at a
single assumed value.

TODO when the team has a cited prevalence for a finding:

1. `sfr/fusion/decision.py`, `DecisionConfig.prior`: replace the assumed value and cite the source
   in the comment. This changes the code hash, so re-run `run_wp6_fusion.py`,
   `run_wp6b_fusion_structure.py` and `run_wp7_dependence_bounds.py`.
2. `scripts/run_wp7_dependence_bounds.py`, `PREVALENCE_GRID` (marked TODO): add the cited value.
3. The paper, section on the decision rule, and the README.

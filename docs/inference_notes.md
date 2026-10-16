# Scoring one new patient: what existed and what was missing

Written on 16 October 2026, before the one-patient tool was built, in answer to a review. It
lists, for every expert, what is needed to score one new case and where each piece lived.

| Expert | Feature extractor | Fitted model | Healthy calibration scores | Gap |
|---|---|---|---|---|
| Rhythm (fetal ECG) | `sfr.ecg.extract.extract_fetal_ecg` with the frozen settings written in `scripts/run_wp3_trust_check.py` (NInFEA: whole record, thoracic leads as maternal reference) and `scripts/run_wp4a_nifeadb_beats.py` (NIFEADB: 60-second segments, chest lead); `sfr.rhythm.features` | Logistic regression on 9 interval features of the 26 NIFEADB recordings. Refitted inside `scripts/run_wp6_fusion.py` from the saved features in `runs/wp4b_rhythm_intervals/per_record_scores.csv`; never saved | Published: out-of-fold scores of the 14 normal NIFEADB recordings (same file). Local: the scores of the 57 readable NInFEA recordings (`runs/wp6_fusion/paired_cohort_ninfea.csv`) | Model not saved, but its features are, so it can be refitted without new extraction |
| Function (Doppler) | `sfr.data.pwd.load_strip`, `sfr.doppler` (envelope, orientation, cycles, template measures) | Reference lines of `sfr.doppler.reference.fit_reference`: cycle length on gestational age, three intervals on cycle length. Fitted leave-one-woman-out inside `scripts/run_wp2b_function_reference.py`; the fit on all 60 recordings was never saved | Leave-one-woman-out scores of the 60 healthy NInFEA recordings (`runs/wp2b_function_reference/per_recording.csv`, column `score`) | Reference not saved, but the measures and gestational ages are, so it can be refitted. A new case needs its gestational age |
| Structure (echo) | `sfr.echo.deidentify` (cleaning), FetalCLIP (`sfr.echo.fetalclip`, weights checked against the published SHA-256), the resizing taken from the model's own preprocessing (`scripts/run_wp5_structure.py`) | Probe (standardise, PCA to 32, logistic regression) fitted on the DELTA Train split in `official_split()` of `scripts/run_wp5_structure.py`; never saved | Probe scores of the 62 normal cases of the DELTA Val split, computed in `official_split()`; never saved | Probe not saved, but the case embeddings are (`runs/wp5_structure_delta/embeddings.npz`, not in git), so it can be refitted without a GPU. Inference needs the FetalCLIP weights (1.7 GB, CC BY-NC 4.0) |
| ECG-Doppler trust check | `sfr.trust.crossmodal` (fixed rule) | none | none | none |
| Decision | `sfr.fusion.decision.decide` | none | none | The output does not say which conditions were checked: "ROUTINE" after an echo and a Doppler without an ECG reads like a normal heart |

Input formats the experts were built on:

- Fetal ECG: a WFDB record with the NInFEA channel layout (24 abdominal channels `uni_abd1`
  to `uni_abd24`, 3 thoracic leads `bi_tho1` to `bi_tho3`) or the NIFEADB layout (channel `ECG`
  from the mother's chest, then `Abdomen_1` and onwards).
- Doppler: a strip image at 284 columns per second that starts and ends with the ECG record,
  as in NInFEA. The trust check assumes that the ECG and the strip share one clock.
- Echo: one image of each of the five planes of DELTA (Abdomen, 4CH, LVOT, RVOT, 3VT). The probe
  was fitted on the mean of the five plane embeddings of a case.

Decisions taken for the tool:

- The tool lives in `sfrtool/`, outside `sfr/`. `sfr/` stays byte for byte as it was, so the code
  hash recorded by every run (`fe14c714b7e8`) still identifies the code that made it. The two
  changes the review asked for inside `sfr/` are made elsewhere: the docstring of
  `sfr/fusion/decision.py` already states both guarantees, and the fields `covered`,
  `not_covered` and `display_label` are added around the unchanged `Decision` by
  `sfrtool.labels.describe`.
- Artifacts are written once by `scripts/export_artifacts.py` into `artifacts/<expert>/`: plain
  NumPy arrays (`.npy`, loaded without pickle) and a `card.json` with the SHA-256 of every file.
  Every refit is checked against the published run before it is written (see
  `runs/tool_artifacts/metrics.json`).
- Nothing new is learned: only the existing linear models are refitted on saved features, with
  the settings of the runs.

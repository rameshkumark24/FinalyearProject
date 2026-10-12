# Data card — NIFEADB

**What it is.** Non-invasive fetal ECG recordings with and without a fetal arrhythmia.
Source: PhysioNet, *Non-Invasive Fetal ECG Arrhythmia Database*, version 1.0.0 (physionet.org/content/nifeadb/1.0.0). Paper: Behar, Bonnemains, Shulgin, Oster, Ostras, Lakhno, *Noninvasive fetal electrocardiography for the detection of fetal arrhythmias*, Prenatal Diagnosis (2019).

**Licence.** Open Data Commons Attribution License v1.0 (read on the dataset page, 11 Oct 2026).

**Role in this project.** The only real labels for the rhythm expert.

| Item | Value | Where it was checked |
|---|---|---|
| Recordings | 26: 12 arrhythmia (`ARR_01`–`ARR_12`), 14 normal rhythm (`NR_01`–`NR_14`) | dataset page; files |
| Channels | one maternal chest lead, four or five abdominal channels | record headers |
| Sampling rate | 1000 Hz in 22 recordings, 500 Hz in 4 (`ARR_06`–`ARR_09`) | record headers |
| Duration | 7.3 to 32.1 minutes | record headers |
| Label | per recording, from the file name | dataset page |
| Beat annotations | none | files |
| Arrhythmia type, gestational age | not on the dataset page ("can be found in the publication"); not used here | dataset page |

**Split.** Leave-one-subject-out over the 26 recordings (`splits/nifeadb_loso.json`).

**Things to know**

- **Twelve against fourteen.** The standard error of an AUROC is about 0.08 to 0.10. Differences between models below about 0.2 cannot be established.
- **All four 500 Hz recordings are arrhythmia recordings.** Everything is resampled to 1 kHz and band-limited to 3–100 Hz before use, and a negative-control input (the maternal chest lead) is reported.
- **Recordings differ in length** (two arrhythmia recordings are 22 and 32 minutes). Features are medians over one-minute segments, which do not depend on length.
- **The extractor was tuned on regular rhythms.** Its quality score includes the regularity of the tracked beats and is lower in the arrhythmia group (median 0.40 against 0.64 in run `wp4a_nifeadb_beats`). The quality score is therefore not a neutral control; the morphology-only quality value is reported beside it.
- **Calibration is 50:50 by design.** Probabilities from a model fitted here are not screening probabilities.

# Data card — NInFEA

**What it is.** Simultaneous non-invasive fetal ECG and pulsed-wave Doppler recordings of healthy fetuses.
Source: PhysioNet, *NInFEA: Non-Invasive Multimodal Foetal ECG-Doppler Dataset for Antenatal Cardiology Research*, version 1.0.0 (physionet.org/content/ninfea/1.0.0). Paper: Sulas et al., *Scientific Data* 8, 30 (2021).

**Licence.** Open Data Commons Attribution License v1.0 (read on the dataset page, 11 Oct 2026).

**Role in this project.** The only real *paired* data: trust check (ECG against Doppler), function expert (healthy reference), and the real paired healthy cohort for the decision rule.

| Item | Value | Where it was checked |
|---|---|---|
| Recordings | 60, from 39 women | dataset page; `splits/ninfea_by_woman.json` |
| Gestation | weeks 21 to 27 | dataset page |
| Fetal status | "cardiologically-healthy foetuses" | dataset page |
| Electrophysiology | 24 unipolar abdominal channels, 3 bipolar maternal thoracic leads, maternal respiration; 2048 Hz | record headers |
| Doppler | one strip image per recording, five-chamber apical window, sweep 75 mm/s | dataset page |
| Image time scale | 284 columns per second (`fs_doppler=284` in the dataset's own viewer code) | `data/ninfea/code/Interfaccia_dataset.m` |
| Duration | 7.5 to 119.8 s per recording, 1,832.8 s in total | record headers (measured) |
| Image width against record length | equal within 0.066 s in all 60 recordings | measured |
| Devices | TMSi Porti7 (ECG), Philips iE33 (ultrasound), video capture of the screen at 60 Hz | dataset page |
| Files on disk | 120 ECG files, 60 images, 3 code files, licence; every file matches PhysioNet's SHA-256 list | `scripts/download_ninfea.py` |

**Grouping.** Several women have two or three recordings. All analyses that fit or set anything keep a woman's recordings together. The grouping and the gestational age per woman are transcribed from tables 2 and 3 of the paper.

**Splits.** `splits/ninfea_dev_test.json`: 13 development women (23 recordings), 26 test women (37 recordings), drawn at random with a fixed seed before any NInFEA ECG was analysed.

**Things found while working with it**

1. **The display orientation differs between strips and is not stored.** Ten strips are exact upside-down copies of strips in the annotated set (see the annotated-set card), which its authors flipped so that filling is above the zero line. So in those ten NInFEA recordings ejection is above the zero line. Orientation must be inferred (from the ECG, or from the Doppler alone).
2. **The time base of a strip can jump.** In some recordings the Doppler interval sequence shows a step of about 14 columns (about 50 ms, three video frames) while the fetal ECG intervals stay constant. The dataset page says the strip was assembled from a 60 Hz video by a custom tool and synchronised by post-processing; our reading is that frames were dropped or repeated. This is an inference from the data, not a statement of the authors.
3. **One mother's heart rate (140 bpm, record 22) is faster than her fetus's.** A fixed "maternal rate below 140" rule is wrong.
4. **No fetal beat annotations.** The Doppler is the only independent reference for the fetal ECG.
5. **No abnormal case.** Nothing about sensitivity can be learned from this dataset.

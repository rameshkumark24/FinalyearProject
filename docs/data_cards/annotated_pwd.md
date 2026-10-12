# Data card — annotated fetal Doppler strips (Cagliari)

**What it is.** 43 pulsed-wave Doppler strips with a cardiologist's marks of complete cardiac cycles.
Source: Mendeley Data, doi:10.17632/pg2djw3sjc.1. Paper: Sulas, Urru, Tumbarello, Raffo, Pani, *Impact of pulsed-wave-Doppler velocity-envelope tracing techniques on classification of complete fetal cardiac cycles*, PLoS ONE 16(4): e0248114 (2021).

**Licence.** CC BY 4.0; ethics approval and consent are stated (read on the Mendeley Data page during the audit of 11 Oct 2026). The paper is open access under the Creative Commons Attribution licence.

**Where it is.** `fetaldata/` on the development machine (ignored by git, never committed).

**Role in this project.** Ground truth for Doppler cycle detection.

| Item | Value | Where it was checked |
|---|---|---|
| Strips | 43 (`img1` … `img50`, with gaps) | files |
| Marks | 2,836 in `annotations.mat` (column, image number) | file |
| Women | 25 low-risk pregnant volunteers, weeks 21 to 27 | paper |
| Device | Philips iE33, sweep speed 75 mm/s | paper |
| What a mark is | the left edge of a 128-column window that holds one complete, measurable cardiac cycle | `Visualization_sel.m` shipped with the data; paper |
| Who marked | "an expert pediatric cardiologist labeled all the complete and measurable fetal cardiac cycles" | paper |
| Tolerance built into the labels | the tool also labelled windows shifted by up to 15 columns before and after each mark (87,736 labelled windows for 2,836 marks ≈ 31 per mark) | paper; our arithmetic |
| Orientation | "we flipped the negatively balanced images upside-down": filling is above the zero line in every strip | paper; confirmed on the data |

**Overlap with NInFEA** (`splits/pwd_overlap.json`, found by file hash, pixel comparison and a search for cropped copies):

| Relation | Strips |
|---|---|
| Byte-identical to a NInFEA strip | 13 |
| Exact upside-down copy of a NInFEA strip | 10 |
| Same strip with 15 extra rows | 1 |
| Not found in NInFEA | 19 |

So this is **not** a second, independent group of women: 24 of the 43 strips are NInFEA recordings.

**Splits.** `splits/pwd_cycle_marks.json`: development = the 19 strips not in NInFEA; test = the 24 strips that are NInFEA recordings (16 women). The woman is not known for the 19 development strips.

**Things to know**

- 43% of the marks sit exactly 128 columns after the previous mark (tiled windows). Such marks drift against the true beat by a few columns per beat, so a mark is not an exact event time. Matching uses 15 columns (53 ms).
- Only complete, measurable cycles are marked. A detected cycle without a mark is not necessarily an error; false detections are counted only between two marks that are one cycle apart.

# Data card — fetal echocardiography (structure expert)

## Status: one labelled dataset is on disk (DELTA, since 12 Oct 2026). The structure expert is run on it.

## DELTA — the dataset used

| Item | Value | Where it was checked |
|---|---|---|
| Record | *DELTA: A Dual-Modality Fetal Echocardiography Dataset for Long-Tail Congenital Heart Disease Diagnosis*, figshare, doi:10.6084/m9.figshare.32358771 (version 2) | figshare API record; DataCite record |
| Authors | Junqi Zhang, Shifa Yao, Dahong Qian, Baoying Ye | figshare API record |
| Posted | 8 Oct 2026 | figshare API record |
| Licence | CC BY 4.0 | figshare API record; DataCite record; the archive's README |
| File | `DELTA_Dataset.zip`, 6,940,210,631 bytes, MD5 `865d952adc0d8fe2be9df326e46cb763` | published value; **computed on the downloaded file on 12 Oct 2026: identical** |
| Content of the archive | 4,514 files: 3,760 JPEG images (479 MB), 752 videos (609 mp4, 143 avi; 6.5 GB), `README.txt`, `metadata.csv` | zip directory |
| Cases | 752: 619 normal, 133 abnormal | `metadata.csv` |
| Defect subtypes (cases) | TOF 33, aortic arch anomalies 17, single great artery 12, DORV 11, right aortic arch 11, persistent left superior vena cava 10, AVSD 9, pulmonary stenosis 8, hypoplastic right heart 7, hypoplastic left heart 6, single ventricle 5, TGA 4 | `metadata.csv`; README |
| Images per case | five standard planes: abdomen, four-chamber, LVOT, RVOT, three-vessel-trachea; plus one sweep video | README; files |
| Authors' partition | patient-level: Train 521 cases (88 abnormal), Val 77 (15), Test 154 (30) | `metadata.csv` |
| Image sizes (cases) | 900×752: 531; 642×506: 123; 806×648: 74; 744×658: 16; 1114×766: 7; 448×404: 1. The five images of a case always share one size | `metadata.csv` |
| De-identification | the README states that all protected health information was removed by a two-stage pipeline | README |
| Ethics approval, consent | **no statement** in the figshare record or in the README | read on 12 Oct 2026 |
| Centre, scanner, gestational age | not given, beyond "a high-volume tertiary center" in the record description | record; README |
| Paper | none yet; the README says a Data Descriptor is to appear in *Scientific Data* | README |

**What this project uses.** The 3,760 still images and the labels. The videos are not used and stay in the archive. Images and `metadata.csv` are extracted to `data/delta/extracted/` (ignored by git). `scripts/make_echo_index_delta.py` turns the authors' `metadata.csv` into `splits/echo_delta_index.csv` (one row per image: case, label, subtype, plane, path, split), without changing or filtering anything.

**Checks made before use (12 Oct 2026)**

- Every image listed in `metadata.csv` exists and has the listed size in bytes.
- No JPEG holds an EXIF, IPTC or comment segment.
- A contact sheet of eight images from different sizes and classes showed scan sectors on a black background with no patient text; one image carries a depth marker. This is a spot check, not a review of all images.
- Our own cleaning step keeps a median of about 70% of each frame (measured on twelve images before the run).

**Things to know**

1. **Image size is tied to the label.** Among the 900×752 cases 21% are abnormal; among the other common sizes 6 to 9%. Size says something about the scanner or its settings, not about the heart, so a model could use it. The run therefore reports a probe on six numbers that carry no anatomy, and repeats the main probes inside the 900×752 cases.
2. **Twelve subtypes, few cases each** (4 to 33). Results by subtype are descriptive.
3. **Single centre, no external test set.** The authors' Test split comes from the same centre.
4. **The dataset is four days old at the time of use** and has no peer-reviewed description yet. How the labels were confirmed (for example by postnatal findings) is not stated.
5. **No ethics statement.** A paper that uses these data must say so and should check the record again before submission.
6. **Our cleaning step removes some tissue on DELTA.** DELTA frames carry no text, so what `deidentify` blanks is mostly bright tissue at the border of the scan sector and the sector outline (seen on a sample of images after the run). The overlay-only control images therefore hold fragments of anatomy and the sector shape; they are not anatomy-free. The control is weaker than intended, and this is stated wherever it is reported.

**Results:** `reports/RESULTS.md`, section 6 (runs `wp5_structure_delta`, `wp5b_structure_controls_delta`) and section 8 (run `wp6b_fusion_structure`).

## Heartbeat and CARDIUM — not available to us

| Dataset | What it holds (from the project's literature notes) | Access |
|---|---|---|
| Heartbeat (BCV-Uniandes) | Fetal echo views with CHD and normal labels from three Colombian hospitals | request form; **not sent** |
| CARDIUM (BCV-Uniandes) | Echo images with clinical data | request form; **not sent** |

Checked on 12 Oct 2026: both repositories (github.com/BCV-Uniandes/Heartbeat, github.com/BCV-Uniandes/Cardium) release images, tables and weights only through their request forms, "exclusively for academic and research purposes". No copy released by the authors was found on Hugging Face, Zenodo, figshare or elsewhere. Copies from any other party would break the access terms and are not used. Only the project owner, with the guide's name, can send the forms.

## How DELTA was obtained

figshare refused scripted downloads (HTTP 403) on 12 Oct 2026, and later that day answered 403 to every page from this network, the project owner's browser included; the likely cause is the scripted attempts made earlier that day from the same address. No further scripted request is made to figshare. The project owner downloaded the file in a browser on the evening of 12 Oct and placed it at `data/delta/DELTA_Dataset.zip`.

## The FCU folder — not used for results

**FCU folder** (`fetaldata/fcu(echocardiogram)-data`, hand-added; source given as Zenodo 10.5281/zenodo.18757737).

| Item | Value |
|---|---|
| Files | 150 JPEG frames in 30 folders (5 per folder) |
| Labels | none |
| Ethics or consent statement | none found |
| Privacy | **patient identifiers are burned into the pixels** (name, number, hospital, date) |

Rules for this folder:

1. It is ignored by git and must never be committed or uploaded.
2. It is not used for any result.
3. It is read in memory only, to test that the de-identification tool and the FetalCLIP loader run (run `wp5_echo_readiness`). Nothing derived from it is written to disk except counts.
4. Identifiers are never copied into notes, logs or reports.

**Zenodo 5225627**, listed in an early project note as an echo image dataset, contains only a paper PDF.

## Code

- `sfr/echo/deidentify.py`: keeps the scan, blanks text and overlays; `overlay_only()` produces the complement for the shortcut audit.
- `sfr/echo/fetalclip.py`: loads the published FetalCLIP weights after a SHA-256 check (CC BY-NC 4.0, research use).
- `scripts/run_wp5_structure.py`: the whole evaluation for a labelled dataset given as an index file. It runs the shortcut audit (overlay-only images, and six numbers that carry no anatomy), the FetalCLIP linear probe with case-level folds, the same network with random weights as control, a split-conformal alarm at the level of the decision rule, the dataset's own partition, the same-image-size check, and results by plane and by defect subtype. Probe, folds and alarm level are fixed in the file. Its statistics are tested on synthetic feature vectors in `tests/test_structure_probe.py`.
- `scripts/run_wp6b_fusion_structure.py`: the decision-rule analysis with the structure expert's scores.

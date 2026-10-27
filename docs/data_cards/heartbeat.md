# Data card — Heartbeat (fetal echocardiography, 2nd and 3rd trimester)

**What it is.** Still fetal echo images of four standard cardiac views with a binary label (CHD or
no CHD) per patient, from the BCV group of Universidad de los Andes (Colombia), released with the
paper Rodríguez S, Pérez A, Echeverry LM, et al. *Heartbeat: a multimodal dataset of fetal
echocardiography and clinical metadata for early detection of congenital heart disease.* Front
Cardiovasc Med. 2026;13:1726484. doi:10.3389/fcvm.2026.1726484. Code and access:
github.com/BCV-Uniandes/Heartbeat.

**Access and terms.** Released only through the authors' request form, which checks the
requester's institutional affiliation; "distributed exclusively for academic and research
purposes" (repository README, read on 20 and 27 Oct 2026). The repository README states no
licence for the code. Requested by the team leader after 20 Oct 2026; on the development machine
from 27 Oct 2026, in `data/heartbeat/` (git-ignored).

**What may be published.** Aggregate results only. Images, the index files that list patient
codes with labels (`data/indexes/`), and every per-patient output (`runs/*/private/`) stay out of
the public repository (`tests/test_privacy.py`).

| Item | Value | Where it was checked |
|---|---|---|
| Images | 6,215 PNG | files |
| Patients | 1,178 unique codes (P0001–P1178, shuffled by the authors) | files; dataset README |
| 2nd trimester (2T) | 784 patients (50 with CHD): cross-validation set 723 (44), test set 61 (6) | metadata files |
| 3rd trimester (3T) | 690 patients (50 with CHD); no test set was published | metadata files; dataset README |
| Patients in both trimesters | 296 (20 with CHD by the 3T label); 2 have different labels in the two trimesters | metadata files |
| Views | 3VT, 4C, RVOT, LVOT, named in each file name; not every patient has every view; **no abdominal view** | files; dataset README |
| Gestational age (weeks) | 2T 13 to 33 (median 22); 3T 24 to 40 (median 33) | metadata_raw.csv |
| Clinical fields | maternal age, gestational age, three vessels, growth percentile (raw and normalised files) | dataset README |
| Partition | 4 folds (train/val) inside each cross-validation set; one 2T test set | fold1..4.csv |
| Released models | Heart-ViT, 4 folds for 2T and 4 for 3T; baselines (MobileNet, ResNet18, ResNet50, VGG16, ViT), 4 folds, 2T only | checkpoint files |
| Reported results (Heart-ViT, 2T test, mean ± SD over the 4 fold models) | sensitivity 79.17 ± 8.33, specificity 93.18 ± 2.29, AUROC 88.56 ± 3.80 | repository README |

**Known problems.**
- The 2T test set holds only 6 patients with CHD: one patient moves sensitivity by about 17 points.
  Results on it are reported, but never alone.
- A case in this project is one patient in one trimester. The 296 patients scanned twice are
  never on both sides of a split, and analyses that pool the trimesters resample by patient.
- **Heartbeat and CARDIUM share many fetuses.** Whole frames never match (the two store frames of
  different sizes), but the cleaned scan regions do: 2812 matched scans; 493
  Heartbeat patients share two or more scans with a CARDIUM patient (505 of 507 pairs
  with the same label), and 180 more pairs share one scan (runs `wp14_echo_overlap`,
  `wp14b_echo_overlap_scan`). The two are therefore not independent external datasets.
- 16 identical images appear under 6 different patients
  (0 with different labels).
- The views differ from DELTA's (no abdominal view), so the frozen DELTA structure expert meets a
  different mix of planes here (see `docs/preregistration_external_echo.md`).

**Index files** (`python scripts/make_echo_index_heartbeat.py`, written to `data/indexes/`):
all cases; 2T (cross-validation and test); 2T test only; 3T.

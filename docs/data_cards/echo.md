# Data card — fetal echocardiography (structure expert)

## Status: no labelled data on disk. No result on heart defects exists in this project.

### What is needed

| Dataset | What it holds (from the project's literature notes) | Access |
|---|---|---|
| Heartbeat (BCV-Uniandes) | Fetal echo views with CHD and normal labels from three Colombian hospitals | request form; **not sent** |
| CARDIUM (BCV-Uniandes) | Echo images with clinical data | request form; **not sent** |

Only the project owner, with the guide's name, can send the forms. Until access is granted the structure expert is "test not done" in every analysis.

**Checked again on 12 Oct 2026.** Both repositories (github.com/BCV-Uniandes/Heartbeat, github.com/BCV-Uniandes/Cardium) still release images, tables and weights only through their request forms, "exclusively for academic and research purposes". No copy released by the authors was found on Hugging Face, Zenodo, figshare or elsewhere. Copies from any other party would break the access terms and are not used.

### An open alternative found on 12 Oct 2026: DELTA

| Item | Value | Where it was checked |
|---|---|---|
| Record | *DELTA: A Dual-Modality Fetal Echocardiography Dataset for Long-Tail Congenital Heart Disease Diagnosis*, figshare, doi:10.6084/m9.figshare.32358771 (version 2) | figshare API record; DataCite record |
| Authors | Junqi Zhang, Shifa Yao, Dahong Qian, Baoying Ye | figshare API record |
| Posted | 8 Oct 2026 | figshare API record |
| Licence | CC BY 4.0 | figshare API record; DataCite record |
| Content (as described by the authors) | 752 cases: 619 normal, 133 abnormal, 12 heart-defect subtypes; five standard planes per case (3,760 images) and one scan video per case (752); de-identified | record description |
| File | `DELTA_Dataset.zip`, 6,940,210,631 bytes, MD5 `865d952adc0d8fe2be9df326e46cb763` | figshare API record |
| Linked paper | none on the record | figshare API record |
| Ethics statement | none in the record description; to be looked for inside the archive | record description |

**Status: not downloaded.** figshare refused scripted downloads (HTTP 403) on 12 Oct 2026. The file has to be downloaded in a browser and placed at `data/delta/DELTA_Dataset.zip`. Nothing in this project has read it, and no claim about its content beyond the record description is made here. Before any use: verify the MD5, read the archive's own documentation, check the images for remaining identifiers, and run the shortcut audit first.

### What is on disk

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

### Ready for the day access is granted

- `sfr/echo/deidentify.py`: keeps the scan, blanks text and overlays; `overlay_only()` produces the complement for the shortcut audit.
- `sfr/echo/fetalclip.py`: loads the published FetalCLIP weights after a SHA-256 check (CC BY-NC 4.0, research use).
- Shortcut audit to run first: train the same classifier on (a) de-identified scans and (b) overlay-only images. If (b) predicts the label, the label leaks through calipers and text.

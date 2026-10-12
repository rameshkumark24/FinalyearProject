# Data card — fetal echocardiography (structure expert)

## Status: no labelled data on disk. No result on heart defects exists in this project.

### What is needed

| Dataset | What it holds (from the project's literature notes) | Access |
|---|---|---|
| Heartbeat (BCV-Uniandes) | Fetal echo views with CHD and normal labels from three Colombian hospitals | request form; **not sent** |
| CARDIUM (BCV-Uniandes) | Echo images with clinical data | request form; **not sent** |

Only the project owner, with the guide's name, can send the forms. Until access is granted the structure expert is "test not done" in every analysis.

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

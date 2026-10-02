# Learning Path — What We Need to Learn

<aside>
🎓

**Purpose.** The shortest path to building the three-input framework and defending it under questioning (Student Guideline 6). Each module says why it is needed.

</aside>

<aside>
✂️

**Skip these:** designing new CNN, U-Net or 1D-CNN architectures (the panel called them crowded) · new fetal-ECG extraction networks (22 studies already) · turning signals into images · training large models from scratch · app or web UI.

</aside>

---

# Tier 1 — before anything runs (all three)

| Module | Why | Est. |
| --- | --- | --- |
| Python, PyTorch, Colab GPU; Git and GitHub | All three experts; a shared audit trail | 2 days |
| Fetal heart basics: the four echo views (4C, LVOT, RVOT, 3VT); the Doppler waveform (E and A filling waves, ventricular ejection); fetal ECG (P-QRS-T, normal fetal rate) | Must be explained at every review | 2 days |
| AHA 2014 statement: anatomy, function and rhythm | The clinical reason for three inputs | ½ day |

# Tier 2 — signals (Niranjana, Risvanth)

| Module | Why | Est. |
| --- | --- | --- |
| Reading PhysioNet data with `wfdb`; filtering; resampling | NInFEA, NIFEADB, CinC 2013 | 1 day |
| Maternal ECG removal: template subtraction and ICA; fetal QRS detection (NeuroKit2) | Runs R1–R2 | 3 days |
| Doppler envelope from video frames: Otsu threshold, area opening, edge detection; cycle detection | Runs F1–F2 | 2 days |
| Heart-rate variability features; Bland–Altman agreement | Runs R3, C1 | 1 day |

# Tier 3 — foundation models (all three)

| Module | Why | Est. |
| --- | --- | --- |
| Loading FetalCLIP (`open_clip`) and ECG foundation models (Hugging Face); extracting embeddings | Runs S3, R5 | 2 days |
| Linear probing vs LoRA fine-tuning | Runs S4, R6 | 2 days |
| Model licences — MIT vs non-commercial | FetalCLIP and HuBERT-ECG are non-commercial | ½ day |

# Tier 4 — evaluation done right (all three)

- Subject-level splits, leave-one-subject-out, and why segment-level splits leak — runs R3, R4 — 1 day
- Sensitivity, specificity, AUROC, AUPRC; bootstrap confidence intervals — 1 day
- Imbalance: weighted loss and sampling (Heartbeat has 6–7% CHD) — 1 day

# Tier 5 — uncertainty and fusion (the novelty)

| Module | Why | Est. |
| --- | --- | --- |
| Calibration: reliability diagrams, ECE, Brier score, temperature scaling (Guo 2017) | Every expert, before fusion | 2 days |
| Subjective logic: opinions (belief, disbelief, uncertainty), the OR operator (Jøsang & McAnally 2004); evidential deep learning (Sensoy 2018) | The fusion layer — main contribution | 4 days |
| Dempster's rule and trusted multi-view classification (Han 2022) — to explain why we do not use consensus | Run X1 | 1 day |
| Conformal risk control (Angelopoulos 2022) — optional | Run X4 | 2 days |

# Tier 6 — publication (all three)

- LaTeX / Overleaf, Zotero, plagiarism and citation rules (Student Guideline 8)
- The **TRIPOD+AI** reporting checklist — following it is an easy strength

---

# Proposed split — confirm as a team

| Member | Owns | Runs |
| --- | --- | --- |
| **Rameshkumar K** | Structure expert (echo, FetalCLIP) and the fusion layer | S1–S9, X1–X4 |
| **Niranjana J** | Rhythm expert (fetal ECG) and data management | D1–D2, R1–R7 |
| **Risvanth V** | Function expert (Doppler), trust check, calibration | F1–F4, C1–C3 |

Every member must still be able to explain the whole pipeline.
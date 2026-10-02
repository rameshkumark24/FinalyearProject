# Novelty Check — Connected Papers & Literature Map

<aside>
🧭

**Purpose.** Shows the panel where this project sits in the literature and what is missing — the novelty check the guide asked for at Review II, using Connected Papers. Updated on 19 Sep 2026 for the three inputs: fetal ECG, Doppler ultrasound and echocardiography. Figures are from papers checked at source; OpenAlex and Europe PMC counts are indicative only. A gap means **not found in this check**.

</aside>

---

# 1. Review II feedback → how the project answers it

| Panel's point | How the project answers it |
| --- | --- |
| Use Connected Papers to check similar papers | Ten seed papers across the three inputs, a citation map already run on OpenAlex (below), and a comparison table |
| 1D-CNN-related papers — there are tons | Agreed — fetal-arrhythmia 1D-CNNs on NIFEADB alone number at least 15 papers. A 1D-CNN appears only as a **baseline**. The rhythm expert tests **pretrained ECG foundation models**, which no fetal-ECG study in our check has done |
| Medical imaging always uses U-Net | No segmentation. The structure expert adapts a **fetal-ultrasound foundation model** (FetalCLIP) as a classifier; the contribution is the fusion |
| ECG is already digital — why image-type questions? | Every input stays in its native form: fetal ECG and the Doppler envelope as **1-D signals**, echo as images. Nothing is converted into a picture |

---

# 2. Where the literature is crowded, and where it is open

| Angle | Key existing work | Status |
| --- | --- | --- |
| CHD classification from fetal echo | Arnaout 2021; Tang 2023; Lei 2025; 15-study meta-analysis 2025 — 51 OpenAlex matches | **Crowded** |
| Echo segmentation; image + clinical fusion | Nurmaini 2021; Lu 2024; CARDIUM 2025; Heart-ViT 2026 | **Crowded / done** |
| Fetal ECG extraction; fetal arrhythmia from fetal ECG | 2026 systematic review (22 studies); at least 15 NIFEADB arrhythmia papers | **Crowded** |
| Arrhythmia and timing from Doppler | Yang 2024 (Medical Image Analysis); HR-IQS 2024 (Heart Rhythm) — private data | **Done** |
| Fetal ECG → Doppler translation | Verma 2025; Su 2026 — both on NInFEA | **Taken** |
| Uncertainty-aware fusion within echo | Zhou 2026 (arXiv, private data) | Emerging |
| Fetal-ultrasound foundation model for CHD | FetalCLIP 2026 — internal data only | New |
| **Fetal ECG + Doppler + echo combined** | None found | **Open** |
| **Fusion without paired patients / missing inputs (fetal)** | None found | **Open** |
| **ECG foundation models on fetal ECG** | None found | **Open** |
| **Fetal ECG–Doppler agreement as a trust check** | None found among the 30 papers citing NInFEA | **Open** |

---

# 3. Citation map already run (OpenAlex, 19 Sep 2026)

The same idea as Connected Papers — who cites the key datasets — run on OpenAlex:

- **NInFEA** (cited by 30): fetal QRS detection, extraction, fetal-presentation detection, and two ECG → Doppler translation papers. **Nobody uses it for a trust check or for fusion with echo.**
- **NIFEADB — Behar 2019** (cited by 99): mostly arrhythmia classifiers and extraction methods; no foundation models.
- **de Vries 2023** (cited by 27): reviews only; no one extends fetal-ECG CHD detection with public data.
- **Heartbeat 2026**: not yet cited.

---

# 4. Connected Papers — protocol

Build one graph per seed, then record every relevant paper in the table in section 5.

**Seed papers**

1. Arnaout et al. *An ensemble of neural networks provides expert-level prenatal detection of complex congenital heart disease.* Nature Medicine, 2021. doi:10.1038/s41591-021-01342-5
2. Rodríguez et al. *Heartbeat.* Frontiers in Cardiovascular Medicine, 2026. doi:10.3389/fcvm.2026.1726484
3. Vega et al. *CARDIUM.* arXiv:2510.15208, 2025
4. de Vries et al. *Fetal electrocardiography and artificial intelligence for prenatal detection of congenital heart disease.* Acta Obstet Gynecol Scand, 2023. doi:10.1111/aogs.14623
5. Sulas et al. *A non-invasive multimodal foetal ECG–Doppler dataset for antenatal cardiology research (NInFEA).* Scientific Data, 2021. doi:10.1038/s41597-021-00811-3
6. Behar et al. *Noninvasive fetal electrocardiography for the detection of fetal arrhythmias.* Prenatal Diagnosis, 2019. doi:10.1002/pd.5412
7. Yang X. et al. *An intelligent quantification system for fetal heart rhythm assessment.* Heart Rhythm, 2024. doi:10.1016/j.hrthm.2024.01.024
8. Maani et al. *FetalCLIP.* npj Digital Medicine, 2026. doi:10.1038/s41746-026-02907-9
9. Su et al. *Cross-modal generative framework … fetal–maternal ECG to fetal Doppler.* arXiv:2607.08073, 2026 — may be too new for a graph
10. Zhou et al. *Trusted multi-view deep learning classification of fetal congenital heart disease.* arXiv:2606.15265, 2026 — may be too new for a graph

**Steps**

- [ ]  Build the graph for each seed; paste a screenshot here
- [ ]  Check **Prior works** and **Derivative works**
- [ ]  Add every relevant paper to the table below, one row each
- [ ]  Our novelty is the columns that stay empty: all three inputs, works without paired data, trust check
- [ ]  Any paper that fills those columns is prior work to cite and differentiate from — add it to the Research Package

---

# 5. Comparison table — fill from Connected Papers

| Paper | Year | Inputs used | Task | Public data? | Subject-level? | Calibration / uncertainty? | Works with a missing input? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Arnaout et al. | 2021 | Echo | CHD detection, 5 views | No | Yes | Not in abstract | No |
| Heartbeat / Heart-ViT | 2026 | Echo + metadata | CHD detection | On request | Yes | No | No |
| de Vries et al. | 2023 | Fetal ECG | CHD detection | No | ? | ? | No |
| HR-IQS (Yang X. et al.) | 2024 | Doppler | Cardiac time intervals | No | ? | ? | No |
| Su et al. | 2026 | Fetal ECG + Doppler | ECG → Doppler translation | Yes (NInFEA) | ? | ? | No |
| Zhou et al. | 2026 | Echo (5 views) | CHD, uncertainty-aware fusion | No | ? | Uncertainty | Low-quality views |
| **This project** | 2026 | **Fetal ECG + Doppler + echo** | Referral with uncertainty | Yes | Yes | Yes | **Yes** |
| *add from Connected Papers* |  |  |  |  |  |  |  |

---

# Sources

Full citations are on the Methodology Research page and in the Research Package. OpenAlex, Europe PMC and arXiv searches were run on 19 Sep 2026.
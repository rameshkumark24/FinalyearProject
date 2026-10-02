# Research Package — Prenatal Heart Screening (Echo + Doppler + Fetal ECG)

<aside>
🔬

**Purpose.** The research base for the prenatal heart-screening project: background, a literature survey (22 papers, plus 18 added on 19 Sep for the three-input methodology in section 2.7) and what it leaves open. The panel has not yet seen this topic's survey, so it carries into Review III.

**✅ = checked at source on 19 Sep 2026** — citation confirmed on Crossref or arXiv, and the abstract (or the paper / dataset page) read. Nothing here is written from memory.

</aside>

---

# 1. Background

- **CHD is the most common birth defect.** Fetal screening ultrasound gives five views of the heart that together can detect **90% of complex CHD**, but in practice sensitivity is **as low as 30%** (Arnaout et al., *Nature Medicine* 2021).
- **Fetal echocardiography** is a specialised ultrasound examination of the fetal heart. The standard views are the four-chamber view (4C), the left and right ventricular outflow tracts (LVOT, RVOT) and the three-vessel-trachea view (3VT).
- **AI is accurate but not yet trusted.** A 2025 meta-analysis of 15 studies found pooled sensitivity 0.89 and specificity 0.91 — but only one study had external validation, risk of bias was moderate to high, there were no prospective studies, and explainability was underexplored (*eClinicalMedicine* 2025). A 2026 review states that most fetal-ultrasound classifiers "remain opaque and miscalibrated".
- **Skilled examiners are scarce in remote regions** (*Frontiers in Cardiovascular Medicine* 2025 meta-analysis).
- **Fetal ECG** is recorded with electrodes on the mother's abdomen, where maternal and fetal signals mix. It carries electrical information (rhythm), not heart structure.
- **Pulsed-wave Doppler ultrasound** records blood-flow velocity through the heart over time — filling, ejection and their timing (function). In this project it is used as a 1-D waveform, not as a picture.
- **A full fetal heart assessment covers anatomy, function and rhythm** (AHA scientific statement, 2014) — the reason this project uses echo, Doppler and fetal ECG together.
- **India has more than 200,000 CHD births a year**, at a birth prevalence of 9 per 1,000; about one-fifth need intervention in the first year (Saxena, Indian Pediatrics 2018).

---

# 2. Literature survey — 40 papers

## 2.1 Datasets and benchmarks

| No. | Reference | What it provides | Limitation / gap |
| --- | --- | --- | --- |
| 1 ✅ | Rodríguez S. et al. *Heartbeat: a multimodal dataset of fetal echocardiography and clinical metadata for early detection of congenital heart disease.* Frontiers in Cardiovascular Medicine, 2026. | 1,475 patients, 6,215 images, 4 views, 2nd and 3rd trimester (6.50% and 7.25% CHD), three Colombian hospitals. Heart-ViT baseline on the 2nd-trimester test: sensitivity 79.17%, specificity 93.18%, AUROC 88.56%, F1 65.63% | No cross-dataset, uncertainty or calibration analysis; test set of 61 patients with only 6 CHD; no 3rd-trimester test split; still images only; referral-centre prevalence above the ~1% of the general population; access by request form. **Repository (checked 19 Sep):** released Heart-ViT weights for both trimesters and 5 baselines; the code has no licence file; the README's cross-validation F1 (2T 62.57 ± 8.37, 3T 68.00 ± 2.67) differs from the paper's 3T figure — cite the paper |
| 2 ✅ | Vega D. et al. *CARDIUM: Congenital Anomaly Recognition with Diagnostic Images and Unified Medical records.* ICCV 2025 Workshop (CVAMD); arXiv:2510.15208. | 6,558 images, 16.3% CHD, maternal clinical records; cross-attention transformer, F1 79.8 ± 4.8% | Same lab as Heartbeat, same scanners, overlapping years (mostly 2019–2023) and near-identical trimester cohorts — overlap likely, check before use; images by request form. **Repository (checked 19 Sep):** code Apache-2.0, data CC BY-NC 4.0; the clinical JSON (1,104 records, 26 maternal variables) is public; 1,103 patients with 74 CHD in the public files (paper: 79); 101 / 694 / 684 patients in the 1st / 2nd / 3rd trimester; 321 patients in both 2nd and 3rd; views not labelled; colour and power Doppler stills included |
| 3 ✅ | Sulas E. et al. *A non-invasive multimodal foetal ECG–Doppler dataset for antenatal cardiology research (NInFEA).* Scientific Data, 2021. | 60 entries from 39 women, weeks 21–27; 27-channel fetal ECG at 2048 Hz with synchronised pulsed-wave Doppler (five-chamber window, mitral and aortic flow); PhysioNet licence ODC-By 1.0 | Healthy fetuses only — no CHD labels |
| 4 ✅ | *Non-Invasive Fetal ECG Database (NIFECGDB).* PhysioNet. | 55 recordings, 2 thoracic + 3–4 abdominal channels, 1 kHz, weeks 21–40 | One subject; no diagnosis labels |
| 5 ✅ | *FOCUS: Four-chamber Ultrasound Image Dataset for Fetal Cardiac Biometric Measurement.* Zenodo, 2025. | 300 four-chamber images with cardiac and thoracic annotations and cardiothoracic ratio; CC BY 4.0 | Small; CHD labels not stated |

## 2.2 CHD detection from fetal ultrasound

| No. | Reference | What it did | Limitation / gap |
| --- | --- | --- | --- |
| 6 ✅ | Arnaout R. et al. *An ensemble of neural networks provides expert-level prenatal detection of complex congenital heart disease.* Nature Medicine, 2021. | 107,823 images from 1,326 studies at 18–24 weeks, five views. Internal test of 4,108 surveys (0.9% CHD): AUC 0.99, sensitivity 95%, specificity 96%. Also tested on outside-hospital and lower-quality images | Private data; echo only |
| 7 ✅ | *Artificial intelligence-enabled prenatal ultrasound for the detection of fetal cardiac abnormalities: a systematic review and meta-analysis.* eClinicalMedicine, 2025. | 15 studies; pooled sensitivity 0.89, specificity 0.91 | Names the gaps: one external validation, moderate–high risk of bias, no prospective studies, explainability underexplored |
| 8 ✅ | Liastuti L.D. et al. *Diagnostic accuracy of AI models in detecting CHD in the second-trimester fetus: a systematic review and meta-analysis.* Frontiers in Cardiovascular Medicine, 2025. | Second-trimester review; notes the shortage of proficient examiners in remote regions | Review, not a method |
| 9 ✅ | Tang et al. *A multicenter study on two-stage transfer learning model for duct-dependent CHDs screening in fetal echocardiography.* npj Digital Medicine, 2023. | DDCHD-DenseNet on 6,698 images and 48 videos; sensitivity from 0.973 down to 0.759 across its test sets | The drop across test sets shows shift matters |
| 10 ✅ | Lei et al. *An interpretable deep learning model for first-trimester fetal cardiac screening.* npj Digital Medicine, 2025. | 108,521 first-trimester screenings in China; 8,062 Doppler four-chamber images; interpretable model | First trimester and Doppler only |
| 11 ✅ | Yu et al. *Deep learning-based differentiation of ventricular septal defect from tetralogy of Fallot in fetal echocardiography images.* Technology and Health Care, 2024. | CNN separating two CHD types | Two classes, one task |
| 12 ✅ | Ungureanu et al. *Learning deep architectures for the interpretation of first-trimester fetal echocardiography (LIFE) — a study protocol.* BMC Pregnancy and Childbirth, 2023. | Protocol for decision support from 2D cardiac sweep videos | Protocol — no results |

## 2.3 Views, quality and segmentation

| No. | Reference | What it did | Limitation / gap |
| --- | --- | --- | --- |
| 13 ✅ | Dong et al. *A generic quality control framework for fetal ultrasound cardiac four-chamber planes.* IEEE Journal of Biomedical and Health Informatics, 2020. | Automatic quality control of four-chamber planes | One view; echo only |
| 14 ✅ | Nurmaini S. et al. *Deep learning-based computer-aided fetal echocardiography: application to heart standard view segmentation for congenital heart defects detection.* Sensors, 2021. | Segmentation of standard views | Segmentation — a crowded, U-Net-style line of work |
| 15 ✅ | Lu et al. *A YOLOX-based deep instance segmentation neural network for cardiac anatomical structures in fetal ultrasound images.* IEEE/ACM TCBB, 2024. | Instance segmentation of four-chamber structures | Segmentation, not screening decisions |
| 16 ✅ | Liu et al. *Cross-center online generalization algorithm with unadversarial consistency for fetal heart ultrasound view recognition.* Journal of Imaging Informatics in Medicine, 2026. | Adapts view recognition across centres without target labels | View recognition, not CHD |

## 2.4 Trust: uncertainty, calibration, explainability

| No. | Reference | What it did | Limitation / gap |
| --- | --- | --- | --- |
| 17 ✅ | Zhou T. et al. *Trusted multi-view deep learning classification of fetal congenital heart disease with feature-level and decision-level fusion.* arXiv:2606.15265, 2026. | Five views; uncertainty-based component for low-quality images | Private data; Dempster-style fusion of views of the same input — we fuse different inputs with an OR rule (run X1 compares both) |
| 18 ✅ | *Uncertainty-calibrated explainable AI for fetal ultrasound plane classification: a systematic review.* arXiv:2601.00990, 2026. | Finds most classifiers "opaque and miscalibrated" | Plane classification, not CHD |

## 2.5 Fetal ECG

| No. | Reference | What it did | Limitation / gap |
| --- | --- | --- | --- |
| 19 ✅ | Vullings R. *Fetal electrocardiography and deep learning for prenatal detection of congenital heart disease.* Computing in Cardiology, 2019. | Early fetal-ECG CHD detection | Conference paper |
| 20 ✅ | de Vries I.R. et al. *Fetal electrocardiography and artificial intelligence for prenatal detection of congenital heart disease.* Acta Obstetricia et Gynecologica Scandinavica, 2023. | 122 measurements (65 healthy, 57 CHD): accuracy 71%, sensitivity 63%, specificity 77%; positioned as complementary to ultrasound | Data not public |
| 21 ✅ | Jaeger et al. *Power-MF: robust fetal QRS detection from non-invasive fetal electrocardiogram recordings.* Physiological Measurement, 2024. | Fetal QRS detection | Signal processing, not CHD |
| 22 ✅ | Li et al. *Review of non-invasive fetal electrocardiography monitoring techniques.* Sensors, 2025. | Acquisition, extraction and anomaly classification; limits of existing datasets | Review |

## 2.6 Methods we build on — calibration (runs R7, S9) and optional target-sensitivity thresholds (run X4)

- ✅ Guo C. et al. *On calibration of modern neural networks.* arXiv:1706.04599, 2017 — temperature scaling
- ✅ Angelopoulos A.N., Bates S. *A gentle introduction to conformal prediction and distribution-free uncertainty quantification.* arXiv:2107.07511, 2021
- ✅ Angelopoulos A.N. et al. *Conformal risk control.* arXiv:2208.02814, 2022

---

## 2.7 Added 19 Sep — for the three-input methodology

✅ = citation confirmed and abstract or dataset page read. 🟡 = metadata confirmed only — read before citing.

| No. | Reference | What it provides | Use in this project |
| --- | --- | --- | --- |
| 23 ✅ | Donofrio M.T. et al. *Diagnosis and treatment of fetal cardiac disease: an AHA scientific statement.* Circulation, 2014. doi:10.1161/01.cir.0000437597.44550.5d | A fetal echocardiogram assesses anatomy, function and rhythm; fetal ECG and magnetocardiography complement rhythm assessment | Clinical basis for three inputs |
| 24 ✅ | Saxena A. *Congenital heart disease in India: a status report.* Indian Pediatrics, 2018. doi:10.1007/s13312-018-1445-7 | 9 per 1,000; more than 200,000 CHD births a year in India | Motivation; prevalence in run X3 |
| 25 ✅ | Behar J.A. et al. *Noninvasive fetal electrocardiography for the detection of fetal arrhythmias.* Prenatal Diagnosis, 2019. doi:10.1002/pd.5412 | NIFEADB: 12 arrhythmia + 14 normal recordings; ODC-By 1.0 | Rhythm expert data |
| 26 ✅ | Silva I. et al. *Noninvasive fetal ECG: the PhysioNet/Computing in Cardiology Challenge 2013.* Computing in Cardiology, 2013 | Set A: 75 one-minute, 4-channel recordings at 1 kHz with reference fetal QRS | Benchmark for extraction (R1–R2) |
| 27 ✅ | Țarălungă D.D. et al. *A systematic review on AI-driven noninvasive fetal ECG processing and analysis methods.* Frontiers in Digital Health, 2026. doi:10.3389/fdgth.2026.1926578 | 22 studies; no multimodal fusion; no end-to-end model; prospective validation absent | Evidence for gaps G1 and G4 |
| 28 ✅ | Maani F. et al. *FetalCLIP.* npj Digital Medicine, 2026. doi:10.1038/s41746-026-02907-9 | 210,035 fetal ultrasound images; CHD linear probe AUROC 78.72% on 418 internal 4C videos; weights public, non-commercial | Structure expert backbone |
| 29 ✅ | Saha P. et al. *Self-supervised normality learning … zero-shot CHD detection in fetal ultrasound videos.* arXiv:2503.07799, 2025 | Normal-only video models merged across 5 sites | Prior work — why normal-only modelling is not our headline |
| 30 ✅ | Yang Y. et al. *ORBIT — annotation-free cardiac phase detection in fetal echocardiography.* Medical Image Analysis, 2026. [doi:10.1016/j.media.2026.104288](https://doi.org/10.1016/j.media.2026.104288) | ED/ES frames without ECG | Prior work |
| 31 ✅ | Yang X. et al. *Hierarchical online contrastive anomaly detection for fetal arrhythmia diagnosis in ultrasound.* Medical Image Analysis, 2024. [doi:10.1016/j.media.2024.103229](https://doi.org/10.1016/j.media.2024.103229) | Doppler arrhythmia detection, 3,850 cases incl. 266 arrhythmias | Prior work (Doppler) |
| 32 ✅ | Yang X. et al. *An intelligent quantification system for fetal heart rhythm assessment (HR-IQS).* Heart Rhythm, 2024. doi:10.1016/j.hrthm.2024.01.024 | 17 cardiac time intervals from 6,498 Doppler spectra, 2,630 fetuses, 14 centres; reference ranges | Sanity check for run F3 |
| 33 ✅ | Verma A. et al. *Towards reconstruction of pulsed-wave Doppler signals from non-invasive fetal ECG.* LNCS, 2025. doi:10.1007/978-3-031-87663-9_32 | Fetal ECG → Doppler reconstruction | Prior work — translation is taken |
| 34 ✅ | Su T. et al. *Cross-modal generative framework for signal translation from fetal–maternal ECGs to fetal Doppler waveforms.* arXiv:2607.08073, 2026 (EMBC 2026) | 885 NInFEA segments, 39 pregnancies; heart-rate error 4.71 ± 0.77 bpm | Prior work — translation is taken |
| 35 ✅ | Nii M. et al. *Assessment of fetal atrioventricular time intervals by tissue Doppler and pulse Doppler echocardiography.* Heart, 2006. doi:10.1136/hrt.2006.093070 | ECG PR measurable in 61% of exams; Doppler AV intervals track it only loosely | Why the trust check uses beat timing, not PR |
| 36 ✅ | Stampalija T. et al. *Fetal and maternal heart rate confusion during intra-partum monitoring.* J Matern Fetal Neonatal Med, 2012. doi:10.3109/14767058.2011.636090 | Trans-abdominal ECG vs Doppler telemetry in 41 labours; maternal–fetal confusion is a real problem | Motivation for the trust check |
| 37 ✅ | Li J. et al. *ECGFounder.* arXiv:2410.04133 | ECG foundation model, over 10 million ECGs; single-lead version; weights MIT | Rhythm expert backbone |
| 38 ✅ | Megahed Y. et al. *First-trimester fetal heart views with ultrasound-specific self-supervised learning (USF-MAE).* arXiv:2512.24492, 2025 | Ultrasound foundation model; 6,720 first-trimester images | Alternative backbone |
| 39 🟡 | Jøsang A., McAnally D.S. *Multiplication and comultiplication of beliefs.* Int J Approximate Reasoning, 2004. doi:10.1016/j.ijar.2004.03.003 | Subjective-logic OR operator | Fusion layer |
| 40 🟡 | Han Z. et al. *Trusted multi-view classification with dynamic evidential fusion.* IEEE TPAMI, 2022. doi:10.1109/tpami.2022.3171983 | Dempster-based evidential fusion | Comparison in run X1 |

Also metadata-confirmed (🟡), to read before citing: Sensoy 2018 (evidential deep learning, arXiv:1806.01768); Wu 2024 (missing-modality survey, arXiv:2409.07825); Coppola 2024 (HuBERT-ECG, doi:10.1101/2024.11.14.24317328); Andreotti 2016 (FECGSYN, doi:10.1088/0967-3334/37/5/627).

---

# 3. What the survey shows

| Area | Papers | State |
| --- | --- | --- |
| CHD classification | 6, 9, 10, 11 | Crowded |
| Segmentation (U-Net-style) | 14, 15 | Crowded |
| Image + clinical fusion | 1, 2 | Done by the dataset authors |
| Uncertainty, calibration, explainability | 10, 17, 18 | Thin and very recent |
| Shift between centres | 9 (shows the drop), 16 (views only) | Not addressed for CHD decisions |
| Fetal-ultrasound foundation models | 28, 38 | New — not tested on public CHD data or across trimesters |
| Fetal ECG for CHD | 19, 20 | Blocked — no public labelled data |
| Fetal arrhythmia and timing from fetal ECG or Doppler; ECG → Doppler translation | 25, 27, 31–34 | Crowded or taken |
| Fusion of fetal ECG + Doppler + echo without paired patients; ECG foundation models on fetal ECG; ECG–Doppler trust check | none found | **Open in this check — the current methodology** |
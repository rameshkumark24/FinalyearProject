# Review I — Research Package

<aside>
✅

**Every claim on this page was checked against a primary source** — the challenge's own score tables and paper PDF, journal versions rather than preprints, dataset pages, and repository metadata. Where a citation could not be opened, it is marked **⚠️ verify** in the survey table below.

Nothing here is written from memory. Numbers that appear without a source were not usable and were left out.

</aside>

<aside>
⚠️

The Title, Problem Statement and Objectives below remain **DRAFT** until the approved statement is supplied. See the hub page.

</aside>

---

# 1. Project Title & Team Details

**Title (draft):** From Phone to Waveform — Robust Digitization of Photographed ECG Printouts and Clinically-Grounded Evaluation of Downstream Cardiac Classification

**Course:** U21AM704 — Project Work Phase I, Dept. of CSE (AI & ML), KPR Institute of Engineering and Technology

**Team:** Rameshkumar, Niranjana, Risvanth

---

# 2. Introduction & Background

The electrocardiogram is the most widely used cardiac diagnostic test in the world. Although hospital ECG machines acquire the signal digitally, in a large share of clinical practice the **only artefact that survives the encounter is a paper printout**. The organisers of the George B. Moody PhysioNet Challenge 2024 state the problem directly:

> "Despite the rise of digital ECGs, paper ECGs remain prevalent, especially in the Global South … ECG interpretation algorithms generally expect ECG time-series instead of images, limiting the utility of the paper ECGs."
> 

> — Reyna et al., CinC 2024
> 

This creates a structural inequity. Every modern AI ECG interpretation model consumes a **time-series**. A clinic that produces only paper is therefore locked out of the entire field, and decades of archived paper records are invisible to research.

**ECG digitization** is the task of recovering the time-series from an image of the printout. It is not simple image processing: the pixel-to-physical mapping must be recovered from the printed grid itself, because amplitude in millivolts and time in milliseconds are encoded only by the grid geometry. If the grid is misjudged, the reconstructed signal is silently mis-scaled — the waveform looks right and every clinical measurement derived from it is wrong.

In practice, the way a paper ECG becomes an image in a resource-limited clinic is that **somebody photographs it with a phone**. That is the case this project targets.

---

# 3. Literature Survey

Minimum required: **15 papers**. Listed here: **20**.

<aside>
🔍

**✅ verified** = the source document was opened and the citation and figures read directly.

**⚠️ verify** = identified via search but the full text could not be opened (paywall or access block). The citation must be confirmed before it appears in a submitted reference list.

</aside>

## 3.1 Benchmark, challenge and datasets

| No. | Reference | Contribution | Limitation / gap left open |
| --- | --- | --- | --- |
| 1 ✅ | Reyna M.A. et al. *Digitization and Classification of ECG Images: The George B. Moody PhysioNet Challenge 2024.* Computing in Cardiology, 2024. | Defines both tasks; SNR for digitization, macro F-measure for classification. Hidden set of 1,977 waveforms and 35,595 images from PTB-XL and Emory. | Organisers state SNR "does not directly capture or reflect clinical measurements". **The metric itself is unresolved.** |
| 2 ✅ | Shivashankara K.K., Deepanshi, Mehri Shervedani A., Clifford G.D., Reyna M.A., Sameni R. *ECG-Image-Kit.* Physiological Measurement, 45(5):055019, 2024. | Synthetic ECG image generator reproducing standard grid, with creases, shadows, handwriting, perspective and paper-ageing artifacts. | Synthetic only. Last repository commit Oct 2024 — dependency rot expected. |
| 3 ✅ | *ECG-Image-Database.* Physiological Measurement, 47(7):075015, 2026. doi:10.1088/1361-6579/ae85b2 | 37,191 images from 2,243 records across Germany, USA and Norway. CC BY 4.0. | **No Indian printouts.** Labels are software-generated, not clinician-adjudicated. |
| 4 ✅ | Wagner P. et al. *PTB-XL, a large publicly available electrocardiography dataset.* Scientific Data, 7:154, 2020. doi:10.1038/s41597-020-0495-6 | 21,799 records / 18,869 patients (PhysioNet v1.0.3), 10 s, 500 Hz and 100 Hz, 16-bit, 1 uV/LSB. CC BY 4.0. | Diagnostic labels only — **no outcome or follow-up data, so genuine risk prediction is not possible from it.** |
| 5 ✅ | *PTB-XL-Image-17K.* arXiv:2602.07446, Feb 2026. | 17,271 synthetic images with segmentation masks, YOLO lead boxes and ground-truth signals. | Dataset paper; reports no baseline results. Synthetic only. |

## 3.2 Digitization methods

| No. | Reference | Method | Result / limitation |
| --- | --- | --- | --- |
| 6 ✅ | Krones F., Walker B., Lyons T., Mahdi A. *Combining Hough Transform and Deep Learning Approaches to Reconstruct ECG Signals From Printouts.* arXiv:2410.14185, 2024. | Hough transform rotation correction, U-Net segmentation, mask vectorisation. | **1st place.** CV SNR 17.02; hidden-test score 12.15. Negative SNR on every phone-photo category. |
| 7 ✅ | Yu X., Huang Y., Wu J., Wang J., Cai W. *From Paper to Digital: ECG Processing with U-Net Digitization and ResNet Classification.* CinC 2024. | YOLOv8-Tiny tilt correction, ResUNet with CBAM, column-wise scan to 1-D; ResNet50 classifier. | SNR 2.202 (5/16); macro F 0.393 (8/16). |
| 8 ✅ | Shang H., Hutter C., Zhang Y. *Automated Digitization of Paper ECG Records Using Convolutional Networks.* CinC 2024, ETH Zurich. | Faster R-CNN detection plus U-Net pixel segmentation. | SNR 0.893 (6/16). Handles rotation, cropping, creases, text. |
| 9 ✅ | *Digitizing paper ECGs at scale: an open-source algorithm for clinical research.* npj Digital Medicine, 2025. arXiv:2510.19590 | Open-source full pipeline, validated on 1,596 real hospital thermal-paper printouts. | **Mean SNR 19.65 dB — but explicitly "on scanned papers".** Phone-photo performance not broken out. |
| 10 ✅ | Karbasi R., Rahimi M., Vahabie A., Moradi H. *Deep Learning-Based Digitization of Overlapping ECG Images.* arXiv:2506.10617, 2025. | Two-stage U-Net segmentation plus adaptive grid detection. | IoU 0.87; rho 0.9644 non-overlapping, 0.9641 overlapping. Addresses lead overlap specifically. |
| 11 ⚠️ | Yoon H.-C. et al. *Segmentation-based Extraction of Key Components from ECG Images.* CinC 2024. | Segmentation-based framework for classification and digitization. | 2nd place. Citation to confirm. |
| 12 ⚠️ | *ECGMiner: A flexible software for accurately digitizing ECG.* Computer Methods and Programs in Biomedicine. doi prefix S016926072400049X | General-purpose digitization software. | Paywalled — volume/year to confirm. |
| 13 ⚠️ | *Digitizing ECG image: a new method and open-source software code.* Computer Methods and Programs in Biomedicine, 2022. S0169260722002723 | Classical (non-deep-learning) digitization with released code. | Pre-dates the challenge; no standard benchmark. |
| 14 ⚠️ | *ECGScan: A method for conversion of paper electrocardiographic printouts to digital ECG files.* Journal of Electrocardiology, 2005. | Earliest systematic paper-to-digital conversion method. | Historical baseline — establishes that the problem is 20 years old and still unsolved. |

## 3.3 Classification from ECG images, and standards

| No. | Reference | Relevance |
| --- | --- | --- |
| 15 ⚠️ | *High precision ECG digitization using artificial intelligence.* Journal of Electrocardiology. S0022073625000287 (PMcardio / Powerful Medical) | Commercial system. Reported PCC above 0.91, SNR above 12.5 dB, RMSE below 0.10 mV, ~6.62% failure under extreme conditions. Closed source. |
| 16 ⚠️ | *Cardiac Classification with Multi-Scale Convolutional Neural Network From Paper ECG.* medRxiv 2025.10.05.25337357 | Direct image-to-diagnosis without a digitization step — a baseline for the comparison arm. |
| 17 ⚠️ | *Image-based deep learning for emergency electrocardiogram classification.* medRxiv 2026.06.18.26355968 | Reports performance retained across scanned and photographed ECGs; expert-adjudicated labels. |
| 18 ⚠️ | *Learning ECG Image Representations via Dual Physiological-Aware Alignments.* arXiv:2604.01526 | Representation learning directly on ECG images — recent competing paradigm. |
| 19 ⚠️ | *Hybrid deep learning framework for heart disease prediction using ECG signal images.* Scientific Reports, 2025. s41598-025-10062-6 | Representative of the crowded image-to-disease literature — useful to cite as the *saturated* area to avoid. |
| 20 ⚠️ | Kligfield P. et al. *Recommendations for the Standardization and Interpretation of the Electrocardiogram, Part I.* Heart Rhythm, 4:394–412, 2007; co-published in Circulation and JACC. | The authority for ECG recording standards and measurement definitions. Paywalled — obtain via college library before quoting any values from it. |

---

# 4. Research / Problem Gap

<aside>
🎯

**The strongest feature of this project is that the gaps are not invented — they are stated by the PhysioNet Challenge organisers themselves, in the Discussion of their own summary paper.** That is the most defensible possible basis for a research gap at Review I.

</aside>

## Gap 1 — The evaluation metric does not measure clinical usefulness

> "the SNR still does not directly capture or reflect clinical measurements that are likely to influence the downstream interpretation of an ECG"
> 

> — Reyna et al., CinC 2024, Discussion
> 

SNR is a signal-engineering metric. A clinician does not act on decibels — they act on a QT interval in milliseconds or an ST deviation in millivolts. **No published work evaluates ECG digitization in clinical measurement units.** This gap needs no GPU to close.

## Gap 2 — It is unknown whether digitization is necessary at all

> "several teams with negative SNR scores, i.e, more noise than signal, could achieve F-measure classification scores that were close to, but lower than, the highest-performing teams. This observation may suggest that intermediate time-series representations may not be necessary for ECG image interpretation, but it is more likely an indictment of our signal fidelity measure."
> 

> — Reyna et al., CinC 2024, Discussion
> 

The organisers pose the question and explicitly decline to resolve it. **Nobody has run the controlled three-arm comparison that would settle it.**

## Gap 3 — Photographs are the unsolved case, and even the commercial leader is inconsistent

From the official score table, the winning team by image type:

| Image type | SNR (winning team) |
| --- | --- |
| Colour scans, clean paper | +4.930 |
| Black-and-white scans, clean paper | +3.479 |
| Colour scans, deteriorated paper | +0.506 |
| **Mobile phone photo, clean paper** | **−1.071** |
| **Mobile phone photo, stained paper** | **−0.723** |
| **Mobile phone photo, deteriorated paper** | **−1.304** |
| **Photo of a computer monitor** | **−1.759** |

Negative SNR means more noise than signal. **No team in the field exceeded roughly +0.8 on any phone-photo category.** And on the commercial system:

> "We found that PMcardio outperformed the Challenge algorithms but, like the Challenge teams, they demonstrated variable performance across the different variants of the hidden data."
> 

> — Reyna et al., CinC 2024, Discussion
> 

## Gap 4 — Reproducibility

> "Unfortunately, we were unable to evaluate any of them, including ones for which code was available. Some solved a different problem … or they would not share their code or allow the independent evaluation of their algorithm."
> 

> — Reyna et al., CinC 2024, Discussion
> 

## Gap 5 — Geography

ECG-Image-Database covers Germany, the USA and Norway. **No Indian ECG printouts appear in any public dataset**, despite paper ECGs being the dominant format in Indian clinics.

---

# 5. Problem Statement (draft)

Twelve-lead ECGs in most Indian clinics exist only as paper printouts, and the practical way to move one into software is to photograph it on a phone. Existing digitization methods are benchmarked on flatbed scans and collapse on photographs: in the official PhysioNet Challenge 2024 score tables the winning team recorded **negative signal-to-noise ratio on every mobile-phone-photo category**, while a recent open-source method reports 19.65 dB explicitly on *scanned* paper. The artefact that actually circulates in Indian practice is the one the field handles worst.

Compounding this, the challenge organisers state that SNR "does not directly capture or reflect clinical measurements", and that it remains unresolved whether digitization is even required for automated interpretation.

This project therefore (a) quantifies digitization failure on photographed printouts in **clinical measurement units** rather than decibels alone, (b) develops a photo-robustness front-end to recover the lost accuracy, and (c) settles by controlled experiment whether digitizing before classification helps or harms diagnostic accuracy.

---

# 6. Objectives (draft)

1. **Reproduce and benchmark** at least two open-source ECG digitization pipelines on public data, reporting SNR separately for each image type to confirm and quantify the scan-versus-photograph gap.
2. **Define and validate a clinical-parameter error metric** — measuring digitization error in heart rate (BPM) and PR, QRS and QT intervals (ms) — and report its relationship to SNR.
3. **Develop a photo-robustness front-end** (perspective rectification, grid recovery, illumination normalisation) and measure the SNR and clinical-parameter error it recovers.
4. **Construct an Indian ECG printout evaluation set** using printouts on Indian ECG grid stock, photographed under realistic clinic conditions.
5. **Run a controlled three-arm comparison** — direct image classification, digitize-then-classify, and fusion — to determine whether digitization improves diagnostic accuracy under photographic degradation.

---

# 7. Time Line

| Period | Work | Milestone |
| --- | --- | --- |
| Sep 2026 (week 1–2) | Literature survey, environment setup, reproduce a baseline digitizer, scan-vs-photo demonstration | **Review II — 14/15 Sep** |
| Sep–Oct 2026 | Full benchmark across image types; clinical-parameter metric implemented and validated | **Review III — 10/12 Oct** |
| Oct–Nov 2026 | Paper drafting and submission to target venue | Scopus submission |
| Phase II | Robustness front-end, Indian evaluation set, three-arm classification comparison | Phase II reviews |

---

# A. Current Industrial Solutions & Methodology

| Solution | Type | Method | Status / limitation |
| --- | --- | --- | --- |
| **PMcardio** (Powerful Medical) | Commercial, CE-marked Class IIb under EU MDR | Grid detection and distortion correction, then deep-learning lead extraction. Under 7 s. Handles smartphone photos. | **Closest competitor.** Closed source; marketed in EU/UK only. Organisers found variable performance across image variants. |
| **npj Digital Medicine open-source algorithm** | Academic, open source | Full pipeline validated on real hospital thermal printouts | 19.65 dB — on **scans**. Photo behaviour unreported. |
| **ECG-Digitiser** (Krones et al.) | Academic, BSD-2 | Hough transform plus U-Net plus vectorisation | Challenge winner. Fails on photos. Last commit Jun 2025. |
| **ECGMiner**, **ECGScan** | Software tools | Classical computer vision | Predate deep learning; not benchmarked on the challenge data. |
| **GE MUSE, Philips IntelliSpace** and similar | Hospital ECG management systems | Store natively digital ECG (XML/SCP) | Solve the problem by **avoiding** paper — unavailable to the clinics this project targets. Explains why the paper problem persists. |

<aside>
💡

**The honest positioning.** A commercial product already claims good smartphone-photo digitization, so the project must not claim "nobody can do this". The defensible claim is that **no open, reproducible, independently-evaluated method handles photographs**, that the one commercial exception is closed and regionally restricted, and that neither has been evaluated on Indian printouts or in clinical measurement units.

</aside>

---

# B. ECG Parameters and Their Units

## B.1 The printed grid — how paper encodes physical units

Source: Shivashankara et al., Physiological Measurement 45(5):055019, 2024.

| Convention | Standard value | Meaning |
| --- | --- | --- |
| Paper speed | **25 mm/s** | Horizontal axis is time |
| Amplitude gain | **10 mm/mV** (1 mV per 10 mm) | Vertical axis is voltage |
| Fine grid square | 1 mm x 1 mm | **40 ms** horizontally, **0.1 mV** vertically |
| Coarse grid square | 5 mm x 5 mm | **0.2 s** horizontally, **0.5 mV** vertically |

<aside>
🔑

**This is the theoretical heart of the project.** The grid is the *only* carrier of physical units in the image. A photograph taken at an angle changes the apparent grid spacing non-uniformly across the page. If the digitizer estimates the grid wrongly, every millivolt and every millisecond it outputs is wrong by a scale factor — while the waveform still *looks* correct. This is why photographs break digitizers in a way that scans do not.

</aside>

## B.2 Clinical parameters a digitizer must preserve

| Parameter | Unit | What it measures | Depends on which axis |
| --- | --- | --- | --- |
| Heart rate (HR) | **BPM** (beats per minute) | Beats per minute, derived from RR | Horizontal (time) |
| RR interval | **ms** | Time between successive R peaks | Horizontal |
| PR interval | **ms** | Atrial to ventricular conduction time | Horizontal |
| QRS duration | **ms** | Ventricular depolarisation time | Horizontal |
| QT interval / QTc | **ms** | Depolarisation plus repolarisation; QTc is rate-corrected | Horizontal |
| ST deviation | **mV** (or mm on paper) | Elevation/depression — the key ischaemia marker | **Vertical (amplitude)** |
| P, R, T wave amplitude | **mV** | Wave heights | **Vertical** |
| Electrical axis | **degrees** | Mean direction of depolarisation | Both |
| Sampling frequency | **Hz** | Digital reconstruction rate (PTB-XL: 500 or 100 Hz) | — |

<aside>
⚠️

**Normal reference ranges are deliberately omitted.** Every accessible source for them was a blog or aggregator, and this project does not cite secondary sources for numbers. Obtain the AHA/ACCF/HRS statement (reference 20) through the college library and take the ranges from there.

</aside>

## B.3 Digitization quality metrics

| Metric | Unit | Note |
| --- | --- | --- |
| SNR | **dB** | Challenge metric. Negative means more noise than signal. Organisers say it does not reflect clinical measurement. |
| Pearson correlation (PCC) | dimensionless, −1 to 1 | Shape agreement. **Insensitive to amplitude scaling errors** — so it can look good while millivolts are wrong. |
| RMSE | **mV** | Absolute amplitude error. |
| IoU | dimensionless, 0 to 1 | Segmentation quality, before signal extraction. |
| **Clinical parameter error (proposed)** | **ms and BPM and mV** | **The project's proposed contribution** — error expressed in units a clinician acts on. |

---

# C. Scope for Paper Publication

## C.1 Why this is publishable

The contribution is not "another CNN". It answers two questions the field's own organisers posed and left open (Gaps 1 and 2), on a public benchmark, with released code. Negative or surprising results remain publishable here because the question itself is open.

## C.2 Realistic venue types

| Venue type | Fit | Consideration |
| --- | --- | --- |
| Domain conference (e.g. Computing in Cardiology) | Directly on topic; the challenge literature lives here | Short format, fixed annual cycle |
| Biomedical signal/image journals (e.g. Biomedical Signal Processing and Control; Computer Methods and Programs in Biomedicine; Physiological Measurement) | Strong fit — these already publish this exact literature | Longer review cycles; **check whether an APC applies** (BSPC lists an open-access APC of USD 3,470) |
| Indian/IEEE conferences | Fastest route to satisfying the requirement | Quality varies widely — verify indexing individually |

<aside>
🚩

**Verify Scopus indexing yourself on the official Scopus source list at the time of submission.** Indexing status changes and titles are de-listed. Do not rely on a journal's own website, on an aggregator, or on this page. Beware of venues that promise guaranteed acceptance and fast Scopus indexing.

</aside>

## C.3 How to reach the standard

- [ ]  **Compare against a published baseline on the same public data** — without this a paper is a demo, not research
- [ ]  **Report per-image-type results**, never a single averaged number — the averaged number is exactly what hides the photo failure
- [ ]  **Release code and the evaluation script** — directly addresses Gap 4 and reviewers reward it
- [ ]  **Use patient-independent train/validation/test splits** — patient overlap is a known failure mode in this literature and reviewers check
- [ ]  **Report negative results honestly** — the organisers' own open question makes a negative answer valuable
- [ ]  **State limitations explicitly**, including anything synthetic and any dataset the work did not cover

---

# D. End Results to Prove, and the Theory Behind Each

<aside>
🧪

Each hypothesis is stated so it can be **falsified**. A result that contradicts the hypothesis is still a result — that is what makes these safe to commit to before the experiments are run.

</aside>

## H1 — Digitization accuracy collapses from scans to photographs

**Claim:** For a fixed digitizer, mean SNR on phone photographs is significantly lower than on scans of the same records.

**Theory:** A scan is an orthographic, evenly-lit projection preserving the grid as a uniform lattice. A photograph is a **projective** transform under non-uniform illumination, so apparent grid spacing varies across the page. Since the grid is the sole carrier of the mm-to-mV and mm-to-s mapping, an error in grid estimation propagates as a multiplicative scale error over the whole reconstruction.

**Measure:** Paired SNR per image type; report the difference with a confidence interval.

## H2 — The error is dominated by geometry, not by tracing

**Claim:** On photographs, the dominant error is calibration/geometry rather than failure to segment the trace.

**Theory:** Total error decomposes into (a) segmentation error — wrong pixels identified as trace — and (b) calibration error — right pixels, wrong physical scale. These are separable: compute IoU against the mask for (a) and grid-spacing error for (b), then correlate each against final SNR.

**Measure:** IoU and estimated-versus-true grid spacing, each regressed on SNR.

**Why it matters:** If H2 holds, the fix is rectification rather than a bigger segmentation model — which is precisely what makes the project **feasible on free Colab**.

## H3 — SNR is a poor proxy for clinical usefulness

**Claim:** SNR correlates only weakly with error in clinically actionable parameters; two reconstructions with equal SNR can differ substantially in QT or ST error.

**Theory:** SNR is a whole-signal energy ratio dominated by high-amplitude regions — principally the R peak. Clinical decisions depend on **low-amplitude** features: the ST segment, the P wave, the T wave. Energy-based error is therefore concentrated where clinicians look least. The organisers say this in their own words (Gap 1).

**Measure:** Correlation between SNR and error in HR (BPM), PR/QRS/QT (ms) and ST deviation (mV).

**Why it matters:** This is the **cheapest and most novel** contribution — no GPU, no new model, and it is a metric contribution that others must then cite.

## H4 — Digitization may not improve classification

**Claim:** Under photographic degradation, classifying the image directly matches or exceeds digitize-then-classify.

**Theory:** CNNs on images learn morphology-level features that are robust to global scale error. Digitization instead bakes calibration error into the signal, and amplitude-sensitive diagnoses (ST changes, hypertrophy) inherit it. Evidence: challenge teams with negative SNR still reached macro F near the top.

**Measure:** Macro F-measure across three arms — direct image, digitize-then-classify, fusion — on identical splits.

**Either outcome publishes:** if digitization helps, that justifies the whole pipeline; if it does not, digitization is shown to be valuable for archival and interoperability rather than accuracy, and the organisers' open question is answered.

## H5 — Indian printouts degrade performance further

**Claim:** Digitizers tuned on European and US printouts lose accuracy on Indian printout formats.

**Theory:** Grid colour, thermal paper response, lead layout and print density vary by manufacturer. Models trained on PTB-XL-derived renderings encode those specific visual statistics.

**Measure:** SNR and clinical-parameter error on the Indian set versus the public benchmark.

<aside>
⚠️

**H5 carries an ethics dependency.** Real clinical printouts carry patient identifiers, so collection requires Institutional Ethics Committee review under ICMR guidelines. The dependency can be removed entirely by printing PTB-XL signals onto Indian ECG grid stock and photographing those — real paper, real camera degradation, **no patient data**. Treat any real de-identified clinical set as a bonus, never as a critical path.

</aside>
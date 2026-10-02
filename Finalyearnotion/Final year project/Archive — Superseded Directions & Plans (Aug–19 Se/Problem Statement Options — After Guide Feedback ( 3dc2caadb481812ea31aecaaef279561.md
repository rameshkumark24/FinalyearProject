# Problem Statement Options — After Guide Feedback (15 Sep 2026)

<aside>
🧭

**Why this page exists.** On 15 Sep 2026 the guide said the ECG area is fine, but ECG digitization and classification are common and already done by many groups. He asked for a different problem statement within ECG.

**Follow-up the same day:** the guide wants an **upgrade to the existing idea** — for example **early prediction** or a **software enhancement** — rather than another digitization or classification study. Options A–D below stay on file; new options should be framed that way — see [How ECG Machines Work Today & Doable Enhancements (15 Sep 2026)](How%20ECG%20Machines%20Work%20Today%20&%20Doable%20Enhancements%20%203dc2caadb48181cfb9bcd40b06836763.md).

This page records what was checked on 15 Sep: which directions are already crowded, four less-crowded options, and a recommendation. **Nothing is decided** until the guide picks an option and the Project Coordinator approves the change (General Guideline 3).

</aside>

<aside>
🔍

**How this was checked.** Web and literature searches on 15 Sep 2026. Sources marked **(opened)** were read at the source — the paper, its abstract, or the dataset page. Sources marked **(listing)** were seen only in search results; open them before citing.

A quick search cannot prove that nobody has done something. Every "gap" below means **not found in this check**, and needs a proper literature survey before it goes in the paper.

</aside>

---

# 1. Already crowded — avoid as the headline

| Direction | What already exists | Evidence |
| --- | --- | --- |
| ECG image digitization and classification | The guide's judgement. The 2024 PhysioNet Challenge alone ranked 16 teams | Guide, 15 Sep; Research Package |
| Synthetic ECG generation (diffusion, GAN) | Diffusion models trained on PTB-XL; a 2025 preprint already reports generated-ECG errors in PR, QRS, QT, QTcF and ST | arXiv 2510.05492, arXiv 2301.08227 (listing) |
| Explainable AI for ECG | Systematic reviews exist; a July 2026 preprint already evaluates explanations against guideline-defined waveform regions | arXiv 2607.24035 (listing) |
| Benchmarking ECG delineators | ECGdeli validation on LUDB and QTDB; comparisons on smartwatch and wearable data, 2024–2026 | medRxiv 2026 (listing) |
| Large language models reading ECGs | ECGInstruct (over 1 million samples) and PULSE; HeartcareGPT; npj Digital Medicine 2026 | (listing) |
| Uncertainty estimation for ECG AI | Six methods compared on 526,656 ECGs (European Heart Journal – Digital Health, 2021) | (listing) |
| Lead-misplacement detection | Lightweight deep-learning detectors, macro F1 93.42–99.61% (2024) | PubMed 38663434 (listing) |
| ECG denoising | Evaluations already include QT-interval error, not only SNR | arXiv 2301.02607 (listing) |
| Sex and demographic bias in ECG AI | Fairness-aware models already published (Scientific Reports, 2026) | (listing) |
| Re-identification of "anonymous" ECG data | Linkage-attack studies in 2024 and 2025 (85% re-identification in one); membership inference on ECG foundation models, 2026 | arXiv 2408.10228, 2508.15850, 2604.10424 (listing) |

---

# 2. Four less-crowded options — summary

| Option | Problem in one line | Data | Compute | Novelty in this check | Main risk |
| --- | --- | --- | --- | --- | --- |
| **A. Same ECG, different verdict** (recommended) | Commercial and open-source ECG algorithms may put the same ECG on different sides of a clinical cut-off | PTB-XL + PTB-XL+ (open, CC BY 4.0) | CPU; features already computed | Medium — prior work reports averages, not per-ECG decision flips | Incremental over Kligfield et al. |
| **B. Swapped electrodes, silent AI errors** | Electrode reversal may make AI models and automatic programs report false diagnoses | PTB-XL + simulated reversals | Light (inference only) | Medium — single-model tests exist | Needs pretrained classifiers; may look like classification |
| **C. Children are not small adults** | Adult-tuned measurement tools and cut-offs applied to children's ECGs | ZZU pECG (figshare, CC BY-NC-ND 4.0) | CPU | Medium — 2026 pediatric studies exist | Interval ground truth not confirmed in the dataset |
| **D. Checking the labels** | Do diagnostic labels in public ECG datasets agree with their own measurements? | PTB-XL + PTB-XL+ | CPU | Medium — ML-based label-error detection exists | Criteria must come from guidelines |

---

# 3. Option details

## A. Same ECG, different verdict — recommended

**Problem.** Digital electrocardiographs print automatic measurements — heart rate and the PR, QRS and QT intervals — and clinicians and researchers act on them. Different algorithms measure the same waveform differently. Near a clinical cut-off, a few milliseconds can change the decision: prolonged QTc or not, first-degree AV block or not, wide QRS or not.

**What already exists**

- **Am Heart J 2014** (Kligfield et al.) — four US manufacturers, 600 FDA-warehouse ECGs. Differences were small in normal subjects; in long-QT patients, differences of means reached 14.0 ms for QRS and 18.1 ms for QT. (opened — abstract)
- **Am Heart J 2018** — 800 ECGs (normal, moxifloxacin, LQT1, LQT2). Pairwise mean differences 0.2–3.6 ms (PR), 0.1–8.1 ms (QRS), 0.1–9.3 ms (QT), larger in long-QT subjects. (opened — abstract)
- **J Electrocardiol 2020** — seven programs; the number of large, clinically significant errors differed up to two-fold between programs. (opened — abstract)
- **PTB-XL+** (Scientific Data 2023) — features for PTB-XL from Uni-G and 12SL, commercial algorithms "distributed in millions of ECG devices world-wide", and from open-source ECGDeli. Agreement is reported as correlation; no cut-off analysis. (opened)
- **npj Cardiovascular Health 2026** — benchmarks each algorithm against PR and QTc cut-offs, but does not report disagreement between the algorithms themselves. (opened)

**Gap targeted — not found in this check.** How often the **same ECG gets a different clinical decision** depending on which algorithm measured it, on a large open clinical dataset, including open-source tools — and what drives the disagreement.

**Data.** PTB-XL+ v1.0.1 on PhysioNet: open access, CC BY 4.0, 2.0 GB uncompressed, features in standard units (mV, ms), ECGDeli fiducial points, 12SL statements. PTB-XL signals for running NeuroKit2 as a fourth algorithm.

**Why it fits the team.** CPU only, no training. Features are already computed, so a first real result is possible within days. It keeps the clinical-parameter idea (HR, PR, QRS, QT), NeuroKit2 and PTB-XL.

**Risks.** Incremental over Kligfield et al. — read those papers in full first. Cut-offs (QTc formula and threshold, PR, QRS) must come from the AHA/ACCF/HRS recommendations. PTB-XL is German data; do not claim the result transfers to Indian machines without evidence.

## B. Swapped electrodes, silent AI errors

**Problem.** Electrode reversal is reported in 0.4–4% of recordings (listing) and can mimic myocardial infarction. How AI models and automatic programs respond is rarely tested.

**What already exists.** Deep-learning reversal detectors (2024, listing); a published method for simulating arbitrary electrode reversals (2019, listing); one AI model for left-ventricular dysfunction tested on 12 reversal morphologies from 681 ECGs kept an AUROC of 0.893–0.971 (Diagnostics 2025, opened).

**Gap targeted — not found in this check.** A multi-model, multi-diagnosis audit: which false diagnoses each reversal produces, and whether a lightweight detector placed in front prevents them.

**Data.** PTB-XL; limb-lead reversals can be simulated mathematically.

**Risks.** Needs pretrained classifiers, so the guide may see it as classification. Chest-lead swaps are harder to simulate.

## C. Children are not small adults

**Problem.** Children's ECGs differ from adults' in heart rate, intervals and amplitudes, but many automatic tools and cut-offs are adult-derived.

**What already exists.** Machine-learning pediatric reference intervals from 35,088 ECGs (Physiological Measurement 2026, opened — abstract). Automated QTc may be overestimated in children — 385 healthy and 208 HCM subjects (J Electrocardiol 2026, opened — abstract). Adult-trained ECG-age models perform worse in children (listing).

**Gap targeted — not found in this check.** An age-band analysis of how open-source measurement tools fail or drift on a large open pediatric dataset, and how many children adult cut-offs would flag.

**Data.** ZZU pECG (Scientific Data 2025, opened): 14,190 records from 11,643 children aged 0–14; 500 Hz; 12,334 twelve-lead and 1,856 nine-lead records; 5–120 s; 19 cardiovascular disease types; age in days and sex. Hosted on figshare under **CC BY-NC-ND 4.0** — fine for academic analysis, but modified data cannot be redistributed.

**Risks.** The paper does not state that interval measurements are included, so accuracy may not be measurable directly — check the files. Pediatric reference ranges must come from primary sources.

## D. Checking the labels

**Problem.** AI benchmarks trust dataset labels. Some labels have numeric definitions — first-degree AV block needs a prolonged PR, bundle branch block needs a wide QRS, bradycardia and tachycardia depend on heart rate — so they can be checked against measurements.

**What already exists.** A deep network trained on proprietary data flagged 515 possible rhythm-label errors in PTB-XL, 2.36% (IEEE 2023, listing). The PTB-XL+ paper compares 12SL statements with the cardiologists' labels (a median MCC of 0.45 appears in search summaries — confirm in the paper).

**Gap targeted — not found in this check.** A transparent audit using guideline criteria and three measurement algorithms, and the effect of flagged labels on benchmark results.

**Data.** PTB-XL + PTB-XL+. CPU.

**Note.** Uses the same data as Option A, so it works well as A's second contribution.

---

# 4. Recommendation — Option A, with D as its second contribution

- **Clearly neither digitization nor classification.** It asks whether the numbers that clinicians and AI rely on are consistent.
- **Data is open and ready now.** CPU only; a first real chart is possible before Review II if the guide agrees quickly.
- **Keeps the useful work.** The clinical-parameter idea, NeuroKit2, PTB-XL, Bland-Altman-style measurement analysis.
- **This type of study gets published** — the prior comparisons appeared in the American Heart Journal, the Journal of Electrocardiology and Scientific Data.

## Draft title (for the guide — not approved)

> **Same ECG, Different Verdict: Clinical-Decision Disagreement Between Commercial and Open-Source Automated ECG Measurement Algorithms**
> 

## Draft problem statement (not approved)

Digital electrocardiographs print automatic measurements — heart rate and the PR, QRS and QT intervals — and both clinicians and AI researchers act on them. Different algorithms measure the same waveform differently. Earlier comparisons of commercial programs found mean differences that are small in normal subjects but larger in patients with long-QT syndrome, and up to two-fold differences between programs in the number of large, clinically significant errors. Those studies report averages and error counts on reference sets. They do not report how often the same ECG falls on different sides of a clinical cut-off — prolonged QTc, first-degree AV block, wide QRS — depending on which algorithm measured it, and they do not include freely available open-source tools. This project quantifies clinical-decision disagreement between two commercial algorithms (Uni-G and 12SL) and open-source tools (ECGDeli and NeuroKit2) on the open PTB-XL clinical ECG dataset and its PTB-XL+ feature release, identifies the ECG characteristics that drive disagreement, and measures how far diagnostic labels used to train and benchmark ECG AI agree with their own measurements.

## Draft objectives (not approved)

1. **Measure agreement** between Uni-G, 12SL, ECGDeli and NeuroKit2 for HR, PR, QRS and QT/QTc on PTB-XL — bias and limits of agreement. *(Phase I)*
2. **Quantify clinical-decision discordance** — the share of ECGs placed on different sides of guideline cut-offs by different algorithms. *(Phase I)*
3. **Identify what drives disagreement** — diagnosis class, heart rate, signal quality, age and sex. *(Phase I)*
4. **Audit diagnostic labels** that have numeric definitions against the measurements, and measure the effect of flagged labels on benchmark results. *(Phase I → II)*
5. **Build a disagreement flag** that warns when a measurement sits near a cut-off and algorithms disagree. *(Phase II)*
6. **Release** the analysis code and per-ECG discordance tables. *(Throughout)*

---

# 5. Before switching — decisions and approvals

- [ ]  Guide picks an option (A–D), or asks for more options
- [ ]  Project Coordinator approves the change of problem statement (General Guideline 3); record it in the Log Book
- [ ]  Decide what Review II on 17 Sep presents — the new direction, or the current deck plus the guide's feedback
- [ ]  Read Kligfield et al. 2014 and 2018 and the 2020 seven-program study in full before claiming the gap
- [ ]  Get the AHA/ACCF/HRS recommendations from the college library for the clinical cut-offs
- [ ]  Once decided: update the hub title and problem statement, the Review II page, the Canva deck and the Roadmap

---

# 6. Sources

**Opened**

- [PTB-XL+ paper — Scientific Data 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10183020/)
- [PTB-XL+ v1.0.1 — PhysioNet](https://physionet.org/content/ptb-xl-plus/1.0.1/)
- [FeatureDB validation — npj Cardiovascular Health 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13392070/)
- [Kligfield et al., Am Heart J 2014 — PubMed 24439975](https://pubmed.ncbi.nlm.nih.gov/24439975/)
- [Automated interval measurements by widely used algorithms, Am Heart J 2018 — PubMed 29898835](https://pubmed.ncbi.nlm.nih.gov/29898835/)
- [PR, QRS and QT by seven ECG programs, J Electrocardiol 2020 — PubMed 33142185](https://pubmed.ncbi.nlm.nih.gov/33142185/)
- [ZZU pediatric ECG database — Scientific Data 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12106700/)
- [Data-driven pediatric ECG reference intervals, Physiol Meas 2026 — PubMed 41570383](https://pubmed.ncbi.nlm.nih.gov/41570383/)
- [Automated vs measured QTc in children, J Electrocardiol 2026 — PubMed 42556221](https://pubmed.ncbi.nlm.nih.gov/42556221/)
- [AI LVSD model with lead-reversal tests — Diagnostics 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12346549/)

**Listing only — open before citing**

- [Lead-misplacement detection, 2024 — PubMed 38663434](https://pubmed.ncbi.nlm.nih.gov/38663434/)
- [Simulating arbitrary electrode reversals, 2019 — PubMed 31266252](https://pubmed.ncbi.nlm.nih.gov/31266252/)
- [Label errors in large ECG datasets — IEEE 2023](https://ieeexplore.ieee.org/document/10081629/)
- [Synthetic ECG generation with interval errors — arXiv 2510.05492](https://arxiv.org/pdf/2510.05492)
- [Guideline-grounded XAI evaluation for ECG — arXiv 2607.24035](https://arxiv.org/html/2607.24035v1)
- [Multimodal LLMs and 12-lead ECG images — npj Digital Medicine 2026](https://www.nature.com/articles/s41746-026-02551-3)
- [Uncertainty estimation for 12-lead ECG AI — EHJ Digital Health 2021](https://academic.oup.com/ehjdh/article/2/3/401/6272223)
- [Gaussian-process ECG denoising with QT evaluation — arXiv 2301.02607](https://arxiv.org/pdf/2301.02607)
- [Demographic-aware fair ECG model — Scientific Reports 2026](https://www.nature.com/articles/s41598-026-54206-8)
- [Linkage attacks on public ECG data — arXiv 2508.15850](https://arxiv.org/abs/2508.15850)
- [Client re-identification in ECG datasets — arXiv 2408.10228](https://arxiv.org/pdf/2408.10228)
- [Membership inference on ECG foundation encoders — arXiv 2604.10424](https://arxiv.org/pdf/2604.10424)
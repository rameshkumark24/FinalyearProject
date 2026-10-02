# Live Documentation — Paper Draft & Progress Log

<aside>
🎯

**Document as the work happens.** This page is three things at once: the **Log Book** required by Student Guideline 2, the evolving **paper draft**, and the evidence Review III asks for ("Challenges Faced & Solutions", "Work Completed vs. Planned Work").

</aside>

---

# 1. How to log

<aside>
📏

One entry per working session, dated, with who did it. Five headings:

- **Did** — what was actually done
- **Result** — numbers, even bad ones
- **Decided** — what was chosen and why the alternative was rejected
- **Went wrong** — errors, dead ends, corrections
- **Open** — what is unresolved
</aside>

---

# 2. Paper draft — section status

| Section | Status | Source material |
| --- | --- | --- |
| Title | Draft on the hub, awaiting approval | Hub |
| Abstract | Waits for results | — |
| 1. Introduction | Sources ready | Research Package §1 — Arnaout 2021, the 2025 meta-analyses, AHA 2014, Saxena 2018, the 2026 fetal-ECG review |
| 2. Related work | Sources ready — 40 papers (22 + 18 for the three-input method) | Research Package §2 |
| 3. Methods | Designed — awaiting approval | Methodology Research page §5; Experiment Plan runs R, F, C, S, X |
| 4. Results | **Empty** — R, F and C can start on open data; S waits for access | — |
| 5. Discussion | Empty | — |
| 6. Limitations | Partly known | Methodology Research page §5.7 — no real patients with all three inputs; Doppler healthy-only; NIFEADB 26 subjects; same-lab echo datasets; no Indian data |
| References | 40 papers in the Research Package (✅ read at source; 🟡 metadata only, to read) | Research Package |

---

# 3. Progress log

<aside>
🗄️

Entries for the earlier directions (7–19 Sep, including the echo-only plan of 19 Sep) are kept on the Archive page as part of the Log Book record. This log holds only the current project.

</aside>

## 2026-09-17 — Review II

**Did**

- Presented at Review II

**Result**

- Panel feedback, as reported by Rameshkumar: use **Connected Papers** to check similar papers; **1D-CNN**-related papers are already plentiful; medical imaging work **nearly always uses U-Net**; ECG signals are already available as **digital input**, so why image-type questions

**Decided**

- The ECG-image directions were dropped

**Went wrong / corrected**

- Record which deck version was presented

**Open**

- Attach the panel's written remarks, if any

## 2026-09-19 (evening) — Three inputs decided; methodology researched

**Did**

- The team decided the project uses three inputs: **fetal ECG, ultrasound and echocardiography**, and asked for a fresh search for the best methodology, replacing the earlier echo-only plan
- Checked datasets at source (PhysioNet pages for NInFEA, NIFEADB and CinC 2013; the Heartbeat paper), searched Europe PMC, arXiv and OpenAlex, and ran a citation map on the key datasets (NInFEA cited by 30, NIFEADB by 99, de Vries 2023 by 27)
- Compared seven candidate methodologies (M1–M7) and wrote the Methodology Research page; rewrote the hub, Experiment Plan, Research Gaps, Novelty Check, Roadmap, Learning Path, How It Works and the Tanglish briefing; added 18 references to the Research Package

**Result**

- **No public dataset has all three inputs for one fetus**, and only echo has public CHD labels. Only NInFEA pairs two inputs (fetal ECG + Doppler), healthy fetuses only
- Crowded or taken: CHD classification from echo, segmentation, fetal-ECG extraction, fetal-arrhythmia classifiers on NIFEADB, Doppler timing AI, fetal ECG → Doppler translation (2025, 2026)
- Open in our check: combining all three inputs; fusion without paired patients; ECG foundation models on fetal ECG; fetal ECG–Doppler agreement as a trust check; FetalCLIP on public CHD data

**Decided (recommendation, pending approval)**

- **Structure–Function–Rhythm framework:** echo → structure expert (FetalCLIP); Doppler → function expert; fetal ECG → rhythm expert (ECG foundation models); a fetal ECG–Doppler trust check; subjective-logic OR-fusion in which a missing input counts as full uncertainty
- Rejected: one network over concatenated features (needs paired patients; random matching invents patients); ECG → Doppler translation (taken); single-input networks as the headline (crowded)
- "Ultrasound" is defined as the **Doppler waveform**, because echocardiography is itself ultrasound
- Calibration, trimester shift, missing-view handling and optional target-sensitivity thresholds are part of the method (runs S6, S8, S9, X4)

**Went wrong / corrected**

- The Research Package listed NInFEA as CC BY 4.0; PhysioNet lists **ODC-By 1.0** — corrected
- FetalCLIP's licence differs between the paper (CC BY-NC-ND 4.0) and Hugging Face (CC BY-NC 4.0) — read the licence file before use

**Open**

- Guide and Coordinator approval of the methodology (General Guideline 3)
- Start D1–D2, R1–R2 and F1–F2 on open data now; echo access still pending
- Read the 🟡 method papers before citing them
- AI acknowledgement: this research and the rewrite were AI-assisted; each member must read the sources behind their own expert

## 2026-09-19 (night) — Heartbeat and CARDIUM repositories checked

**Did**

- Read both GitHub repositories at source (README, file tree, licence, data loaders) and analysed CARDIUM's public clinical JSON and trimester file with a Python script

**Result**

- **Heartbeat:** 6,215 still images, 4 labelled views, 4 metadata fields; 2T 723 + 61 test patients (44 + 6 CHD), 3T 690 patients (50 CHD) with no test split. **Released Heart-ViT weights for both trimesters and 5 baselines** — the reference result and a first trimester-shift test need no training. No ECG, Doppler signal or video. Code has no licence file
- **CARDIUM:** 6,558 images arranged per patient; views not labelled; colour and power Doppler stills included. Clinical JSON public in the repo: 1,104 records, 26 maternal variables. Trimester file: 1,103 patients, 74 CHD (paper says 79); 101 / 694 / 684 patients in the 1st / 2nd / 3rd trimester; **321 patients (25 CHD) scanned in both 2nd and 3rd**. Code Apache-2.0, data CC BY-NC 4.0
- **Overlap is likely:** same lab, scanners and years, and near-identical 3rd-trimester cohorts (684 / 50 CHD vs 690 / 50 CHD)

**Decided**

- Heartbeat = main data for the structure expert; CARDIUM = second source only after the overlap is ruled out, plus same-fetus consistency (S10), a descriptive first-trimester look (S11) and an optional clinical prior (X5)
- Neither repository helps the rhythm or function experts
- Added runs D4 (image-hash overlap check), D5, S10, S11, X5; S2 and S8 now start from the released weights
- Rewrote the Tanglish page to explain the final method step by step

**Went wrong / corrected**

- Heartbeat's 2T test has only 6 CHD patients, so one case moves sensitivity by 16.7 points — pooled out-of-fold results will be reported alongside it
- Released counts differ slightly from the papers (Heartbeat 784 vs 785 2T patients; CARDIUM 74 vs 79 CHD) — cite the papers for their results, report received counts for ours

**Open**

- Submit both access forms; ask the authors about overlap
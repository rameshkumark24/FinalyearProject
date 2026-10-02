# Roadmap — Steps to Review III & the Paper (19 Sep → 12 Oct 2026)

<aside>
🗺️

**Purpose.** Every step from today to Review III and the start of the paper, in order, for the three-input Structure–Function–Rhythm framework. Tick a box only when its result is logged in the Progress Log. Run IDs (R, F, C, S, X) refer to the Experiment Plan.

</aside>

<aside>
🟢

**Critical path changed for the better.** The fetal ECG and Doppler parts (R, F, C) use **open PhysioNet data** — work starts today without any form. Only the echo part (S) waits for Heartbeat / CARDIUM access. If echo access is not granted by **26 September**, Review III shows R, F and C results plus the fusion design, and says so plainly.

</aside>

---

# Overview

| Stage | Dates | Done when |
| --- | --- | --- |
| 0. Approvals and access | 19–22 Sep | Guide and Coordinator approval recorded; access forms submitted |
| 1. Open data and preprocessing | 19–26 Sep | Data cards and split files committed; fetal QRS and Doppler envelopes working |
| 2. Experts | 27 Sep – 5 Oct | R, F and C runs logged; S runs logged if data arrived |
| 3. Fusion | 3–7 Oct | X1–X3 logged |
| 4. Lock results | 6–7 Oct | Test sets run once; paper gate ticked |
| 5. Start the paper | from 6 Oct | Methods and Results drafted |
| 6. Review III | 10 & 12 Oct | Presented; remarks logged |

---

# Stage 0 — Approvals and access (19–22 Sep)

- [ ]  **0.1** Guide approval of the three-input methodology (Methodology Research page) — Rameshkumar · 20 Sep
- [ ]  **0.2** Project Coordinator approval (General Guideline 3), recorded in the Log Book — Rameshkumar · by 22 Sep
- [ ]  **0.3** Submit the Heartbeat and CARDIUM access forms, with the guide named as supervisor — Niranjana · 19 Sep
- [ ]  **0.4** Ask the authors whether CARDIUM and Heartbeat share patients (the numbers suggest they do) — Niranjana · 20 Sep; then the image-hash check D4 once both datasets arrive
- [ ]  **0.5** Connected Papers: build the ten seed graphs and fill the comparison table — all · 21 Sep
- [ ]  **0.6** Agree individual contributions — all · 20 Sep

# Stage 1 — Open data and preprocessing (19–26 Sep)

- [ ]  **1.1** Link the existing empty repo [github.com/rameshkumark24/FinalyearProject](http://github.com/rameshkumark24/FinalyearProject); README and a `.gitignore` that keeps all data and weights out; a Colab GPU notebook — Rameshkumar
- [ ]  **1.2** Download NInFEA, NIFEADB and CinC 2013 set A; write a data card for each; commit split files (D1–D2) — Niranjana
- [ ]  **1.3** Maternal ECG removal and fetal QRS detection (R1–R2) — Niranjana
- [ ]  **1.4** Doppler envelopes and cycle detection (F1–F2) — Risvanth
- [ ]  **1.5** Load FetalCLIP and the three ECG foundation models; read their licences — Rameshkumar
- [ ]  **1.6** Download CARDIUM's public clinical JSON and trimester file; data card (D5) — Niranjana

# Stage 2 — Experts (27 Sep – 5 Oct)

- [ ]  **2.1** Rhythm expert: baselines, leakage check, foundation-model probes, calibration (R3–R7) — Niranjana
- [ ]  **2.2** Function expert: timing measures and healthy range (F3–F4) — Risvanth
- [ ]  **2.3** Trust check: agreement, corrupted pairs, false-alarm gate (C1–C3) — Risvanth
- [ ]  **2.4** Structure expert on access: start with the **released Heart-ViT weights** (S2, and S8 with no training), then S1, S3–S9; otherwise the FOCUS fallback — Rameshkumar

# Stage 3 — Fusion (3–7 Oct)

- [ ]  **3.1** Fusion rules, missing-input matrix, simulated cohorts (X1–X3) — Rameshkumar, with all

# Stage 4 — Lock results (6–7 Oct)

- [ ]  **4.1** Run the frozen test sets once; produce figures; tick the paper gate on the Experiment Plan — all

# Stage 5 — Start the paper (from 6 Oct)

- [ ]  **5.1** Shortlist venues with the guide; check indexing on the official Scopus source list. Ask whether an early conference paper on the open-data parts (R, F, C) would help Review III's publication status — Rameshkumar
- [ ]  **5.2** Overleaf with the venue template; Zotero with the Research Package references — Niranjana
- [ ]  **5.3** Methods and Results first draft by 9 Oct — all

# Stage 6 — Review III (10 & 12 Oct)

- [ ]  **6.1** Review III content page and deck — Canva needs re-authorisation first — Rameshkumar
- [ ]  **6.2** Present, then log the panel's remarks the same day — all
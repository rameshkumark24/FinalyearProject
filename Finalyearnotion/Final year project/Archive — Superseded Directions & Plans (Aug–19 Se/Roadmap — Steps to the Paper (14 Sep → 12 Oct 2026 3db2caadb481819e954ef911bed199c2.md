# Roadmap — Steps to the Paper (14 Sep → 12 Oct 2026)

<aside>
🗺️

**What this page is.** Every step from today (14 Sep 2026) to the point where the Scopus paper is being written, in the order it has to happen. Tick a box only when its **Done when** line is true.

Built from [Review II — Presentation Content (17 Sep 2026)](Review%20II%20%E2%80%94%20Presentation%20Content%20(17%20Sep%202026)%203db2caadb48181f6899fe7de764822c9.md), [Experiment Plan — What to Try, in Order](Experiment%20Plan%20%E2%80%94%20What%20to%20Try,%20in%20Order%203d42caadb481819e95bbf1c6159f198f.md), [Learning Path — What We Need to Learn](Learning%20Path%20%E2%80%94%20What%20We%20Need%20to%20Learn%203d42caadb4818125bdafcc47836b6102.md), [Review Schedule & What Each Review Expects](../Review%20Schedule%20&%20What%20Each%20Review%20Expects%203d42caadb4818196859ee2d273d32058.md) and [Rules & Regulations — Phase I](../Rules%20&%20Regulations%20%E2%80%94%20Phase%20I%203d42caadb48181b9bfc8f0a092b1b677.md). Where this page suggests a number (such as how many records), it says it is a suggestion.

</aside>

<aside>
📓

**The documentation rule — applies to every step.**

1. Same day: add a Progress Log entry in [Live Documentation — Paper Draft & Progress Log](Live%20Documentation%20%E2%80%94%20Paper%20Draft%20&%20Progress%20Log%203d42caadb48181939618f8fc0aa051e3.md) — **Did · Result · Decided · Went wrong · Open**.
2. A number goes into Notion, a slide or the paper only if a run produced it. Save the command and the output file it came from.
3. Each step names **→ Paper**: the paper section it feeds. That is how the paper gets written from Notion instead of from memory.
4. Note what was AI-assisted, so it can be acknowledged (Student Guideline 9).
</aside>

<aside>
👥

**Owners below are the proposed split** from the Learning Path — **Rameshkumar:** pipeline and benchmark harness · **Niranjana:** data and image generation · **Risvanth:** clinical-parameter metric and evaluation. Step 0.4 confirms or changes it. Every member must still be able to explain the whole pipeline.

</aside>

---

# Overview

| Stage | Dates | Goal | Gate — the stage is done when |
| --- | --- | --- | --- |
| **0. Approvals & admin** | 14–16 Sep | Paper trail and team split | Guide approval and contributions recorded |
| **1. Minimum prototype** | 14–16 Sep | One real scan-versus-photo SNR chart | Chart on slide 12 — or the failure documented |
| **2. Review II** | 16–17 Sep | Present and log the feedback | Panel remarks in the Progress Log |
| **3. Foundation + A1** | 18–24 Sep | Frozen splits, benchmark harness, error decomposition | A1 decision logged |
| **4. Clinical-parameter metric** | 22 Sep – 1 Oct | The headline contribution; second baseline | Clinical-error table and SNR-versus-error figure on dev |
| **5. Robustness study** | 25 Sep – 4 Oct | Per-factor degradation table | B1 table complete on dev |
| **6. Lock results** | 2–5 Oct | Test set run once; code released | All six paper-gate boxes ticked |
| **7. Start writing the paper** | from 5 Oct | Venue, template, outline, first sections | Methods and Results drafted in Overleaf |
| **8. Review III** | 10 & 12 Oct | Phase I results and publication status | Remarks logged |

<aside>
⚖️

**Honest check on the calendar.**

- Stage 1 has three days and depends on two tools with known dependency risk (ECG-Image-Kit's last commit was Oct 2024). Its stopping rule is real: if the baseline does not run by the evening of 16 Sep, present the documented failure.
- Stages 3–6 fit into 18 days only if every stopping rule is obeyed.
- **Paper writing formally starts once results are locked (Stage 6).** Do not write Results from dev-set numbers. Methods notes can build up from Stage 4 onward.
- **If behind, cut in this order:** A2 and A3 move to Phase II → then B5. **Never cut** A1, the clinical-parameter metric, B1, the second baseline or the frozen test set — the paper gate needs them.
</aside>

---

# Stage 0 — Approvals & admin (14–16 Sep)

- [ ]  **0.1 Confirm the Review II date and time slot** — Rameshkumar · 15 Sep
    - **Why:** the team was told 17 Sep; the department schedule (15.07.2026) says 14–15 Sep. Slide 1 still shows Review I's 10.00–10.50 AM.
    - **Done when:** the date and slot are written on the hub and on slide 1.
- [ ]  **0.2 Guide review of the Review II page and deck** — Rameshkumar, all attend · 15 Sep
    - **Ask for, and write down:** (a) approval of the five objectives (General Guideline 2); (b) "diagnostic classification" instead of "risk prediction" — PTB-XL has no outcome data; (c) Phase I scope — classification is Phase II; (d) which venue types he prefers for the paper.
    - **Done when:** his approval or requested changes are on the hub with the date, and the hub checkbox is ticked.
    - **→ Paper:** Introduction — the objectives become the contributions list.
- [ ]  **0.3 Get the panel's topic direction in writing** — Rameshkumar · by 17 Sep
    - **Why:** General Guideline 3 needs prior Project Coordinator approval for a change of title, scope or methodology. The panel directed the change; the record just has to exist.
    - **Done when:** it is in the Review I remarks or the Log Book, and the hub says where.
- [ ]  **0.4 Agree who owns what** — all three · 15 Sep
    - **How:** confirm or change the proposed split. One owner per deliverable.
    - **Done when:** the hub's Course & Team table, Review II §10 and slide 13 are filled.
    - **→ Paper:** Author contributions.
- [ ]  **0.5 Manual Canva fixes** — Rameshkumar · 16 Sep
    - Slides 5 and 6: body text from bold back to regular. Slide 1: time slot from 0.1.
- [ ]  **0.6 Bring the department Log Book up to date** — all · 16 Sep
    - Copy in the 7 Sep and 14 Sep Progress Log entries (Student Guideline 2). From now on, update it the same day as Notion.

---

# Stage 1 — Minimum prototype for Review II (14–16 Sep)

<aside>
🎯

**The one thing that must exist:** a scan-versus-photo SNR chart on real output from a real baseline (Learning Path). Everything in this stage serves that chart.

</aside>

- [ ]  **1.1 Set up GitHub and Colab** — Rameshkumar · 15 Sep
    - **How:** clone the existing empty repository; create `data/`, `src/`, `notebooks/`, `results/`, `figures/`; keep `data/` out of git; start `requirements.txt` with exact versions and note the Python version; keep data on Google Drive so a Colab disconnect loses nothing; add all three as collaborators.
    - **Done when:** each member has pushed one commit.
    - **→ Paper:** Implementation details; code release (C6).
- [ ]  **1.2 Load and plot one PTB-XL record** — Niranjana · 15 Sep
    - **How:** download PTB-XL v1.0.3 from PhysioNet — a handful of records is enough for now; read one 500 Hz record with wfdb-python; plot all 12 leads.
    - **Done when:** the plot is in `figures/` and in the Progress Log.
    - **→ Paper:** Datasets.
- [ ]  **1.3 Generate matched scan-like and photo-like images** — Niranjana · 15–16 Sep
    - **How:** install ECG-Image-Kit and record every dependency fix. Render the **same records** (suggested: 10) twice — a clean set, and a degraded set using flags from Experiment Plan §2 such as `--wrinkles`, a lower `-r` resolution or `--random_bw`. Always add `--store_config --lead_bbox`. Save the exact command lines.
    - **Label honestly:** these are *synthetic photo-like* images, not phone photographs.
    - **Done when:** two matched folders and the saved commands are in the repo.
    - **→ Paper:** Methods — image generation (M2).
- [ ]  **1.4 Run one baseline end to end on one image** — Rameshkumar · 15–16 Sep
    - **How:** start with ECG-Digitiser (Krones et al., BSD-2 — the licence is clear). Inference only. Check whether pretrained weights are provided; if it needs training, record that as a blocker rather than working around it this week. Leave Open-ECG-Digitizer until its LICENSE is read (step 3.3).
    - **Stopping rule:** not running by the evening of 16 Sep → stop, and write up the error, the dependency and what was tried. That becomes slide 12.
    - **Done when:** one digitized 12-lead signal file exists.
    - **→ Paper:** Methods — baselines (M3).
- [ ]  **1.5 Implement alignment and SNR** — Risvanth · 15–16 Sep
    - **How:** implement SNR from the definition in Reyna et al. (CinC 2024), including its alignment step. Test it on cases whose answer you can work out by hand from that definition — for example, a signal against itself and an all-zero prediction.
    - **Done when:** the tests pass and the code is committed.
    - **→ Paper:** Methods — evaluation metrics (M4, M5).
- [ ]  **1.6 Produce the scan-versus-photo SNR chart** — Rameshkumar + Risvanth · 16 Sep
    - **How:** run the baseline on both folders; compute SNR per image; chart mean SNR per image type, showing the number of records.
    - **Label it:** "prototype, n = 10, synthetic images" — a demonstration, not a result.
    - **Done when:** the chart and table are on slide 12 and in Review II §9.
    - **→ Paper:** not directly — the final figure is regenerated in Stage 6.
- [ ]  **1.7 Stretch: NeuroKit2 on one pair** — Risvanth · 16 Sep, only if 1.6 is done
    - Delineate one ground-truth and digitized pair; show heart-rate and QRS-duration error.

---

# Stage 2 — Review II (16–17 Sep)

- [ ]  **2.1 Rehearse** — all · 16 Sep
    - Each member runs the whole deck once and answers the "Likely questions" table on the Review II page. Each explains their own part **and** the whole pipeline (Student Guideline 6).
- [ ]  **2.2 Present Review II** — all · 17 Sep
    - Bring the Log Book. Keep the 20-paper survey table ready as a backup slide — the ECG survey has never been shown to a panel.
- [ ]  **2.3 Log the review the same day** — Rameshkumar · 17 Sep
    - Every question asked, every remark, every change requested. If the panel changes title, scope or methodology, get Coordinator approval (General Guideline 3) and update this page before continuing.
    - **→ Paper:** panel questions often become Discussion and Limitations points.

---

# Stage 3 — Evaluation foundation + A1 (18–24 Sep)

- [ ]  **3.1 Freeze the dev and test sets** — Niranjana · 18–19 Sep
    - **How:** split by patient, so no patient appears in both. PTB-XL's metadata carries patient IDs and recommended stratified folds — read the PhysioNet page and follow its recommendation. Save the record-ID lists in the repo and link them here.
    - **Rule:** the test set is not touched until step 6.1.
    - **→ Paper:** Experimental setup.
- [ ]  **3.2 Decide experiment size before running** — Rameshkumar · 19 Sep
    - **How:** time per image (measured in Stage 1) × images × conditions must fit the Colab budget. Write down records per condition.
    - **→ Paper:** Experimental setup.
- [ ]  **3.3 Read the Open-ECG-Digitizer LICENSE and choose the second baseline** — Rameshkumar · 19 Sep
    - **Why:** Objective 1 and the paper gate need at least two baselines. Its licence reads "Other". If it does not permit this use, choose another open pipeline from the survey with a clear licence.
    - **Done when:** the decision and the reason are logged.
- [ ]  **3.4 Build the benchmark harness** — Rameshkumar · 18–22 Sep
    - **How:** one command takes a folder of images and writes one CSV row per image: record, image type, factor, SNR, PCC, RMSE. Settings in a config file; fixed random seeds.
    - **Done when:** it reproduces the Stage 1 numbers exactly.
    - **→ Paper:** Implementation; code release (C6).
- [ ]  **3.5 Write expected results before running** — all · 20 Sep
    - Write the predicted outcome for A1, the clinical-parameter metric (A5) and B1 in the Progress Log (Experimental rule 4).
- [ ]  **3.6 A1 — Error decomposition** — Rameshkumar + Niranjana · 21–23 Sep · **stops after 3 days**
    - **How:** on matched scan-like and photo-like dev images, measure (a) trace overlap against ground truth and (b) the baseline's estimated grid scale against the true scale; regress SNR on each. ECG-Image-Kit gives lead boxes; PTB-XL-Image-17K ships pixel masks if boxes are not enough. Check whether the baseline exposes its scale estimate — if not, add code to log it.
    - **Decision:** calibration error dominates on photos → A2 and A3 go ahead. Segmentation dominates → log it as a finding (still publishable), skip A2 and A3, note Tier C for Phase II.
    - **→ Paper:** Results — error decomposition (H2); Discussion.
- [ ]  **3.7 Get the AHA/ACCF/HRS statement from the college library** — Risvanth · by 24 Sep
    - Reference 20 (Kligfield et al., 2007). Any normal range in the paper must come from it — no blogs.
    - **→ Paper:** Methods and Discussion, wherever clinical ranges appear.
- [ ]  **3.8 Shortlist venues with the guide** — Rameshkumar · by 24 Sep
    - **How:** check each candidate on the official Scopus source list; note deadline, page limit, template, APC and review time. Drop any venue whose deadline has passed, and any that promises guaranteed acceptance.
    - **Why now:** page limit and format decide how much of the results the paper can carry.

---

# Stage 4 — Clinical-parameter metric, the headline contribution (22 Sep – 1 Oct)

- [ ]  **4.1 Fix the delineation protocol in writing** — Risvanth · 22 Sep
    - Record the NeuroKit2 version, the delineation method, which leads, how PR, QRS and QT are computed from onsets and offsets, and where ST deviation is measured. Changing any of these later changes every number.
    - **→ Paper:** Methods (M6).
- [ ]  **4.2 Control run on ground truth** — Risvanth · 23–24 Sep
    - Delineate the clean ground-truth dev signals and record how often NeuroKit2 fails on them. Without this, delineator failures get blamed on digitization.
    - **→ Paper:** Methods; Limitations.
- [ ]  **4.3 Build the clinical-parameter error module** — Risvanth · 24–26 Sep
    - Error in HR (BPM), PR, QRS and QT (ms) and ST deviation (mV), plus delineation failure rate, per image type. Test: a signal against itself gives zero error.
    - **→ Paper:** Methods — the proposed metric (C1).
- [ ]  **4.4 Run it on the harness output** — Risvanth · 27–29 Sep
    - **Done when:** a table of clinical-parameter error per image type exists on the dev set.
    - **→ Paper:** Results — main table.
- [ ]  **4.5 SNR versus clinical error (H3)** — Risvanth · 29 Sep – 1 Oct
    - Choose the correlation measure with the guide and write it down **before** running. Scatter plot per parameter, coloured by image type.
    - **→ Paper:** Results — the headline figure.
- [ ]  **4.6 Run the second baseline through the harness** — Rameshkumar · by 1 Oct
    - **→ Paper:** Results — per-image-type benchmark (C2, Objective 1).

---

# Stage 5 — Robustness study (25 Sep – 4 Oct)

- [ ]  **5.1 B1 — Factorial robustness study** — Niranjana generates · Rameshkumar runs · Risvanth measures · 25 Sep – 3 Oct
    - **How:** vary **one** ECG-Image-Kit factor at a time, everything else fixed at clean settings: calibration pulse, grid presence, grid colour, resolution, creases, black-and-white, handwriting, layout. Record SNR and clinical-parameter error for each level.
    - **Done when:** the per-factor degradation table is complete on the dev set.
    - **→ Paper:** Results — per-factor table, which nobody has published.
- [ ]  **5.2 B5 — Grid colour** — Niranjana · by 3 Oct
    - All five grid colours. Supports the geography argument without printing anything.
    - **→ Paper:** Results; Discussion (geography).
- [ ]  **5.3 A3 quick check — grid spacing variation** — Rameshkumar · half a day · only if A1 found calibration dominates
    - Measure how much true grid spacing varies across a photo-like page. If negligible, skip A3.
- [ ]  **5.4 A2 — Calibration pulse versus grid scaling** — Rameshkumar + Niranjana · **stops after 5 days** · only if A1 supports it
    - First to move to Phase II if the calendar is tight — see the cut order above.

---

# Stage 6 — Lock results (2–5 Oct)

- [ ]  **6.1 Run on the frozen test set, once** — all · 2–3 Oct
    - Use the configuration fixed on dev. No tuning after seeing test numbers.
- [ ]  **6.2 Produce the final figures and tables** — Risvanth (analysis) + Niranjana (figures) · 3–4 Oct
    - Every figure states n, units and image type. Planned set below; numbering is provisional.
- [ ]  **6.3 Release the code** — Rameshkumar · 4–5 Oct
    - README with exact reproduction steps, pinned requirements, the harness and evaluation scripts, result CSVs. Link to PhysioNet for PTB-XL rather than copying the data. Respect each baseline's licence.
    - **→ Paper:** Code availability statement (C6).
- [ ]  **6.4 Compile the limitations list** — all · 5 Oct
    - From every "Went wrong" entry in the Progress Log, plus "What we are NOT claiming" on the Gaps page. At minimum: photo-like images are synthetic; no real Indian printouts yet; no clinician adjudication; NeuroKit2 delineation is the reference, not a cardiologist.
    - **→ Paper:** Limitations.

| Paper item | Content | Comes from |
| --- | --- | --- |
| Fig. 1 | System architecture | Review II §6 — already exists |
| Table 1 | Datasets | Review II §8 — already exists |
| Fig. 2 | SNR per image type, both baselines | 4.6 → 6.1 |
| Table 2 | Clinical-parameter error and delineation failure rate per image type | 4.4 → 6.1 |
| Fig. 3 | SNR versus clinical-parameter error | 4.5 → 6.1 |
| Table 3 | Per-factor degradation | 5.1, 5.2 |
| Fig. 4 | Error decomposition — segmentation versus calibration | 3.6 |

## 6.5 Paper gate — all six before Stage 7

From Experiment Plan §7. **Beating a baseline is not on the list** — the measurement and the mechanism are the contribution.

- [ ]  Per-image-type results for at least two baselines on a public benchmark
- [ ]  Digitization error reported in clinical units, with its relationship to SNR
- [ ]  A per-factor degradation table from the factorial study
- [ ]  At least one mechanism explaining why the failure occurs, supported by the error decomposition
- [ ]  Code and evaluation scripts released
- [ ]  Limitations stated honestly, including everything synthetic

---

# Stage 7 — Start writing the paper (from 5 Oct)

- [ ]  **7.1 Fix the target venue with the guide** — Rameshkumar · 5 Oct
    - Pick from the 3.8 shortlist. Re-check indexing on the official Scopus source list that day and write the date checked on this page.
    - **Done when:** venue, deadline, page limit and template are recorded.
- [ ]  **7.2 Confirm the paper's title and scope with the guide** — Rameshkumar · 5 Oct
    - The approved project title includes "Its Effect on Downstream Cardiac Classification", which is Phase II. The Phase I paper covers digitization and the clinical-parameter metric, so its title should match what it actually shows.
- [ ]  **7.3 Set up Overleaf with the venue's official template** — Rameshkumar · 5 Oct
    - Share it with all three members and the guide.
- [ ]  **7.4 Build the reference library in Zotero** — Niranjana · 5–6 Oct
    - Import all 20 survey references. **References 11–20 are marked ⚠️** in the Research Package — confirm each from full text through the library, or drop it. Cite software the way its documentation asks. Export BibTeX to Overleaf.
- [ ]  **7.5 Write the outline in Notion first** — all · 6 Oct
    - For each section: its paragraphs, one claim sentence each, and a link to the Notion page or result table that supports it. A claim with no link does not get written. Put the outline in the Paper draft section of Live Documentation.
- [ ]  **7.6 Write the sections in the order below** — owners in the table · 6–9 Oct, first pass
- [ ]  **7.7 Draft the required statements** — Rameshkumar · by 9 Oct
    - Author contributions (from 0.4); AI-use acknowledgement per the venue's policy and Student Guideline 9; data availability (PTB-XL, CC BY 4.0); code availability (6.3).
- [ ]  **7.8 Methods and Results drafted before Review III** — all · 9 Oct
    - **Done when:** both are complete first drafts in Overleaf. That is what "Publication Status" at Review III can honestly show.

| Order | Section | Written from | Proposed owner |
| --- | --- | --- | --- |
| 1 | Methods | Review II §5, §6, §8; protocol from 4.1; harness README | Niranjana (data, images) · Rameshkumar (pipeline) · Risvanth (metric) |
| 2 | Results | Stage 6 figures and tables only | Risvanth |
| 3 | Discussion | A1 decision; H1–H3 outcomes | Risvanth · Rameshkumar |
| 4 | Limitations | Step 6.4 list | Niranjana |
| 5 | Introduction | Research Package §2; Review II §1; contributions trimmed to what Results actually show | Rameshkumar |
| 6 | Related Work | Research Package §3 — the 20-paper survey | Niranjana |
| 7 | Conclusion | Results and Discussion | Rameshkumar |
| 8 | Abstract | Written last | All |

<aside>
💡

**Why this order:** Methods and Results are fixed by what was actually done, so they come first. The Introduction comes later so it promises only what Results deliver.

</aside>

---

# Stage 8 — Review III (10 & 12 Oct)

- [ ]  **8.1 Create the Review III content page and deck** — Rameshkumar · 6–8 Oct
    - Same approach as Review II: one Notion section per required item (13 items, Review Schedule page), then the Canva deck.
- [ ]  **8.2 Fill the evidence items from Notion** — all · 8 Oct
    - Preliminary Results ← Stage 6 · Challenges Faced & Solutions ← "Went wrong" entries · Work Completed vs Planned ← ticks on this page · Publication Status ← 7.1 and 7.8, plus a target submission date agreed with the guide.
- [ ]  **8.3 Rehearse, present, and log the remarks** — all · 9–12 Oct

---

# After this page

Full draft → guide review → institutional plagiarism check (Student Guideline 8) → submission to the verified venue (Oct–Nov target, per the Review II timeline) → Phase II: Tier C experiments, Indian printouts (Route A), three-arm classification.
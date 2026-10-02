# Live Documentation — Paper Draft & Progress Log

<aside>
🎯

**Document as the work happens, not at the end.** This page serves three requirements at once:

1. The **Project Progress Record / Log Book** required by Student Guideline 2
2. The evolving **paper draft**, so the Scopus submission is assembled continuously rather than written from memory in October
3. Review evidence — Review III asks for "Challenges Faced & Solutions" and "Work Completed vs. Planned Work", which are impossible to reconstruct later
</aside>

---

# 1. How to log

<aside>
📏

**One entry per working session.** Date it. Name who did it. Keep it short.

Record these five things:

- **Did** — what was actually done
- **Result** — numbers, even bad ones; "SNR 2.1 on photos" beats "tested the model"
- **Decided** — what was chosen **and why the alternative was rejected**
- **Went wrong** — errors, dead ends, sources that disagreed, things believed then corrected
- **Open** — what is still unresolved
</aside>

<aside>
⚠️

**The "went wrong" field is the one nobody can reconstruct later, and the one reviewers ask about.** A failed approach that is documented becomes "Challenges Faced & Solutions" at Review III and a Limitations paragraph in the paper. An undocumented one is just lost time.

</aside>

---

# 2. Paper draft — section status

| Section | Status | Source material |
| --- | --- | --- |
| Title & Abstract | Title approved; abstract waits for results | Hub page |
| 1. Introduction | **Draft exists** | Review I package, section 2 |
| 2. Related Work | **Draft exists** — 20 papers | Review I package, section 3 |
| 3. Background — ECG representation | **Draft exists** | Review I package, section B |
| 4. Methods | Outline only | Existing Methodology page |
| 5. Experiments & Results | **Empty — blocked on running the benchmark** | — |
| 6. Discussion | Empty | — |
| 7. Limitations | **Partly written already** | Gaps page, "What we are NOT claiming" |
| 8. Conclusion | Empty | — |
| References | 20 collected; **10 (nos. 11–20) need confirmation from full text** | Review I package |

<aside>
💡

Sections 1, 2, 3 and 7 are **already substantially written** as a by-product of the Review I research. Roughly half the paper exists before a single experiment has run. The remaining gap is Results — which is exactly what Review III demands anyway.

</aside>

---

# 3. Progress log

## 2026-09-07 — Project setup and Review I research

**Did**

- Connected the Notion workspace; confirmed the hub page and built the sub-page structure
- Confirmed the GitHub repository exists and is empty; local working directory empty and not yet a git repository
- Read both department PDFs and extracted the review schedule and rules
- Completed the Review I research package: 20-paper survey, five research gaps, objectives, timeline
- Researched the four add-on areas: industrial solutions, ECG parameters, publication scope, provable end results

**Result**

- Review I content complete against all 8 required items; Title and Problem Statement approved
- **10 of 20 citations verified directly at source; 10 still need confirmation**
- Confirmed from the official challenge score table: the winning digitization team scored **negative SNR on every mobile-phone-photo category** (−1.071 / −0.723 / −1.304) versus +4.930 on clean colour scans

**Decided**

- **Headline contribution is the clinical-parameter metric (Gap 1), not a new model.** Chosen because the challenge organisers explicitly identify the gap, it needs no GPU, and it is unlikely to be scooped. A new segmentation model was rejected as too slow to build and too easily beaten by better-resourced groups
- **Route A for Indian printouts** — print PTB-XL signals onto Indian ECG grid stock rather than collecting patient printouts. Chosen to remove the ethics-approval dependency from the critical path
- **Classification work deferred to Phase II.** Attempting it alongside the benchmark before Review II would produce three half-finished things
- **"Risk" reframed to "diagnostic classification"** — PTB-XL has diagnostic labels but no outcome or follow-up data, so genuine risk prediction is not possible from it. To be raised with the guide

**Went wrong / corrected**

- The seed brief cited **35,595 images** for ECG-Image-Database. Investigation showed this is the 2024 challenge-time figure (35,595 images / 1,977 records); the peer-reviewed 2026 version reports **37,191 images / 2,243 records**. Not an error in the brief — a stale figure. **Cite the journal version**
- Initially framed the gap as "phone-photo digitization is unsolved". **Corrected:** PMcardio is a CE-marked Class IIb device explicitly handling smartphone photos, and the organisers found it outperformed all challenge algorithms. The defensible claim is narrower — no *open, reproducible, independently evaluated* method reports per-image-type photo results
- A web search confidently attributed the **2024 CinC challenge winners to the 2025–26 Kaggle re-run**. This is wrong; the two events were conflated. **Do not cite that attribution anywhere**
- PTB-XL record counts disagree between sources: PhysioNet v1.0.3 states 21,799 records / 18,869 patients; the 2020 paper states 21,837 / 18,885. Records were removed between versions. **Use the PhysioNet figure — it matches what is actually downloaded**
- Clinical normal ranges for PR/QRS/QT were **deliberately omitted** — every accessible source was a blog or aggregator. Must be taken from the AHA/ACCF/HRS statement via the college library

**Open**

- Project Coordinator approval for the topic change — **unconfirmed, and this is the item that can formally block the project**
- 10 citations awaiting confirmation from full text
- Whether the Review I literature survey was formally presented for the ECG topic
- Open-ECG-Digitizer licence reads "Other" — the LICENSE file must be read before the code is used or redistributed
- No baseline has actually been run yet — this is the Review II critical path

---

<aside>
➕

Add the next entry directly below, newest at the bottom, using the same five headings.

</aside>

## 2026-09-14 — Review II preparation: content page and Canva deck

**Did**

- Re-read the whole workspace: hub plus all eight sub-pages
- Created [Review II — Presentation Content (17 Sep 2026)](Review%20II%20%E2%80%94%20Presentation%20Content%20(17%20Sep%202026)%203db2caadb48181f6899fe7de764822c9.md) — the 11 sections Review II requires, each tagged with the paper section it feeds, plus likely panel questions and a pre-review checklist
- Converted the Review I Canva copy into the Review II deck: all 15 slides' text replaced from that page, slides reordered into the department's required order and renumbered, file renamed "Batch No: PROJ_006 - Review 02"
- Corrected stale statuses on the hub and the Review Schedule page

**Result**

- Deck content is complete for 13 of 15 slides. Slide 11 (Initial Design / Prototype) and slide 12 (Individual Contribution) carry placeholders
- **No experimental numbers were produced** — none exist yet, so none were written

**Decided**

- Every figure on the slides comes only from workspace pages that already cite a primary source. Nothing was added from memory
- The four literature-survey slides were reused for architecture, tools, dataset and contribution, because Review II's required list does not include the survey. The survey table is kept as a backup slide
- Review II date recorded as **17 Sep 2026**, as given by Rameshkumar
- Slide 12 uses the proposed division of work from the Learning Path page, marked to be confirmed — it is not a team decision yet

**Went wrong / corrected**

- **The Review I deck is the welfare-scheme deck**, dated 18-08-2026. So the ECG problem statement, objectives, literature survey and research gap have never been reviewed. The hub's open question on this is now answered "No"
- The Review Schedule page still showed the title and problem statement as "awaiting approval", although both were approved on 7 Sep. Corrected
- **Date conflict, unresolved:** the department schedule (15.07.2026) lists Review II on 14–15 Sep; Rameshkumar gives 17 Sep
- Canva's find-and-replace carried bold formatting into the body text on slides 5 and 6. The API cannot restyle part of a text box, so this needs a manual reset

**Open**

- Project Coordinator approval for the topic change (General Guideline 3) — still the item that can formally block the project
- Baseline not run — slide 11 is empty until it is
- Individual contributions not agreed — slide 12
- Objectives still need the guide's approval
- Review II time slot unknown — slide 1 still shows Review I's 10.00–10.50 AM
- Slide 8 is a table; a drawn block diagram should replace or accompany it
- AI acknowledgement (Student Guideline 9): the Review II page, this entry and the slide text were AI-assisted. Each member must verify and understand what they present

## 2026-09-14 (later) — Block diagram rebuilt, deck finalised at 16 slides

**Did**

- Recorded that the Review I panel directed the topic change and gave this topic — hub, Rules page and Review II page updated
- Rebuilt the system block diagram as native Canva shapes on slide 8: data preparation → five-stage pipeline (Stage 4 highlighted) → evaluation (clinical-parameter error highlighted), with a colour legend
- Retitled the table slide "System Architecture — Module Details", placed it after the diagram, and renumbered slides 9–16
- Reworded timeline row 1 to state that the panel directed the change

**Result**

- 10 of the 11 required Review II items have finished slides. Item 9 (prototype) is placeholders only, and item 10 (contribution) needs the team's split

**Decided**

- Diagram built from editable shapes rather than an image, so it stays sharp and the team can adjust it
- The block diagram image that had been added was removed, not resized — it was squeezed to a 103 px strip and unreadable. It remains in Canva uploads

**Went wrong / corrected**

- The diagram slide added by hand carried a duplicate slide number "8"; fixed during renumbering
- A first formatting pass left the diagram's top-right box under the KPRIET logo; the box was narrowed

**Open**

- Guide review of the page and deck on 15 Sep, then record his approval on the hub
- Baseline run for slide 12; contribution split for slide 13; objectives approval; Review II time slot

## 2026-09-14 (evening) — Step-by-step roadmap to paper writing

**Did**

- Created [Roadmap — Steps to the Paper (14 Sep → 12 Oct 2026)](Roadmap%20%E2%80%94%20Steps%20to%20the%20Paper%20(14%20Sep%20%E2%86%92%2012%20Oct%202026%203db2caadb481819e954ef911bed199c2.md) — 9 stages and about 50 checkbox steps from today to starting the paper. Each step has an owner, a date, a "Done when" test and the paper section it feeds
- Re-read the Review II page, Experiment Plan, Learning Path, Review Schedule, Rules and Research Package, so every step traces to an existing decision
- Corrected stale figures on the Review Schedule, this page and the hub

**Result**

- Plan: approvals and prototype (14–16 Sep) → Review II (17 Sep) → foundation and A1 (18–24 Sep) → clinical-parameter metric (22 Sep – 1 Oct) → factorial study (25 Sep – 4 Oct) → lock results and the paper gate (2–5 Oct) → **start writing the paper from 5 Oct** → Review III (10 & 12 Oct)
- No experiments run; no numbers produced

**Decided**

- **Paper writing starts formally after the test-set run (Stage 6)**, so Results are never written from dev-set numbers. Methods notes may build up from Stage 4
- Cut order if behind: A2 and A3 to Phase II first, then B5. A1, the clinical-parameter metric, B1, the second baseline and the frozen test set are never cut — the paper gate needs them
- Section writing order: Methods → Results → Discussion → Limitations → Introduction → Related Work → Conclusion → Abstract
- Venue shortlisting moved early (by 24 Sep), because page limit and format shape the results write-up

**Went wrong / corrected**

- The paper-status table here and the Review Schedule said **6** references need confirmation. The Research Package marks **10** as ⚠️ (nos. 11–20). Corrected to 10
- The Review Schedule's status block was still dated 7 Sep ("7 days" to Review II). Updated to 14 Sep
- The hub said contributions were due "before 14 September". Changed to Review II on 17 Sep

**Open**

- Owners on the roadmap are the proposed Learning Path split until the team confirms it (step 0.4)
- The approved project title includes downstream classification (Phase II). The Phase I paper's title needs agreeing with the guide (step 7.2)
- AI acknowledgement: the roadmap was AI-assisted. Each member should read and agree to their own steps

## 2026-09-14 (night) — Guide briefing page in Tanglish

**Did**

- Created [Guide Briefing — Project Explained in Tanglish](Guide%20Briefing%20%E2%80%94%20Project%20Explained%20in%20Tanglish%203db2caadb481816eae2dd5f26c49eff4.md) — the whole project explained simply in Tanglish for the guide meeting on 15 Sep
- Covers: the problem, why phone photos break the grid, challenge evidence, the 5 gaps, contributions C1–C6, the 5-stage pipeline, datasets and tools, honest status, roadmap, questions to ask the guide, likely questions, what not to say, and a 1-minute script

**Result**

- No new facts or numbers. Everything is taken from the Review II, Research Package and Experiment Plan pages

**Decided**

- The QT example (10% scale error turns 400 ms into 440 ms) is labelled as an illustration, not a measured result
- "Questions to ask the guide" is a checklist, so his answers can be written on that page with the date

**Went wrong / corrected**

- Nothing

**Open**

- Record the guide's answers on the briefing page after 15 Sep, then tick the approval box on the hub

## 2026-09-15 — Guide feedback and problem-statement options

**Did**

- Recorded the guide's feedback: the ECG area is fine, but ECG digitization and classification are common and already done by many groups. He asked for a different problem statement within ECG
- Searched the literature to see which alternative directions are crowded, and checked the key sources and datasets at source
- Created [Problem Statement Options — After Guide Feedback (15 Sep 2026)](Problem%20Statement%20Options%20%E2%80%94%20After%20Guide%20Feedback%20(%203dc2caadb481812ea31aecaaef279561.md) — crowded directions, four options (A–D), a recommendation, and a draft title, problem statement and objectives for Option A
- Marked the hub's title and problem statement as on hold

**Result**

- Crowded, with recent papers: synthetic ECG generation, explainable AI, delineator benchmarks, ECG LLMs, uncertainty estimation, lead-misplacement detection, denoising, demographic bias, ECG re-identification
- Less crowded: **A.** clinical-decision disagreement between ECG measurement algorithms (PTB-XL+) · **B.** effect of electrode reversal on AI · **C.** adult tools and cut-offs on children (ZZU pECG) · **D.** auditing labels against measurements
- No experiments run; no numbers produced

**Decided**

- Recommended **Option A, with D as its second contribution**: neither digitization nor classification, open CC BY 4.0 data ready now, CPU only, and it keeps the clinical-parameter work. The guide still has to choose
- Every gap is written as "not found in this check", not "nobody has done it"

**Went wrong / corrected**

- Option A looked wide open at first. It is not: Kligfield et al. (Am Heart J 2014, 2018) and a 2020 seven-program study already compared commercial ECG programs' interval measurements. The narrower gap is per-ECG disagreement at clinical cut-offs, including open-source tools, on open data
- The PTB-XL+ paper already compares 12SL statements with the cardiologists' labels, which narrows Option D
- The ZZU pediatric dataset licence is CC BY-NC-ND 4.0, not CC BY — fine for analysis, but modified data cannot be redistributed

**Open**

- The guide's choice of option, then Project Coordinator approval (General Guideline 3), recorded in the Log Book
- What Review II on 17 Sep presents — the current deck and Roadmap still assume digitization
- Read Kligfield 2014 and 2018 and the 2020 study in full before claiming the gap
- AI acknowledgement: the search and options page were AI-assisted; sources marked "listing" must be opened before citing

## 2026-09-15 (later) — Guide clarifies: early prediction or software enhancement

**Did**

- Recorded the guide's clarification: he wants an **upgrade to the existing idea** — for example **early prediction** or a **software enhancement** — not another digitization or classification study
- Before choosing, reviewed how current ECG machines work end to end — electrodes and leads, amplification and filtering, digitization, measurement, computer interpretation, printing and storage, and live monitoring with alarms. Numbers checked against primary sources

**Result**

- No experiments, no decision

**Decided**

- Nothing final. The next round of options will be framed as early prediction or a software enhancement to existing ECG machines and software. Options A–D stay on file

**Went wrong / corrected**

- The 15 Sep options page answered "find a different problem". The guide's follow-up narrows it to "upgrade the existing idea", so its recommendation needs revisiting

**Open**

- Research early-prediction and software-enhancement ideas; guide's choice; Project Coordinator approval (General Guideline 3); what Review II on 17 Sep presents

## 2026-09-15 (evening) — How ECG machines work, and doable enhancements

**Did**

- Created [How ECG Machines Work Today & Doable Enhancements (15 Sep 2026)](How%20ECG%20Machines%20Work%20Today%20&%20Doable%20Enhancements%20%203dc2caadb48181cfb9bcd40b06836763.md), covering: the four kinds of ECG device, the step-by-step pipeline inside a resting 12-lead machine, how bedside monitors raise alarms, six evidenced weak points, five enhancements (E1–E5) and a recommendation
- Checked key facts at source: filter distortion of the ST segment, error rates of automatic interpretation, the 2015 false-alarm challenge, DELTAnet triage, the WARN AF early-warning model, the IRIDIA-AF dataset, and serial-ECG deep learning
- Downloaded PTB-XL v1.0.3 metadata and counted repeat ECGs and quality annotations

**Result**

- Automatic interpretation wrong in **39.5%** of 526 patients (Frontiers in Physiology 2025); computer missed **30%** of 340 confirmed STEMI (2016)
- A 0.5 Hz real-time filter produced clinically significant ST changes in **93%** of 45 patients (ISRN Cardiology 2012)
- PTB-XL (our count): **2,111 patients** have 2 or more ECGs (5,041 ECGs); **179** go from a NORM label to no NORM label; static noise in 3,260 records, baseline drift 1,598, burst noise 612, electrode problems 30; the first download attempt was incomplete (15,961 rows) and was discarded

**Decided**

- Recommended **E1: a "second look" safety layer** that predicts when the ECG machine's own interpretation is wrong, using PTB-XL cardiologist labels and PTB-XL+ 12SL statements and measurements. Phase II adds E2 (changed-since-last-ECG alert) and E5 (record-time quality assistant). The guide still has to choose
- AF early warning (E3) is real early prediction but already active (WARN 2024, 2025, 2026), so it needs a distinct angle; the ST-safe filter (E4) is an old field

**Went wrong / corrected**

- First PTB-XL metadata download was cut off mid-transfer; its statistics were discarded, and the counts above come from a complete 21,799-record file
- The AHA 2007 statement could not be opened (publisher blocked access); its filter quote is taken from the ISRN Cardiology 2012 paper that cites it — open the original before citing

**Open**

- Guide's choice among E1–E5 and A–D; Project Coordinator approval; what Review II on 17 Sep presents
- Confirm the 12SL-to-PTB-XL label mapping in PTB-XL+ before promising E1's first objective

## 2026-09-15 (night) — Message to guide with nine plan titles; India or general

**Did**

- Created [Plan Choice for Guide — 9 Titles & India Question (15 Sep 2026)](Plan%20Choice%20for%20Guide%20%E2%80%94%209%20Titles%20&%20India%20Question%20%203dc2caadb481812fac43f7ac0e26af14.md). It holds a ready-to-send message listing all nine plans with titles, the evidence on India-specific versus general, and a checklist for the guide's decision
- Checked India facts at source: cardiovascular burden, Tamil Nadu STEMI programme, tele-ECG services, public datasets, and data-protection rules

**Result**

- Cardiovascular disease caused 28.1% of deaths in India in 2016, up from 15.2% in 1990; Tamil Nadu and Kerala are among the highest (Lancet Global Health 2018)
- The Tamil Nadu STEMI programme (35 spokes, 4 hubs, 2,420 patients) raised primary PCI from 29.5% to 46.5% and cut 1-year mortality from 17.6% to 14.2% (JAMA Cardiology 2017)
- No public Indian 12-lead ECG dataset found

**Decided**

- Recommended to the team: **keep the Phase I core general on open datasets**; use India in the motivation and add Indian clinic validation in Phase II with ethics approval and consent
- The message recommends plan 1, Second-Look ECG, and lets the guide pick

**Went wrong / corrected**

- A search summary linked "74.3% of primary-care physicians have poor ECG knowledge" to this question. The study is from **Saudi Arabia**, so it must not be quoted for India
- Tricog's figures come from its own website and are company claims, labelled as such

**Open**

- Guide's pick, India-or-general answer, and what Review II on 17 Sep presents; Project Coordinator approval

## 2026-09-16 — Direction fixed: Second-Look ECG; Review II deck rebuilt *(written late — the Notion connector was down on 16 Sep)*

**Did**

- Rameshkumar chose the upgrade direction: **E1 Second-Look ECG** as the Phase I core, with **E2** (change-since-last-ECG alert) and **E5** (recording quality and misplaced-electrode check) as the other modules
- Rebuilt the full Review II deck for that direction — 16 slides in the Canva order — as an HTML artifact, plus a paste-ready text file, because the Canva connector was disconnected

**Result**

- Slide 12 carried real counts from the PTB-XL metadata: 21,799 records / 18,869 patients; 6,813 machine-generated first reports; 2,111 patients with 2 or more ECGs; 179 normal-to-abnormal patients; noise annotations (static noise 3,260, baseline drift 1,598, burst noise 612, electrode problems 30)
- No model trained

**Went wrong / corrected**

- Canva and Notion connectors dropped out; the deck could not be written to Canva

**Open**

- Superseded on 19 Sep — see the next entry

## 2026-09-19 — Problem statement changed to prenatal CHD screening; claims checked; novelty check started

**Did**

- Received unstructured research from another chat proposing **prenatal congenital heart disease (CHD) screening** from fetal echocardiography, with non-invasive fetal ECG as a complementary signal. The guide asked for a novelty check with Connected Papers
- Re-checked every claim at source and created [Novelty Check — Connected Papers & Literature Map](../Novelty%20Check%20%E2%80%94%20Connected%20Papers%20&%20Literature%20Map%203e02caadb48181e2b032df09533c684f.md), covering the checked claims, a crowded-versus-open literature map, a recommended project, a dataset plan, the fetal ECG position and a Connected Papers protocol with seven seed papers
- Ran literature searches on OpenAlex for eight angles

**Result**

- The other chat's figures held up: de Vries 2023 (fetal ECG, 63% sensitivity, not public data); the 2025 meta-analysis (15 studies, sensitivity 0.89, specificity 0.91); CARDIUM (6,558 images, 16.3% CHD); Heartbeat (1,475 patients, 6.50% and 7.25% CHD); NIFECGDB (55 recordings, one subject)
- **Crowded:** CHD classification from fetal echo (51 papers, 36 since 2023; Arnaout, Nature Medicine 2021, AUC 0.99) and image-plus-clinical fusion (done by CARDIUM and Heartbeat)
- **Open in this check:** transfer between trimesters, calibration, and referral with a guaranteed sensitivity (conformal); Heartbeat's own paper lists uncertainty, calibration and cross-dataset validation as not investigated
- **Blocked:** fetal ECG for CHD — no public CHD-labelled fetal ECG; NInFEA pairs fetal ECG with Doppler but has healthy fetuses only

**Decided**

- Recommended **"Know When to Refer"**: a calibrated, shift-aware CHD screening model whose output is a referral decision with a target sensitivity — the second-look idea carried into the new domain. Fetal ECG moves to a Phase II extension that needs paired labelled data
- Every gap is written as "not found in this check"

**Went wrong / corrected**

- I first told Rameshkumar the other chat's Heartbeat numbers were wrong, based on the GitHub split counts (1,474 patients, 44 of 723 CHD). The paper confirms 1,475 patients and 6.50% / 7.25% — the other chat was right; cite the paper
- Semantic Scholar rate-limited most queries; switched to OpenAlex. The Connected Papers about page could not be read, so its data source is not verified
- Canva needs re-authorisation, so the deck could not be updated
- CARDIUM and Heartbeat come from the same lab (Universidad de los Andes); patient overlap is unknown, so they cannot yet be called external to each other

**Open**

- Guide approval of the angle; Project Coordinator approval (General Guideline 3)
- Submit Heartbeat and CARDIUM access requests — the critical path
- Run Connected Papers on the seven seeds
- Record what happened at Review II on 17 Sep
- AI acknowledgement: the checking, search and page were AI-assisted; each member must read the sources behind their part
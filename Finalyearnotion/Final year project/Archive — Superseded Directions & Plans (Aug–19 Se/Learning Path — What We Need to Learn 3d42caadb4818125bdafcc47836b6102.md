# Learning Path — What We Need to Learn

<aside>
🎯

Purpose: the shortest path from where the team is now to being able to **build the system and defend it under questioning**. Student Guideline 6 requires each member to demonstrate understanding of the methodology and their own contribution at every review — so learning is not optional preparation, it is graded.

Each module says **why** it is needed, tied to a specific pipeline stage or objective. Nothing is listed for completeness alone.

</aside>

---

# 0. What we do NOT need to learn

<aside>
✂️

Scope control matters more than coverage. **Skip all of these:**

- Training large models from scratch — existing baselines are used as-is
- Transformers / attention theory — not on the critical path
- Full clinical cardiology — only the parameters in the ECG Parameters table are needed
- Web or app development — nothing in the objectives requires a UI
- Cloud/MLOps deployment — Colab is sufficient
- C++ or CUDA programming

If something is not traceable to an objective, it is a distraction from a 7-day and 33-day deadline.

</aside>

---

# 1. Learning modules, in dependency order

## Tier 1 — Needed before anything else can run

| Module | Why it is needed | Depth required | Est. |
| --- | --- | --- | --- |
| **Python: NumPy + file I/O** | Every stage manipulates arrays; signals are arrays | Working fluency | Assumed known |
| **Google Colab + GPU runtime** | All compute happens here; understanding session limits and drive mounting prevents lost work | Practical | Half day |
| **Git and GitHub** | Three people on one codebase; also Gap 4 (reproducibility) and the required Log Book | clone, branch, commit, push, PR | 1 day |
| **wfdb-python + PTB-XL structure** | Cannot load the ground-truth signals without it | Read records, access metadata and labels | 1 day |
| **ECG fundamentals** | Cannot judge whether output is right without knowing what an ECG should look like | 12 leads, P-QRS-T, the grid conventions, layouts (3x4, 6x2, 12x1) | 1–2 days |

## Tier 2 — Needed for the core contribution (Objectives 1 and 2)

| Module | Why it is needed | Depth required | Est. |
| --- | --- | --- | --- |
| **NeuroKit2 ECG delineation** | **The single most important tool in the project.** Extracts P/QRS/T onsets and offsets, which give PR, QRS and QT — the whole of Objective 2 | Deep — including when it fails and why | 3–4 days |
| **SNR and signal comparison** | The benchmark metric; must be reimplemented to match the challenge definition, including the alignment shift | Able to implement from the definition | 2 days |
| **Resampling and alignment** | Digitized output and ground truth differ in length and offset; comparison is meaningless without alignment | Cross-correlation alignment, interpolation | 2 days |
| **Experimental design** | Patient-independent splits are a known failure point that reviewers check | Group splits, no patient overlap, held-out test | 1 day |

## Tier 3 — Needed to run and improve the baselines (Objectives 1 and 3)

| Module | Why it is needed | Depth required | Est. |
| --- | --- | --- | --- |
| **OpenCV image basics** | Every geometry stage is OpenCV | Load, colour spaces, threshold, morphology | 2 days |
| **Hough transform + homography** | **The heart of Stage 2 and Stage 4** — line detection recovers the grid, homography undoes camera perspective. Objective 3 lives entirely here | Deep enough to modify, not just call | 4–5 days |
| **Autocorrelation for grid spacing** | How Stage 4 recovers mm/s and mm/mV | Conceptual + implementable | 2 days |
| **PyTorch inference** | Running the pretrained baselines. **Inference only — not training** | Load checkpoint, run, read output | 2 days |
| **U-Net segmentation concepts** | Stage 1 in every published method; must be explainable at a review | Conceptual; implementation only if Objective 3 needs retraining | 3 days |

## Tier 4 — Phase II (Objective 5)

| Module | Why | Est. |
| --- | --- | --- |
| CNN image classification (transfer learning) | The direct-image arm of H4 | 4 days |
| 1-D CNN for signals | The digitize-then-classify arm | 4 days |
| Multi-label metrics (macro F) | ECGs carry multiple simultaneous labels | 1 day |

## Tier 5 — Publication skills (needed by all three)

| Module | Why | Est. |
| --- | --- | --- |
| **LaTeX / Overleaf** | Essentially every target venue requires it | 2 days |
| **Reference management** (Zotero or Mendeley) | 20+ references; manual BibTeX invites citation errors | Half day |
| **Reading a paper critically** | The literature survey must state each paper's *limitation*, not just its claim | Ongoing |
| **Plagiarism and citation rules** | Student Guideline 8; violations are treated seriously | Half day |
| **Scientific figures** | The per-image-type SNR chart is the paper's central figure | 2 days |

---

# 2. Suggested split across the team

<aside>
👥

This is a **proposal**, not a decision — adjust to actual strengths. Student Guideline 5 requires each member to have a clearly defined individual contribution, and Review II asks for it explicitly.

Everyone does Tier 1 and Tier 5. Every member must be able to explain the **whole** pipeline, not only their own part, because reviews question individuals.

</aside>

| Member | Owns | Priority learning | Deliverable |
| --- | --- | --- | --- |
| **Rameshkumar** | Pipeline integration — getting baselines running, benchmark harness | Tier 3: PyTorch inference, OpenCV, repo wrangling | Reproducible benchmark script producing per-image-type SNR |
| **Niranjana** | Data and image degradation — ECG-Image-Kit, photo simulation, dataset construction | Tier 1 wfdb/PTB-XL, Tier 3 OpenCV and geometry | Image sets per category with documented generation parameters |
| **Risvanth** | Clinical metric and evaluation — **the headline contribution** | Tier 2: NeuroKit2 delineation, SNR, alignment | Clinical-parameter error module and the SNR-versus-clinical-error analysis |

---

# 3. Seven-day priority for Review II

<aside>
⏰

**Review II is 14/15 September.** Only what is needed for a working prototype and a block diagram matters this week. Everything else waits.

</aside>

- [ ]  **Day 1–2** — Colab set up; PTB-XL downloaded; a record loaded and plotted with wfdb
- [ ]  **Day 2–3** — Clone a baseline digitizer, get it to run on **one** image end to end. *Expect dependency problems; this is the risky step*
- [ ]  **Day 3–4** — Generate scan-like and photo-like images of the same records with ECG-Image-Kit
- [ ]  **Day 4–5** — Run the baseline on both categories; compute SNR; produce the comparison chart
- [ ]  **Day 5–6** — Block diagram of the 5-stage pipeline marking the stage being targeted
- [ ]  **Day 6–7** — Slides; individual contributions written up; timeline

<aside>
💡

If time runs short, **the one thing that must exist is the scan-versus-photo SNR chart on real output from a real baseline.** One honest chart showing the failure is worth more at Review II than a polished architecture diagram of something not yet built.

</aside>
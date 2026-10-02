# Final year project

<aside>
👶

**Current project (from 19 Sep 2026): prenatal heart screening from three inputs — fetal ECG, Doppler ultrasound and fetal echocardiography.** The title, problem statement and methodology below are **drafts** — they need the guide's approval and **prior Project Coordinator approval** (General Guideline 3). The reasoning is on the Methodology Research page; the Tanglish page explains the final method step by step. Earlier directions are kept on the Archive page for the Log Book record.

</aside>

<aside>
⏰

**Next: Review III — 10 & 12 October 2026.** It needs preliminary results and publication status. The fetal ECG and Doppler parts run on **open data now**; the echo part waits for the Heartbeat and CARDIUM access requests (Roadmap, stage 0). Review II was held on 17 Sep; the panel's four points are answered on the How It Works and Novelty Check pages.

</aside>

---

## ▶️ What to do now

<aside>
🔁

**How this list works.** Only current tasks live here. When one is done: tick it, write one line in the Progress Log, **delete it from this list**, and move the top "Up next" item in. Last updated **19 Sep 2026**.

</aside>

- [ ]  **Show the final method to the guide and get approval** — use the Tanglish page and the Methodology Research page · Rameshkumar · 20 Sep
- [ ]  **Project Coordinator approval** for the change of title, scope and method (General Guideline 3); note it in the Log Book · Rameshkumar · by 22 Sep
- [ ]  **Submit the Heartbeat access form** — [forms.gle/QTTn1S7kKxkepVB18](https://forms.gle/QTTn1S7kKxkepVB18), with the guide as supervisor · Niranjana · 20 Sep
- [ ]  **Submit the CARDIUM access form** — [CARDIUM request form](https://docs.google.com/forms/d/e/1FAIpQLSeW3EKB54HnfSVmyOf8yD9swB7DUiaEr0_pSMl6z_zTwdJR-Q/viewform) · Niranjana · 20 Sep
- [ ]  **Ask the dataset authors** ([s.rodriguezr2@uniandes.edu.co](mailto:s.rodriguezr2@uniandes.edu.co)) whether Heartbeat and CARDIUM share patients — the numbers suggest they do · Niranjana · 20 Sep
- [ ]  **Agree individual contributions** (proposed split below) · all · 20 Sep
- [ ]  **Set up GitHub** — clone the empty repo [github.com/rameshkumark24/FinalyearProject](https://github.com/rameshkumark24/FinalyearProject); add a README and a `.gitignore` that keeps **all data and model weights out** (dataset terms forbid redistribution); consider making it private · Rameshkumar · 21 Sep
- [ ]  **Download the open data** — NInFEA, NIFEADB, CinC 2013 set A, CARDIUM clinical JSON; a data card for each; commit the split files (runs D1, D2, D5) · Niranjana · 24 Sep

### Up next

1. Maternal ECG removal and fetal QRS detection (R1–R2) — Niranjana
2. Doppler envelopes and cycle detection (F1–F2) — Risvanth
3. Load FetalCLIP and the ECG foundation models; read their licences — Rameshkumar
4. Connected Papers: 10 seed graphs and the comparison table — all
5. When echo access arrives: score the released Heart-ViT weights (S2, S8) — Rameshkumar

### Also pending

- Record which deck version was presented at Review II (Log Book)
- Re-authorise Canva before building the Review III deck
- Read the 🟡 method papers before citing them
- Image-hash overlap check (D4) once both echo datasets arrive

---

## Course & Team

| Course | U21AM704 — Project Work Phase I |
| --- | --- |
| Department | CSE (Artificial Intelligence and Machine Learning) |
| Institution | KPR Institute of Engineering and Technology |

| Team member | Individual contribution — proposed, confirm as a team |
| --- | --- |
| **Rameshkumar K** (Team Leader) | Structure expert (echo, FetalCLIP) and the fusion layer |
| **Niranjana J** | Rhythm expert (fetal ECG extraction, foundation-model transfer) and data management |
| **Risvanth V** | Function expert (Doppler), cross-modal trust check, calibration |

---

## Project title (draft)

> **Structure, Function and Rhythm: Uncertainty-Aware Fusion of Fetal Echocardiography, Doppler Ultrasound and Non-Invasive Fetal ECG for Prenatal Heart Screening Without Paired Data**
> 

---

## Problem statement (draft)

Congenital heart disease (CHD) is the most common birth defect; India has more than 200,000 CHD births a year (Saxena, *Indian Pediatrics* 2018). Screening ultrasound could detect 90% of complex CHD, but in practice sensitivity is as low as 30% (Arnaout et al., *Nature Medicine* 2021). A full fetal heart assessment covers **anatomy, function and rhythm** (AHA scientific statement, 2014), which three inputs capture: **echocardiography** (structure), **Doppler ultrasound** (flow and timing) and **non-invasive fetal ECG** (electrical rhythm). AI work treats each input alone — CHD from echo images, arrhythmia from fetal ECG, timing from Doppler — and a 2026 systematic review found no study that combines them. The obstacle is data: no public dataset records all three for the same fetus, and only echo has public CHD labels. In practice, too, a pregnant woman rarely has all three tests, because they sit at different levels of care.

This project builds a prenatal heart-screening framework whose three expert models are trained **separately on real public data**, whose fusion **needs no paired patients**, **handles any missing test**, and **does not let one normal result cancel another's alarm**. Where fetal ECG and Doppler overlap, their agreement is used as a trust check. The output is **refer**, **routine**, or **get the missing test**.

---

## Problem statement — simple version

<aside>
🧒

Explained as if to a five-year-old.

</aside>

Before a baby is born, doctors can check the baby's heart in three ways: a picture of the heart (echo), a sound-wave check of the blood flowing through it (Doppler), and tiny heart-electricity signals picked up from the mother's tummy (fetal ECG). Each check sees a different kind of problem.

We build three computer helpers — one for each check — and a referee. The referee also checks that the heartbeat heard by the Doppler matches the heartbeat felt by the ECG.

If any helper sees a problem, the referee says *"please see the heart doctor"*. If a check was not done, it says *"please do this check too"* instead of guessing.

---

## Objectives (draft)

1. Build a **rhythm expert** on fetal ECG and test whether **adult ECG foundation models transfer** to fetal signals, at subject level (NIFEADB)
2. Build a **function expert** on Doppler ultrasound — cardiac-cycle timing and a healthy range for weeks 21–27 (NInFEA)
3. Build a **cross-modal trust check** from synchronised fetal ECG and Doppler, and measure how much it cuts false alarms (NInFEA)
4. Build a **structure expert** on fetal echocardiography with a fetal-ultrasound foundation model, evaluated at patient level and across trimesters (Heartbeat; CARDIUM if granted)
5. Design an **uncertainty-aware OR-fusion** that needs no paired patients and handles any missing input, and compare it with standard fusion rules
6. Release code, per-subject results and a data card for each dataset

---

## Paper idea

> **Claim.** Three fetal heart inputs can be combined for screening without a single patient who had all three tests: experts trained separately, a trust check where the inputs overlap, and an OR-fusion that keeps alarms and says what it does not know.
> 
- **Contributions (each not found in our check):** the first framework combining fetal ECG, Doppler and echo · a fusion that needs no paired patients and handles missing inputs · the first test of adult ECG foundation models on fetal ECG · fetal ECG–Doppler agreement as a trust check · the first evaluation of a fetal-ultrasound foundation model on public CHD data across trimesters · released code
- **Not claimed:** diagnosis · replacing fetal cardiologists · detecting structural CHD from fetal ECG or Doppler · validation of the fused system on real patients with all three tests · Indian validity without Indian data
- **Venue:** to be chosen with the guide; indexing verified on the official Scopus source list
- **Phase II:** paired hospital data from the same fetus — ethics approval, consent, a PCPNDT-registered facility — to validate the fusion on real patients

---

## Pages

[Review Schedule & What Each Review Expects](Final%20year%20project/Review%20Schedule%20&%20What%20Each%20Review%20Expects%203d42caadb4818196859ee2d273d32058.md)

[Rules & Regulations — Phase I](Final%20year%20project/Rules%20&%20Regulations%20%E2%80%94%20Phase%20I%203d42caadb48181b9bfc8f0a092b1b677.md)

[Research Package — Prenatal Heart Screening (Echo + Doppler + Fetal ECG)](Final%20year%20project/Research%20Package%20%E2%80%94%20Prenatal%20Heart%20Screening%20(Echo%20%203e02caadb48181cbbd47ea1185f8d04a.md)

[How Prenatal Heart Screening & AI Work Today](Final%20year%20project/How%20Prenatal%20Heart%20Screening%20&%20AI%20Work%20Today%203e02caadb481814892a0fa4b02e83721.md)

[Research Gaps → Can We Solve Them?](Final%20year%20project/Research%20Gaps%20%E2%86%92%20Can%20We%20Solve%20Them%203e02caadb48181f4bbf3c32be36ffb0d.md)

[Novelty Check — Connected Papers & Literature Map](Final%20year%20project/Novelty%20Check%20%E2%80%94%20Connected%20Papers%20&%20Literature%20Map%203e02caadb48181e2b032df09533c684f.md)

[Experiment Plan — What to Run, in Order](Final%20year%20project/Experiment%20Plan%20%E2%80%94%20What%20to%20Run,%20in%20Order%203e02caadb481817d946adf226f3d76c8.md)

[Learning Path — What We Need to Learn](Final%20year%20project/Learning%20Path%20%E2%80%94%20What%20We%20Need%20to%20Learn%203e02caadb48181d7a5fbe5bc02752e47.md)

[Roadmap — Steps to Review III & the Paper (19 Sep → 12 Oct 2026)](Final%20year%20project/Roadmap%20%E2%80%94%20Steps%20to%20Review%20III%20&%20the%20Paper%20(19%20Sep%20%203e02caadb48181a2b4e6e579c36b876a.md)

[Live Documentation — Paper Draft & Progress Log](Final%20year%20project/Live%20Documentation%20%E2%80%94%20Paper%20Draft%20&%20Progress%20Log%203e02caadb481814da6c7c3892b05294e.md)

[Guide Briefing — Our Final Method in Tanglish (Structure–Function–Rhythm)](Final%20year%20project/Guide%20Briefing%20%E2%80%94%20Our%20Final%20Method%20in%20Tanglish%20(Str%203e02caadb4818128a165f424512e14d9.md)

[Archive — Superseded Directions & Plans (Aug–19 Sep 2026)](Final%20year%20project/Archive%20%E2%80%94%20Superseded%20Directions%20&%20Plans%20(Aug%E2%80%9319%20Se%203e02caadb4818191aa5dcc320f216353.md)

[Methodology Research — ECG + Doppler Ultrasound + Echo: Best Method & Trial Runs](Final%20year%20project/Methodology%20Research%20%E2%80%94%20ECG%20+%20Doppler%20Ultrasound%20+%20%203e02caadb4818121a17fe2e76c30afc9.md)
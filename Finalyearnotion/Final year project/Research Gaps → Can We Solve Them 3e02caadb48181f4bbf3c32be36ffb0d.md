# Research Gaps → Can We Solve Them?

<aside>
🎯

**Purpose.** Each gap is tied to evidence checked at source on 19 Sep 2026, and judged honestly: can a three-person team solve it in Phase I with public data? A gap means **not found in our check**, never "nobody has done it". Updated for the three-input methodology (fetal ECG, Doppler ultrasound, echocardiography).

</aside>

---

# Gap table

| Gap | Evidence | Phase I? | How we address it |
| --- | --- | --- | --- |
| **G1 — No system combines fetal ECG, Doppler and echo** | 2026 systematic review of AI fetal-ECG methods: no multimodal fusion studies. Europe PMC: 0–2 matches for any combination | ✅ Yes | The Structure–Function–Rhythm framework |
| **G2 — No paired data, so fusion must work without paired patients** — *core methodological gap* | No public dataset has all three inputs for one fetus. Missing-input learning in fetal cardiology: 0 matches on Europe PMC and arXiv | ✅ Yes | Experts trained separately; training-free OR-fusion; a missing input = full uncertainty (runs X1–X3) |
| **G3 — Consensus fusion can cancel an alarm** | Trusted multi-view methods (Han, TPAMI 2022; Zhou 2026 for fetal echo) use Dempster's rule, built for views of the same thing. Fetal ECG, Doppler and echo detect different conditions | ✅ Yes | Subjective-logic OR-fusion, compared against Dempster in X1 |
| **G4 — Adult ECG foundation models are untested on fetal ECG** | 0 matches on Europe PMC and arXiv; the 2026 review does not mention foundation models. Open weights exist (ECGFounder, ECG-FM, HuBERT-ECG) | ✅ Yes | Runs R5–R6, subject-level on NIFEADB |
| **G5 — Fetal ECG–Doppler agreement is not used as a trust check** | None of the 30 papers citing NInFEA does this. Maternal–fetal heart-rate confusion is documented (Stampalija 2012). NInFEA work so far translates ECG into Doppler (2025, 2026) | ✅ Yes | Runs C1–C3 on NInFEA |
| **G6 — Fetal-ultrasound foundation model untested on public CHD data or across trimesters** | FetalCLIP's CHD test used 418 internal four-chamber videos (AUROC 78.72%, linear probe). Heartbeat reports each trimester separately | 🟡 On access | Runs S3–S8 on Heartbeat |
| **G7 — Confidence is rarely calibrated** | Heartbeat: calibration and uncertainty not investigated. 2026 review: most fetal-ultrasound classifiers "remain opaque and miscalibrated" | ✅ Yes | Every expert calibrated before fusion (R7, S9) |
| **G8 — CHD labels exist only for echo** | No public CHD-labelled fetal ECG or Doppler. de Vries 2023 (fetal ECG, sensitivity 63%) used private data | ❌ Phase II | Paired hospital data, with ethics approval and a PCPNDT-registered facility |

<aside>
🔎

**A check, not a gap claim.** Many fetal-arrhythmia papers on NIFEADB (26 subjects) report accuracies above 95%. We have not checked each paper's split, so we do not claim leakage. Run R4 simply shows how much a segment-level split inflates results on this dataset.

</aside>

---

# Positioning sentence

> **Fetal ECG, Doppler and echocardiography each see a different part of the fetal heart, but no public dataset records all three for one fetus. We build a screening framework whose experts train separately on real data, whose fusion needs no paired patients and handles missing tests, and which never lets one normal test cancel another's alarm.**
> 

---

# What we are NOT claiming

<aside>
🚫

- That the system diagnoses CHD or replaces a fetal cardiologist — it supports screening and referral
- That fetal ECG or Doppler detect structural CHD
- That the fused system is validated on real patients who had all three tests — no such public data exist
- That the Doppler expert detects disease — it only flags values outside a healthy range
- That results apply to Indian hospitals without Indian data
- That Heartbeat and CARDIUM are independent — not until patient overlap is checked
</aside>
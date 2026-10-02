# Experiment Plan — "Know When to Refer" (echo only, superseded 19 Sep 2026)

<aside>
🗄️

**Superseded on 19 Sep 2026 (evening)** when the team decided to use three inputs — fetal ECG, Doppler ultrasound and echocardiography. Kept for the Log Book (Student Guideline 2). The current plan is on the active Experiment Plan page. Calibration, trimester shift, the conformal referral rule and missing-view handling carry over as parts of the new plan (runs S6, S8, S9, X4).

</aside>

---

# Rules

1. **Patient-level splits** — no fetus appears in two splits.
2. **Freeze the test set** and touch it once.
3. **One change per run.**
4. **Write the expected result first.**
5. **Log every run in the Progress Log**, including failures.
6. **Report 95% bootstrap confidence intervals** — test sets are small (Heartbeat's 2nd-trimester test has 61 patients).

---

# Tier A — before Review III

## A0. Setup and baseline reproduction

- **Do:** run the released Heart-ViT code on Heartbeat with patient-level splits.
- **Done when:** patient-level sensitivity, specificity, AUROC and AUPRC are reported with CIs, alongside the paper's figures.
- **Stop after 3 days** if the code will not run — switch to a pretrained ViT or ResNet baseline and record why.

## A1. Calibration

- **Do:** reliability diagram, expected calibration error and Brier score, before and after temperature scaling on the validation split.
- **Expected (H2):** calibration improves in-distribution.

## A2. Trimester shift

- **Do:** train on 2nd trimester and test on 3rd; train on 3rd and test on 2nd; compare with in-trimester results.
- **Expected (H1):** sensitivity and AUROC drop, and calibration worsens. Either outcome is publishable.

## A3. Conformal referral — the main contribution

- **Do:** set the decision threshold with conformal risk control for a target patient-level sensitivity (95%, and 90% as a second setting). Report achieved sensitivity and the share of pregnancies referred, in-distribution and under trimester shift.
- **Expected (H3):** the guarantee holds in-distribution and weakens under shift unless the threshold is re-set with a few target-trimester patients — test that too.

## A4. Poor or missing views

- **Do:** remove each view at inference; weight views by confidence in the patient decision.
- **Expected (H4):** fewer missed CHD at the same referral rate.

---

# Tier B — if time allows

- **B1. Cohort shift Heartbeat ↔ CARDIUM** — only after the authors confirm no shared patients.
- **B2. Explanation check on FOCUS** — share of attention inside the annotated heart.
- **B3. Imbalance strategies** — weighted loss vs focal loss vs weighted sampling.

---

# Tier C — Phase II

- **C1.** Fetal ECG groundwork on NInFEA — fetal heart rate against the synchronised Doppler.
- **C2.** Paired fetal ECG + echo data through a hospital collaboration, with ethics approval — then echo-only vs ECG-only vs combined.
- **C3.** Validation on Indian hospital scans, with ethics committee approval and consent.

---

# Paper gate — all before writing Results

- [ ]  Patient-level baseline with confidence intervals
- [ ]  Calibration before and after
- [ ]  Trimester-shift results
- [ ]  Referral rule: achieved vs target sensitivity, in-distribution and under shift
- [ ]  Limitations stated: referral-centre prevalence, same-lab datasets, no Indian data, no fetal ECG
- [ ]  Code released
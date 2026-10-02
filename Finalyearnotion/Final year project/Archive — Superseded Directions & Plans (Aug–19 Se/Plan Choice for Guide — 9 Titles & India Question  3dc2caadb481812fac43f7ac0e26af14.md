# Plan Choice for Guide — 9 Titles & India Question (15 Sep 2026)

<aside>
📬

**What this page is.** The message to the guide listing every plan with its title so he can pick one, the evidence on whether to make the project India-specific or keep it general, and a place to record his decision.

Details behind the plans: [How ECG Machines Work Today & Doable Enhancements (15 Sep 2026)](How%20ECG%20Machines%20Work%20Today%20&%20Doable%20Enhancements%20%203dc2caadb48181cfb9bcd40b06836763.md) (E1–E5) and [Problem Statement Options — After Guide Feedback (15 Sep 2026)](Problem%20Statement%20Options%20%E2%80%94%20After%20Guide%20Feedback%20(%203dc2caadb481812ea31aecaaef279561.md) (A–D).

</aside>

---

# 1. Message to the guide

> Good evening Sir,
> 

> 
> 

> As you suggested, we looked for an upgrade to our ECG idea — early prediction or a software enhancement — instead of plain digitization or classification.
> 

> 
> 

> **What we have added so far:** the Review II content and slides, a step-by-step plan up to paper writing, a study of how current ECG machines work and where they fail, and a check of recent papers so we do not pick something already done. For example, a 2025 study found the ECG machine's automatic interpretation was wrong in 39.5% of 526 patients.
> 

> 
> 

> **Kindly pick one plan:**
> 

> 
> 

> *Software enhancement / early prediction*
> 

> 1. ⭐ **Second-Look ECG: Predicting When the ECG Machine's Automatic Report Is Wrong** — software that flags which automatic reports the doctor should check first. Open data (PTB-XL, PTB-XL+). We did not find this in published work, and it is doable in Phase I. *(Our recommendation)*
> 

> 2. **ECG Change Alert: Early Detection of New Abnormalities by Comparing With the Patient's Previous ECG**
> 

> 3. **AF Early Warning: Predicting Atrial Fibrillation Minutes Before Onset From Holter or Wearable ECG** — some recent papers exist
> 

> 4. **Smart Recording Assistant: Real-Time Check for Noise and Misplaced Electrodes Before the ECG Is Printed**
> 

> 5. **ST-Safe Live Monitoring: Real-Time Filtering That Keeps the ST Segment Accurate** — older research area
> 

> 
> 

> *Other ECG options*
> 

> 6. **Same ECG, Different Verdict: When Different ECG Software Gives Different Clinical Decisions**
> 

> 7. **Swapped Electrodes, Silent Errors: How Lead Reversal Misleads ECG AI**
> 

> 8. **Children Are Not Small Adults: Adult ECG Software on Children's ECGs**
> 

> 9. **Checking the Labels: Auditing Diagnoses in Public ECG Datasets Against Their Measurements**
> 

> 
> 

> **About India:** we could not find any public Indian 12-lead ECG dataset, and collecting ECGs from clinics needs ethics committee approval and patient consent. So we plan to build and test on open datasets first, and add validation on Indian clinic ECGs in Phase II with approval. Is that okay, or should it be India-specific from the start?
> 

> 
> 

> For Review II on 17 Sep, should we present the new plan or the current slides? We will also get the Project Coordinator's approval for the change.
> 

> 
> 

> Regards,
> 

> Rameshkumar K — PROJ_006 (with Niranjana J and Risvanth V)
> 

---

# 2. The nine plans at a glance

| No. | Title | Type | Crowding in our check | Data |
| --- | --- | --- | --- | --- |
| 1 (E1) | Second-Look ECG ⭐ | Software enhancement | Low–medium — AI triage exists, but not error prediction for the machine's own report | PTB-XL + PTB-XL+ (open) |
| 2 (E2) | ECG Change Alert | Early detection | Medium | PTB-XL: 2,111 patients with repeat ECGs |
| 3 (E3) | AF Early Warning | Early prediction | Medium–high (WARN 2024 and later) | IRIDIA-AF (open, CC BY 4.0) |
| 4 (E5) | Smart Recording Assistant | Software enhancement | Medium–high | PTB-XL quality annotations |
| 5 (E4) | ST-Safe Live Monitoring | Software enhancement | High — since 1991 | PTB-XL + MIT-BIH noise records |
| 6 (A) | Same ECG, Different Verdict | Measurement reliability | Medium | PTB-XL+ (open) |
| 7 (B) | Swapped Electrodes, Silent Errors | Safety / robustness | Medium | PTB-XL + simulated reversals |
| 8 (C) | Children Are Not Small Adults | Population transfer | Medium | ZZU pECG (CC BY-NC-ND 4.0) |
| 9 (D) | Checking the Labels | Data quality | Medium | PTB-XL + PTB-XL+ |

---

# 3. India-specific or general?

## What is happening in India — evidence

| Fact | Source |
| --- | --- |
| Cardiovascular disease caused **28.1% of all deaths in India in 2016**, up from 15.2% in 1990. Ischaemic heart disease is the largest part, and the burden varies about nine-fold between states; Kerala and Tamil Nadu are among the highest | Lancet Global Health 2018 (opened) |
| The **Tamil Nadu STEMI programme** linked 35 spoke centres to 4 PCI hub hospitals (2,420 patients). Primary PCI rose from 29.5% to 46.5%, and 1-year mortality fell from 17.6% to 14.2%. Its STEMI kits recorded and transmitted 12-lead ECGs from ambulances and hospitals | JAMA Cardiology 2017 (opened — abstract); ECG kit details from ACC journal scan (listing) |
| AI-plus-specialist tele-ECG reading is already a commercial service in India, including primary health centres and ambulances (Tricog: "40 million+ patients diagnosed", reports "within 1–2 minutes") | Tricog website (opened — **company claims, not independently verified**) |
| Rural clinics can send ECGs to a teaching hospital: 380 ECGs from 5 clinics, 98.9% noise-free | Indian Journal of Community Medicine (listing) |
| **No public Indian 12-lead ECG dataset** was found | Search on 15 Sep; matches Gap 5 in the Research Package |
| Patient data in India needs ethics committee approval (ICMR guidelines) and, under the DPDP Act 2023 and its Rules notified 13 Nov 2025, informed consent — with a research exemption under conditions | Research Package (ICMR); Lexology / MeitY (listing) — confirm with the college ethics committee |

<aside>
⚠️

**Do not quote "74% of primary-care doctors have poor ECG knowledge" for India.** That study is from Saudi Arabia (Healthcare 2025, 257 physicians). No Indian study on doctors' ECG-reading accuracy was found in this check.

</aside>

## Recommendation — general core, India angle

- **Phase I on open datasets (general).** No Indian dataset exists; approvals and consent take time; Review III on 10 Oct needs real results.
- **India in the motivation.** Heart-disease burden, Tamil Nadu's ECG networks, and remote or automatic ECG reading already in use.
- **India in Phase II validation.** For Second-Look ECG: collect machine reports and cardiologist confirmations from a local hospital, with ethics approval, to test whether the error patterns hold on machines used in India.
- **Why general is safer for the paper.** A general method tested on public data is reproducible and easy for reviewers to check; an India-specific claim without Indian data would be challenged.

---

# 4. Guide's decision — fill in after his reply

- [ ]  Plan picked: —
- [ ]  India-specific or general: —
- [ ]  What Review II on 17 Sep presents: —
- [ ]  Date of reply: —
- [ ]  Project Coordinator approval (General Guideline 3), recorded in the Log Book: —

---

# Sources

**Opened**

- [Cardiovascular diseases in the states of India — Lancet Global Health 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6227386/)
- [Tamil Nadu STEMI programme — JAMA Cardiology 2017 (PubMed 28273293)](https://pubmed.ncbi.nlm.nih.gov/28273293/)
- [Tricog Health — company website](https://tricog.com/)
- [ECG competency of primary-care physicians, Saudi Arabia — Healthcare 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12692582/)
- [Most common errors in automatic ECG interpretation — Frontiers in Physiology 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12137353/)

**Listing only — open before citing**

- [A system of care for STEMI in India — ACC journal scan](https://www.acc.org/latest-in-cardiology/journal-scans/2017/03/09/13/15/a-system-of-care-for-patients-with-stemi)
- [Telecardiology connecting rural clinics — Indian Journal of Community Medicine](https://www.ovid.com/jnls/ijcm/fulltext/10.4103/ijcm.ijcm_368_16~feasibility-of-telecardiology-solution-to-connect-rural)
- [DPDP Act 2023 — MeitY](https://www.meity.gov.in/static/uploads/2024/06/2bf1f0e9f04e6fb4f8fef35e82c42aa5.pdf)
- [DPDP Rules 2025 overview — Lexology](https://www.lexology.com/library/detail.aspx?g=7e3af947-10aa-4712-bc1e-54179a613409)
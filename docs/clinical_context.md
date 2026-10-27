# Clinical context: intended use, outputs, harms, and what a clinician must decide

Written on 20 October 2026. This is a research prototype. Nothing here has been reviewed by a
clinician. Places that need a clinician's judgement are marked **[NEEDS CLINICIAN INPUT]**;
statements that need a published source are marked **[unverified - needs a source]**. Numbers
are not repeated here: they are in `reports/RESULTS.md`, generated from the run folders.

## Intended use (proposed, not validated)

- A research aid that combines whatever of three prenatal tests were done - fetal echo
  (structure), pulsed-wave Doppler (timing of filling and ejection) and non-invasive fetal ECG
  (rhythm) - into one of four suggestions: refer, repeat a test, get a missing test, routine.
- For researchers studying how separately built test readers can be combined, and how their
  false-alarm rates behave when they move between devices.
- Setting in which a future study could place it **[NEEDS CLINICIAN INPUT]**: which visit, which
  gestational weeks, which staff run each test, and who would receive a referral.

## Not for

- Diagnosis, or ruling out heart disease. "Routine" means no expert alarmed on the tests that
  were done; it is never a statement that the heart is normal, and the tool says so.
- Use on patients outside a research study approved by an ethics committee.
- Use with a device, population or gestational age on which the experts were not calibrated
  (calibration carried over from another device did not hold: run `wp6_fusion`).
- Twin pregnancies, fetuses with known arrhythmias under treatment, or any situation the datasets
  did not contain (see `docs/data_cards/`).

## What the outputs mean

| Output | Meaning | What it does not mean |
|---|---|---|
| REFER | at least one expert's result was unusual compared with the healthy cases it was calibrated on | that a defect or arrhythmia is present |
| REPEAT | a test could not be read, or the simultaneous ECG and Doppler did not follow the same heart | that the test is abnormal |
| GET_TEST | a test was not done and its condition is common enough to ask for it (an assumed prior) | that the condition is likely in this fetus |
| ROUTINE | no alarm on the tests that were done | that the heart is normal; untested conditions stay untested |

## Harms to weigh

- **False referrals.** With three tests the ceiling is the sum of the levels; at screening
  prevalences most referrals are false (run `wp10_allocation` gives referrals and false
  referrals per 1000 people screened, and run `wp7_dependence_bounds` the precision over a grid of
  prevalences). Each false referral costs a specialist visit, travel, and maternal anxiety
  **[NEEDS CLINICIAN INPUT: how much, relative to the benefit of a needed referral]**.
- **False reassurance.** A missed defect or arrhythmia after "routine" **[NEEDS CLINICIAN INPUT:
  consequences by condition]**. The structure expert caught under half of the defects at its
  level in DELTA, its results there are an upper bound, and on data from another country it was at chance.
- **Automation bias.** Staff may trust a "routine" more than their own reading
  **[NEEDS CLINICIAN INPUT]**.
- **Equity.** Every expert was built on data from one centre or one database; none on Indian
  patients. Prevalence of congenital heart disease in India and the share of pregnancies that get
  each test **[unverified - needs a source]**.

## Referrals per clinic

For a clinic that screens V pregnant women in a period, with prevalence p_k of condition k,
sensitivity s_k of its expert and healthy false-referral rate f:

- expected true referrals = V x sum over k of (p_k x s_k), counting only conditions whose expert
  has a measured sensitivity (structure and rhythm here; the function expert has none);
- expected false referrals = V x (1 - sum of p_k) x f, with f at most the sum of the effective
  alarm levels when every expert is calibrated on healthy cases from that clinic's devices.

Run `wp10_allocation` evaluates these per 1000 people (V = 1000) at the decision rule's priors,
which are assumptions: the structure prior is the decision rule's value, the function and rhythm
priors have no source **[unverified - needs a source]**. Replace them with a clinic's own figures
**[NEEDS CLINICIAN INPUT]**, then scale by V.

## Questions for the guide and a clinician

1. Which condition matters most to detect, and is a fixed false-referral ceiling the right way to
   set the experts' levels, or should each test keep its own clinical threshold?
2. What harm ratio (one unnecessary referral against one needed referral) is acceptable for
   prenatal cardiac referral in our setting? Run `wp10_allocation` shows net benefit across a
   grid of these ratios.
3. Which prevalences should be used for structural heart disease, abnormal Doppler timing and
   fetal arrhythmia in the screened population, and from which sources?
4. Would local calibration on healthy recordings from each device be feasible (run `wp13` gives
   the number of healthy recordings needed for the rhythm expert on one device pair)?
5. Is the ECG-Doppler trust check useful in practice, given that heart rate alone nearly does the
   same job against the mother's beats (run `wp3b_trust_baselines`)?
6. Which hospital could provide paired data (all three tests on the same fetus) under ethics
   approval, consent and a PCPNDT-registered facility, for a Phase II study?
7. Would the clinician agree to be named as a co-author, and on which parts?

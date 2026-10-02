# Research Gaps → Can We Actually Solve Them?

<aside>
🎯

Purpose: for each gap, answer two questions honestly — **is it real?** (what is the evidence) and **can this team close it?** (with what resources, by when, and what could go wrong).

A gap that is real but unsolvable by us is worth naming in the paper as future work, not promising as a contribution.

</aside>

---

# Summary

| Gap | Real? | Solvable by us? | Cost | Lands in |
| --- | --- | --- | --- | --- |
| **1. SNR is not a clinical metric** | Yes — organisers' own words | **Yes, fully** | CPU only | Review III + paper core |
| **2. Is digitization even necessary?** | Yes — organisers pose and decline it | **Yes, partially** | Modest GPU | Phase II |
| **3. Photo robustness unmeasured** | Yes — no per-type public results | Yes for measuring; partly for fixing | Free Colab | Review II + III |
| **4. Reproducibility** | Yes — organisers could not evaluate prior work | Yes — by construction | Discipline only | Throughout |
| **5. No Indian printouts** | Yes — verified absent from public data | **Conditionally** | Printing + ethics | Phase II |

---

# Gap 1 — SNR does not reflect clinical measurements

**Is it real?** Yes, and it is not our assertion. The challenge organisers write that SNR "does not directly capture or reflect clinical measurements that are likely to influence the downstream interpretation of an ECG", and that they already modified the standard SNR calculation and it still did not fix this.

**Can we solve it?** **Yes — completely, and it is the cheapest thing on this list.**

**How:**

1. Take ground-truth PTB-XL signals and the digitized outputs of an existing pipeline.
2. Run **NeuroKit2** delineation on both to extract P/QRS/T onsets and offsets.
3. Compute HR (BPM), PR, QRS, QT (ms) and ST deviation (mV) from each.
4. Report **absolute error per parameter**, and correlate against SNR.

**What it needs:** CPU only. No training. No GPU. No new data.

**What could go wrong:** delineation can fail on badly reconstructed signals — which is itself a reportable finding ("digitization degraded X% of signals beyond the point of measurability"). Mitigate by reporting delineation failure rate as a first-class result.

<aside>
⭐

**This is the project's headline contribution.** It is a *metric* contribution answering a question the field's organisers explicitly asked, it needs no GPU, and once published other groups must cite it to use it. It is also the least likely to be scooped by a well-resourced lab, because it is unglamorous.

</aside>

---

# Gap 2 — Is an intermediate time-series representation necessary?

**Is it real?** Yes. The organisers observed that teams with negative SNR still reached near-top classification scores, and wrote that this "may suggest that intermediate time-series representations may not be necessary for ECG image interpretation" — then declined to resolve it.

**Can we solve it?** **Partially.** A clean three-arm comparison on identical patient-independent splits is a genuine answer for the diagnoses PTB-XL covers. It is not a universal answer, and the paper must say so.

**How:** train three arms — direct image CNN, digitize-then-1D-model, fusion — on identical splits, evaluate macro F across image types.

**What it needs:** modest GPU. Colab free tier is likely enough with a small backbone; Colab Pro if not.

**What could go wrong:** a weak classifier in any arm invalidates the comparison. Mitigate by using established architectures and reporting each arm's standalone performance so a reviewer can see none was crippled.

**Either result publishes.** If digitization helps, the pipeline is justified. If it does not, digitization is repositioned as valuable for archival and interoperability rather than accuracy — and the organisers' open question is answered.

---

# Gap 3 — Photograph robustness is unmeasured

**Is it real?** Yes, but state it carefully. Open-ECG-Digitizer implements full perspective dewarping and PMcardio targets smartphone photos, so the claim is **not** that photos are unhandled. The claim is that **no published work reports per-image-type results for photographs on a common benchmark**. The only public per-category evidence is the challenge score table, where the winner is negative on every photo category.

**Can we solve it?**

- **Measuring it: yes, entirely.** Run existing pipelines across image types, report per-type. This is the Review II prototype and the Review III result.
- **Fixing it: partly.** Improving Stage 4 grid estimation under perspective is a bounded, well-posed problem. Beating a well-funded commercial system is not a promise to make.

**What could go wrong:** the baselines may prove hard to install and run. **This is the single most likely thing to consume the next 7 days.** Budget for dependency problems and start immediately.

---

# Gap 4 — Reproducibility

**Is it real?** Yes, stated plainly: the organisers "were unable to evaluate any of them, including ones for which code was available."

**Can we solve it?** Yes — not by discovery but by discipline. Release code, pinned dependencies, the evaluation script, and per-image-type result tables. Costs nothing but consistency, and reviewers reward it.

---

# Gap 5 — No Indian ECG printouts in any public dataset

**Is it real?** Yes. ECG-Image-Database covers Germany, the USA and Norway across 2,243 records. No Indian data. Searching found no counter-example.

**Can we solve it? Conditionally — and the route chosen decides whether it is easy or blocked.**

| Route | Patient data? | Ethics approval | Verdict |
| --- | --- | --- | --- |
| **A.** Print PTB-XL signals onto Indian ECG grid stock, photograph the physical prints | **None** | **Not required** | **Recommended.** Real paper, real camera degradation, real Indian grid geometry, zero ethics dependency |
| **B.** Photograph real de-identified clinical printouts | Yes | **Institutional Ethics Committee review required** under ICMR guidelines | Bonus only — never the critical path |

<aside>
⚠️

ECG printouts carry patient name, age, sex and hospital ID — identifiable health data. Route B needs prior IEC review, which takes weeks to months. **Start the application early if it is wanted at all, but build the project so that it does not depend on the outcome.**

Route A still supports the geography claim, because what differs about Indian printouts is the machine's grid, paper and print density — not the patients.

</aside>

---

# What we are NOT claiming

<aside>
🚫

Keeping these off the table protects the work from an easy challenge at a review or from a reviewer:

- **Not** "nobody can digitize phone photos" — PMcardio is CE-marked and does
- **Not** "we beat the state of the art" — unless and until measured
- **Not** "risk prediction" — PTB-XL has diagnostic labels, no outcome or follow-up data
- **Not** "clinically validated" — no clinician adjudication is planned
- **Not** "first to digitize ECG images" — the problem dates to at least ECGScan in 2005
</aside>
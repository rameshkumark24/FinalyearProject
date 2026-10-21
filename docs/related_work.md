# Related work: the methods this project uses and where they come from

Written on 20 October 2026. Each record below was checked on 20 October 2026 against Crossref or
Europe PMC (title, authors, journal, volume, pages, DOI); records marked **[not yet checked at
the source]** could not be opened there and must be checked before they are cited. The one-line
summaries describe how this project uses each work; where a summary says what a paper shows, it
was read from the paper's abstract. Sources already checked for the paper draft (references 1 to
42 of the draft) are not repeated unless they are needed here.

## Combining tests and the false-referral ceiling

- Dunn OJ. Multiple comparisons among means. *J Am Stat Assoc.* 1961;56(293):52-64.
  doi:10.1080/01621459.1961.10482090. The Bonferroni (union) bound: the chance that any of several
  tests alarms on a healthy person is at most the sum of their levels, whatever their dependence.
  The project's false-referral ceiling is this bound.
- Marshall RJ. The predictive value of simple rules for combining two diagnostic tests.
  *Biometrics.* 1989;45(4):1213. doi:10.2307/2531772. "Either positive" (OR) and "both positive"
  rules for two tests; the decision rule here is the OR rule over three tests.
- Pepe MS, Thompson ML. Combining diagnostic test results to increase accuracy. *Biostatistics.*
  2000;1(2):123-140. doi:10.1093/biostatistics/1.2.123. Combining markers when the same person has
  all of them (here no one has all three tests, so such combinations cannot be fitted).
- Holm S. A simple sequentially rejective multiple test procedure. *Scand J Stat.* 1979;6(2):65-70.
  **[not yet checked at the source]** (JSTOR could not be opened from this machine). For the yes/no
  referral, Holm's first step is the Bonferroni test, so it refers the same people
  (`tests/test_allocation.py`).
- Simes RJ. An improved Bonferroni procedure for multiple tests of significance. *Biometrika.*
  1986;73(3):751-754. doi:10.1093/biomet/73.3.751.
- Sarkar SK. Some probability inequalities for ordered MTP2 random variables: a proof of the Simes
  conjecture. *Ann Stat.* 1998;26(2):494-504. doi:10.1214/aos/1028144846. Simes's test keeps its
  level under positive (MTP2) dependence.
- Benjamini Y, Yekutieli D. The control of the false discovery rate in multiple testing under
  dependency. *Ann Stat.* 2001;29(4):1165-1188. doi:10.1214/aos/1013699998. Positive regression
  dependence (PRDS), the condition under which Simes-type procedures stay valid. Whether three
  tests that were never done on the same person satisfy it cannot be checked, so the rule keeps
  the Bonferroni form.

## Choosing the alarm levels

- Rubin D, Dudoit S, van der Laan M. A method to increase the power of multiple testing
  procedures through sample splitting. *Stat Appl Genet Mol Biol.* 2006;5(1):Article 19.
  doi:10.2202/1544-6115.1148. Derives the cut-offs that maximise the expected number of true
  positives while the expected number of false positives stays under a threshold, and estimates
  them on one part of the data before testing on the other. Run `wp10_allocation` applies this
  idea to the three experts' levels (a split by person); it is not a new method.
- Roeder K, Wasserman L. Genome-wide significance levels and weighted hypothesis testing.
  *Stat Sci.* 2009;24(4):398-413. doi:10.1214/09-STS289. Reviews weighted p-values, derives optimal
  weights and reports that power is robust to misspecified weights.
- Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models.
  *Med Decis Making.* 2006;26(6):565-574. doi:10.1177/0272989X06295361. Net benefit, used in run
  `wp10_allocation` over a grid of harm ratios because the clinical value is not known.

## Conformal p-values, calibration and shift

- Vovk V. Conditional validity of inductive conformal predictors. *Mach Learn.* 2013;92(2-3):349-376.
  doi:10.1007/s10994-013-5355-6. Validity conditional on the calibration set (the Beta law of the
  realised false-alarm rate) and on categories of cases (Mondrian, group-conditional, calibration).
- Angelopoulos AN, Bates S. Conformal prediction: a gentle introduction. *Found Trends Mach Learn.*
  2023;16(4):494-591. doi:10.1561/2200000101. (The paper draft cites the arXiv version,
  arXiv:2107.07511.)
- Bates S, Candès E, Lei L, Romano Y, Sesia M. Testing for outliers with conformal p-values.
  *Ann Stat.* 2023;51(1):149-178. doi:10.1214/22-AOS2244. Conformal p-values for "is this case
  unlike the healthy reference set" - the use made of them here - and their positive dependence
  when they share one calibration set (the three experts here do not share one).
- Barber RF, Candès EJ, Ramdas A, Tibshirani RJ. Conformal prediction beyond exchangeability.
  *Ann Stat.* 2023;51(2):816-845. doi:10.1214/23-AOS2276. Coverage under drift between calibration
  and use; the project's calibration-transfer result (runs `wp6_fusion`, `wp13`) is an instance.
- Romano Y, Barber RF, Sabatti C, Candès E. With malice toward none: assessing uncertainty via
  equalized coverage. *Harvard Data Sci Rev.* 2020. doi:10.1162/99608f92.03f00592. Calibration
  within groups so that every group gets the promised rate; run `wp5c` does this by image size.

## Shortcuts

- Geirhos R, Jacobsen JH, Michaelis C, Zemel R, Brendel W, Bethge M, Wichmann FA. Shortcut learning
  in deep neural networks. *Nat Mach Intell.* 2020;2(11):665-673. doi:10.1038/s42256-020-00257-z.
- DeGrave AJ, Janizek JD, Lee SI. AI for radiographic COVID-19 detection selects shortcuts over
  signal. *Nat Mach Intell.* 2021;3(7):610-619. doi:10.1038/s42256-021-00338-7.
- Zech JR, Badgeley MA, Liu M, Costa AB, Titano JJ, Oermann EK. Variable generalization performance
  of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional study.
  *PLoS Med.* 2018;15(11):e1002683. doi:10.1371/journal.pmed.1002683. Site and acquisition
  features predicting the label; the reason for the image-size checks (runs `wp5b`, `wp5c`).

## Doppler as a reference for fetal ECG

- Sulas E, Urru M, Tumbarello R, Raffo L, Sameni R, Pani D. A non-invasive multimodal foetal
  ECG-Doppler dataset for antenatal cardiology research. *Sci Data.* 2021;8:30.
  doi:10.1038/s41597-021-00811-3. The dataset's own abstract presents simultaneous pulsed-wave
  Doppler as a reference for foetal heart activity when no invasive scalp lead is available. The
  trust check is therefore a consistency (substitution) check in an established setting.
- Nii M et al. 2006 (*Heart*, doi:10.1136/hrt.2006.093070) and Stampalija T et al. 2012
  (*J Matern Fetal Neonatal Med*, doi:10.3109/14767058.2011.636090), both checked for the paper
  draft (references 28 and 20): Doppler and fetal ECG timing compared, and fetal and maternal heart
  rate confused by monitors - the failure the substitution test imitates.

## Not cited

- The Vovk, Gammerman and Shafer book (*Algorithmic Learning in a Random World*) is the original
  source of Mondrian conformal prediction, but its record could not be opened on 20 October 2026;
  Vovk 2013 is cited instead.
- Hommel's 1983 bound on Simes's test under arbitrary dependence could not be found in Crossref;
  instead, `sfrtool.allocation.simes_worst_case_two` builds a two-test example in which Simes refers
  1.5 times its total, checked by `tests/test_allocation.py`.

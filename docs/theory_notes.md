# Theory notes: the standard results the decision rule relies on

Written on 20 October 2026. None of these results is new; each is stated with a source from
`related_work.md`, the test that checks the project's code against it, and the part of run
`wp12_theory_simulation` that shows it at this project's sample sizes.

1. **Conformal p-value.** With healthy calibration scores c_1..c_n and a new score s,
   p = (1 + #{i: c_i >= s}) / (n + 1). If the new case is healthy and exchangeable with the
   calibration cases, P(p <= a) <= floor(a (n + 1)) / (n + 1) <= a. The smallest possible value is
   1/(n + 1), so a level below it never alarms, and a level between two multiples of 1/(n + 1)
   acts as the lower one. Sources: Vovk 2013; Angelopoulos and Bates 2023. Tests:
   `tests/test_theory.py` (formula, grid, average rate); simulation part A.
2. **Rate for a fixed calibration set.** Given the calibration set, the alarm rate on future
   healthy cases is random; for continuous scores it follows Beta(k, n + 1 - k) with
   k = floor(a (n + 1)). Source: Vovk 2013. Test: `test_conditional_alarm_rate_follows_the_beta_law`;
   simulation part A (percentiles against the Beta law).
3. **Union bound and its ends.** For alarm rates r_1..r_m on healthy people, the chance that any
   alarms lies between max r_k and min(1, sum r_k) for every dependence; nested alarms reach the
   lower end, alarms that never come together the upper. Source: Dunn 1961 (upper end). Tests:
   `tests/test_guarantees.py`; simulation part B.
4. **Shift.** If calibration scores come from P and use scores from Q, the rate at any fixed
   threshold changes by at most the Kolmogorov-Smirnov distance sup_t |F_P(t) - F_Q(t)|, which is
   itself at most the total-variation distance. Source for coverage under drift: Barber et al. 2023.
   Test: `test_shift_moves_the_rate_at_a_threshold_by_at_most_the_ks_distance`; simulation part C;
   run `wp13` reports the distance between the NIFEADB and NInFEA healthy scores.
5. **Group-conditional calibration.** Ranking each case among the healthy calibration cases of
   its own group keeps the rate at the level inside every group, if cases are exchangeable within
   the group; a group needs ceil(1/a) - 1 calibration cases to alarm at level a. Sources: Vovk 2013;
   Romano et al. 2020. Tests: `test_group_conditional_calibration_keeps_the_level_inside_each_group`,
   `test_minimum_group_size_and_groups_without_calibration`; simulation part D; run `wp5c`.
6. **Combining for the yes/no referral.** Holm's first step is the Bonferroni test, so both refer
   the same people; Simes is valid under positive dependence (Simes 1986; Sarkar 1998; Benjamini
   and Yekutieli 2001) and can exceed its total otherwise. Tests: `tests/test_allocation.py`; run
   `wp10` (size check).
7. **Splitting a budget.** Maximising expected true referrals sum_k p_k s_k(a_k) subject to
   sum_k a_k <= B has a closed-form optimum (equal marginal gains p_k s_k'(a_k)) when every
   sensitivity curve s_k is smooth and concave; measured curves are step functions, so run `wp10`
   searches numerically and estimates the levels on one half of the people before valuing them on
   the other (Rubin, Dudoit and van der Laan 2006). Test: `test_concave_curves_reach_the_analytic_optimum`,
   `test_search_matches_brute_force`.

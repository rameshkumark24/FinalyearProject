"""WP3 - ECG-Doppler trust check on NInFEA (simultaneous fetal ECG and pulsed-wave Doppler).

For each of the 60 recordings the fetal ECG is extracted with the frozen WP1 pipeline (thoracic
leads as the maternal reference) and the cardiac cycles are detected on the Doppler strip with
the frozen WP2 detector. The two beat series are compared interval by interval.

Mismatches are built from the real recordings:
  maternal   - the mother's own R peaks instead of the fetal ones (same recording)
  swap       - the fetal R peaks of another woman's recording
  shift      - the fetal R peaks delayed by a fixed time (labelled perturbation)
  drop       - a random share of fetal R peaks removed (labelled perturbation)
  jitter     - Gaussian timing noise added to fetal R peaks (labelled perturbation)

The thresholds of the check were set on the 13 development women only
(``splits/ninfea_dev_test.json``); the 26 test women are reported separately.

    python scripts/run_wp3_trust_check.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.data.ninfea import ALL_RECORDS, load_record, record_files, strip_file, woman_of_record  # noqa: E402
from sfr.data.pwd import load_strip  # noqa: E402
from sfr.doppler.cycles import CycleParams, detect_cycles, prepare  # noqa: E402
from sfr.doppler.envelope import PIXELS_PER_SECOND, extract_envelopes  # noqa: E402
from sfr.doppler.orientation import from_doppler, from_ecg  # noqa: E402
from sfr.ecg.extract import extract_fetal_ecg  # noqa: E402
from sfr.eval.stats import auroc_ci, bland_altman, clopper_pearson  # noqa: E402
from sfr.paths import SPLITS  # noqa: E402
from sfr.provenance import Run  # noqa: E402
from sfr.trust.crossmodal import TrustRule, paired_intervals, plausible_intervals, trust_statistics, verdict  # noqa: E402

ECG_FROZEN = dict(n_components=3, max_sources=8, n_tracked=3, smoothness=4.0, min_score=0.20, seed=0)
CYCLE_FROZEN = CycleParams()
RULE = TrustRule()
SHIFTS_S = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0, 2.0)
DROP_SHARES = (0.1, 0.2, 0.3, 0.5)
JITTER_MS = (2.0, 5.0, 10.0, 20.0)
REPEATS = 5
MIN_COMMON_S = 8.0
GAP_BINS = ((0, 1), (1, 2), (2, 4), (4, 8), (8, 999))
SEED = 0


def process(number: int) -> dict:
    record = load_record(number)
    ecg = extract_fetal_ecg(record.abdominal, record.fs, maternal_reference=record.thoracic, **ECG_FROZEN)
    envelopes = extract_envelopes(load_strip(strip_file(number)))
    prepared = prepare(envelopes, params=CYCLE_FROZEN)
    r_s = ecg.fetal_peaks / ecg.fs
    side_ecg = from_ecg(prepared, r_s * PIXELS_PER_SECOND)
    side_doppler = from_doppler(prepared, params=CYCLE_FROZEN)
    # The ECG decides the display orientation only if the extractor itself reports success.
    side = (side_ecg["side"] if ecg.ok else None) or side_doppler["side"] or side_ecg["side"] or "lower"
    cycles = detect_cycles(envelopes, side, params=CYCLE_FROZEN, prepared=prepared)
    edge = detect_cycles(envelopes, side, params=replace(CYCLE_FROZEN, use_click_start=False), prepared=prepared)
    return {
        "number": number, "duration_s": record.duration_s, "r": r_s, "maternal": ecg.maternal_peaks / ecg.fs,
        "raw": ecg.fetal_peaks_raw / ecg.fs,
        "c": cycles.starts / PIXELS_PER_SECOND, "c_edge": edge.starts / PIXELS_PER_SECOND,
        "ecg_ok": bool(ecg.ok), "ecg_quality": float(ecg.quality["score"]), "ecg_source": ecg.source,
        "maternal_bpm": float(ecg.quality["maternal_bpm"]), "side_ecg": side_ecg, "side_doppler": side_doppler,
        "side_used": side, "clicks_seen_share": float(np.mean(cycles.has_click)) if cycles.n else 0.0,
        "track_score": cycles.track_score,
    }


def clip(times: np.ndarray, duration: float) -> np.ndarray:
    return times[times < duration]


def flagged(stats: dict) -> bool:
    return not verdict(stats, RULE)["trusted"]


def beat_level(items: list[dict]) -> dict:
    """Pooled interval differences and the Bland-Altman limits of instantaneous rate."""
    diff_ms, rate_e, rate_d, keep = [], [], [], []
    for item in items:
        pairs = paired_intervals(item["r"], item["c"])
        if pairs["cc"].size == 0:
            continue
        rr_ok, cc_ok = plausible_intervals(np.diff(item["r"])), plausible_intervals(np.diff(item["c"]))
        middle = 0.5 * (item["c"][:-1] + item["c"][1:])
        which_c = np.clip(np.searchsorted(middle, pairs["t"]), 0, cc_ok.size - 1)
        which_r = np.clip(np.searchsorted(item["r"], pairs["t"], side="right") - 1, 0, rr_ok.size - 1)
        diff_ms += list(1000.0 * (pairs["cc"] - pairs["rr"]))
        rate_e += list(60.0 / pairs["rr"])
        rate_d += list(60.0 / pairs["cc"])
        keep += list(cc_ok[which_c] & rr_ok[which_r])
    diff, keep = np.array(diff_ms), np.array(keep, dtype=bool)
    rate_e, rate_d = np.array(rate_e), np.array(rate_d)
    if diff.size == 0:
        return {"n_pairs": 0}
    return {
        "n_pairs": int(diff.size),
        "abs_diff_median_ms": float(np.median(np.abs(diff))),
        "share_within_5ms": float(np.mean(np.abs(diff) <= 5)), "share_within_10ms": float(np.mean(np.abs(diff) <= 10)),
        "share_within_20ms": float(np.mean(np.abs(diff) <= 20)),
        "diff_percentiles_ms": {str(q): float(np.percentile(diff, q)) for q in (2.5, 25, 50, 75, 97.5)},
        "rate_limits_all_pairs": bland_altman(rate_d, rate_e),
        "share_plausible": float(keep.mean()),
        "rate_limits_plausible_pairs": bland_altman(rate_d[keep], rate_e[keep]),
        "rate_diff_percentiles_plausible_bpm": {str(q): float(np.percentile((rate_d - rate_e)[keep], q)) for q in (2.5, 50, 97.5)},
    }


def main() -> None:
    split = json.loads((SPLITS / "ninfea_dev_test.json").read_text(encoding="utf-8"))
    overlap = json.loads((SPLITS / "pwd_overlap.json").read_text(encoding="utf-8"))
    who = woman_of_record()
    known = {int(r): "lower" for r in overlap["ninfea_orientation_known"]["ejection_below"]}
    known.update({int(r): "upper" for r in overlap["ninfea_orientation_known"]["ejection_above"]})
    split_of = {int(r): name for name in ("development", "test") for r in split["records"][name]}
    assert sorted(split_of) == ALL_RECORDS

    started = time.time()
    items: dict[int, dict] = {}
    for number in ALL_RECORDS:
        items[number] = process(number)
        items[number].update(who[number])
        items[number]["split"] = split_of[number]
        print(f"record {number} done ({time.time() - started:.0f} s)", flush=True)

    rows, beats = [], {}
    for number, item in items.items():
        genuine = trust_statistics(item["r"], item["c"], RULE.tau_ms)
        maternal = trust_statistics(item["maternal"], item["c"], RULE.tau_ms)
        edge = trust_statistics(item["r"], item["c_edge"], RULE.tau_ms)
        decision = verdict(genuine, RULE)
        item["genuine"], item["maternal_stats"], item["trusted"] = genuine, maternal, decision["trusted"]
        rows.append({
            "record": number, "woman": item["woman"], "split": item["split"], "gestation": item["gestation"],
            "duration_s": item["duration_s"], "ecg_ok": item["ecg_ok"], "ecg_quality": item["ecg_quality"],
            "ecg_source": item["ecg_source"], "maternal_bpm": item["maternal_bpm"],
            "rate_ecg_bpm": genuine["rate_ecg_bpm"], "rate_doppler_bpm": genuine["rate_doppler_bpm"],
            "n_pairs": genuine["n_pairs"], "locked_share": genuine["locked_share"], "drift_ms": genuine["drift_ms"],
            "abs_diff_median_ms": genuine["abs_diff_median_ms"], "delay_ms": genuine["delay_ms"],
            "delay_iqr_ms": genuine["delay_iqr_ms"], "trusted": decision["trusted"], "reasons": "; ".join(decision["reasons"]),
            "maternal_locked_share": maternal["locked_share"], "maternal_drift_ms": maternal["drift_ms"],
            "maternal_flagged": flagged(maternal),
            "wave_end_locked_share": edge["locked_share"], "wave_end_abs_diff_median_ms": edge["abs_diff_median_ms"],
            "side_from_ecg": item["side_ecg"]["side"], "side_from_ecg_margin": item["side_ecg"]["margin"],
            "side_from_doppler": item["side_doppler"]["side"], "side_from_doppler_rule": item["side_doppler"]["rule"],
            "side_known": known.get(number, ""), "side_used": item["side_used"],
            "clicks_seen_share": item["clicks_seen_share"], "doppler_track_score": item["track_score"],
        })
        beats[str(number)] = {"r_s": np.round(item["r"], 4).tolist(), "closure_s": np.round(item["c"], 4).tolist(),
                              "raw_s": np.round(item["raw"], 4).tolist(), "duration_s": item["duration_s"]}
    rng = np.random.default_rng(SEED)
    swap_rows, perturb_rows = [], []
    for a, item_a in items.items():
        for b, item_b in items.items():
            if item_a["split"] != item_b["split"] or item_a["woman"] == item_b["woman"]:
                continue
            common = min(item_a["duration_s"], item_b["duration_s"])
            if common < MIN_COMMON_S:
                continue
            stats = trust_statistics(clip(item_b["r"], common), clip(item_a["c"], common), RULE.tau_ms)
            swap_rows.append({"split": item_a["split"], "doppler_record": a, "ecg_record": b, "doppler_woman": item_a["woman"],
                              "ecg_woman": item_b["woman"], "common_s": common,
                              "rate_gap_bpm": abs(stats["rate_ecg_bpm"] - stats["rate_doppler_bpm"]),
                              "locked_share": stats["locked_share"], "drift_ms": stats["drift_ms"],
                              "delay_ms": stats["delay_ms"], "flagged": flagged(stats)})
        if not item_a["trusted"]:
            continue                               # perturbations start from pairs that agree
        for shift in SHIFTS_S:
            stats = trust_statistics(clip(item_a["r"] + shift, item_a["duration_s"]), item_a["c"], RULE.tau_ms)
            perturb_rows.append({"split": item_a["split"], "record": a, "woman": item_a["woman"], "kind": "shift",
                                 "level": shift, "repeat": 0, "locked_share": stats["locked_share"],
                                 "drift_ms": stats["drift_ms"], "delay_ms": stats["delay_ms"], "flagged": flagged(stats)})
        for repeat in range(REPEATS):
            for share in DROP_SHARES:
                keep = rng.random(item_a["r"].size) >= share
                stats = trust_statistics(item_a["r"][keep], item_a["c"], RULE.tau_ms)
                perturb_rows.append({"split": item_a["split"], "record": a, "woman": item_a["woman"], "kind": "drop",
                                     "level": share, "repeat": repeat, "locked_share": stats["locked_share"],
                                     "drift_ms": stats["drift_ms"], "delay_ms": stats["delay_ms"], "flagged": flagged(stats)})
            for sigma in JITTER_MS:
                noisy = np.sort(item_a["r"] + rng.normal(0.0, sigma / 1000.0, item_a["r"].size))
                stats = trust_statistics(noisy, item_a["c"], RULE.tau_ms)
                perturb_rows.append({"split": item_a["split"], "record": a, "woman": item_a["woman"], "kind": "jitter",
                                     "level": sigma, "repeat": repeat, "locked_share": stats["locked_share"],
                                     "drift_ms": stats["drift_ms"], "delay_ms": stats["delay_ms"], "flagged": flagged(stats)})
    run = Run(
        "wp3_trust_check",
        description="ECG-Doppler trust check on the 60 NInFEA recordings: agreement of real pairs and detection of built mismatches",
        script="scripts/run_wp3_trust_check.py",
        config={"ecg": ECG_FROZEN, "cycles": asdict(CYCLE_FROZEN), "rule": asdict(RULE), "shifts_s": SHIFTS_S,
                "drop_shares": DROP_SHARES, "jitter_ms": JITTER_MS, "repeats": REPEATS, "min_common_s": MIN_COMMON_S,
                "split_file": "splits/ninfea_dev_test.json"},
        inputs={"ninfea_ecg": [p for n in ALL_RECORDS for p in record_files(n)],
                "ninfea_strips": [strip_file(n) for n in ALL_RECORDS]},
        n_real=len(ALL_RECORDS),
        n_synthetic=len(perturb_rows),
        synthetic_kind="real beat series altered by a stated rule: time shift, dropped beats or timing jitter (files perturbations.csv); maternal and swap cases are unaltered real series",
        seed=SEED,
    )

    run.write_csv("per_recording.csv", rows)
    run.write_json("beats.json", beats)
    run.write_csv("swaps.csv", swap_rows)
    run.write_csv("perturbations.csv", perturb_rows)

    metrics: dict = {"rule": asdict(RULE), "by_split": {}}
    for name in ("development", "test"):
        chosen = [r for r in rows if r["split"] == name]
        chosen_items = [items[r["record"]] for r in chosen]
        swaps = [s for s in swap_rows if s["split"] == name]
        n_trusted = sum(r["trusted"] for r in chosen)
        women = sorted({r["woman"] for r in chosen})
        women_any = sum(any(r["trusted"] for r in chosen if r["woman"] == w) for w in women)

        locked = np.array([r["locked_share"] for r in chosen])
        maternal_locked = np.nan_to_num(np.array([r["maternal_locked_share"] for r in chosen]))
        labels = np.r_[np.ones(locked.size), np.zeros(maternal_locked.size)]
        groups = np.r_[[r["woman"] for r in chosen], [r["woman"] for r in chosen]]
        auc_maternal = auroc_ci(labels, np.r_[np.nan_to_num(locked), maternal_locked], groups=groups, seed=SEED)
        swap_locked = np.nan_to_num(np.array([s["locked_share"] for s in swaps]))
        labels = np.r_[np.ones(locked.size), np.zeros(swap_locked.size)]
        groups = np.r_[[r["woman"] for r in chosen], [s["doppler_woman"] for s in swaps]]
        auc_swap = auroc_ci(labels, np.r_[np.nan_to_num(locked), swap_locked], groups=groups, seed=SEED)

        n_mat_flagged = sum(r["maternal_flagged"] for r in chosen)
        by_gap = {}
        for lo, hi in GAP_BINS:
            inside = [s for s in swaps if lo <= s["rate_gap_bpm"] < hi]
            by_gap[f"{lo}-{hi if hi < 999 else 'more'}"] = {
                "n": len(inside), "flagged": int(sum(s["flagged"] for s in inside)),
                "share": sum(s["flagged"] for s in inside) / len(inside) if inside else float("nan")}
        perturbation = {}
        for kind in ("shift", "drop", "jitter"):
            perturbation[kind] = {}
            for level in sorted({p["level"] for p in perturb_rows if p["kind"] == kind}):
                inside = [p for p in perturb_rows if p["split"] == name and p["kind"] == kind and p["level"] == level]
                perturbation[kind][str(level)] = {
                    "n": len(inside), "flagged_share": float(np.mean([p["flagged"] for p in inside])) if inside else float("nan"),
                    "locked_share_median": float(np.nanmedian([p["locked_share"] for p in inside])) if inside else float("nan")}

        rate_e = np.array([r["rate_ecg_bpm"] for r in chosen])
        rate_d = np.array([r["rate_doppler_bpm"] for r in chosen])
        trusted_mask = np.array([r["trusted"] for r in chosen], dtype=bool)
        ok_mask = np.array([r["ecg_ok"] for r in chosen], dtype=bool)
        within = np.abs(rate_d - rate_e) <= 5.0
        woman_diff = [float(np.mean([(r["rate_doppler_bpm"] - r["rate_ecg_bpm"]) for r in chosen if r["woman"] == w])) for w in women]
        delay = np.array([r["delay_ms"] for r in chosen])[trusted_mask]
        paired = [(r["locked_share"], r["wave_end_locked_share"], r["abs_diff_median_ms"], r["wave_end_abs_diff_median_ms"]) for r in chosen]

        with_truth = [r for r in chosen if r["side_known"]]
        ecg_correct = sum(r["side_from_ecg"] == r["side_known"] for r in with_truth)
        doppler_correct = sum(r["side_from_doppler"] == r["side_known"] for r in with_truth)
        without_truth = [r for r in chosen if not r["side_known"]]
        agree = sum(r["side_from_ecg"] == r["side_from_doppler"] for r in without_truth)

        metrics["by_split"][name] = {
            "n_recordings": len(chosen), "n_women": len(women),
            "trusted_recordings": int(n_trusted), "trusted_share": n_trusted / len(chosen),
            "trusted_share_ci": list(clopper_pearson(n_trusted, len(chosen))),
            "women_with_a_trusted_recording": int(women_any),
            "not_trusted": [{"record": r["record"], "reasons": r["reasons"], "locked_share": r["locked_share"],
                             "drift_ms": r["drift_ms"], "ecg_quality": r["ecg_quality"]} for r in chosen if not r["trusted"]],
            "locked_share_median": float(np.nanmedian(locked)), "locked_share_min": float(np.nanmin(locked)),
            "abs_diff_median_ms_median": float(np.nanmedian([r["abs_diff_median_ms"] for r in chosen])),
            "auroc_vs_maternal": {"auroc": auc_maternal[0], "ci": [auc_maternal[1], auc_maternal[2]], "n_pos": len(chosen), "n_neg": len(chosen)},
            "auroc_vs_swap": {"auroc": auc_swap[0], "ci": [auc_swap[1], auc_swap[2]], "n_pos": len(chosen), "n_neg": len(swaps)},
            "maternal_flagged": {"n": len(chosen), "flagged": int(n_mat_flagged), "share": n_mat_flagged / len(chosen),
                                 "ci": list(clopper_pearson(n_mat_flagged, len(chosen)))},
            "swap_flagged": {"n": len(swaps), "flagged": int(sum(s["flagged"] for s in swaps)),
                             "share": sum(s["flagged"] for s in swaps) / len(swaps) if swaps else float("nan"),
                             "by_rate_gap_bpm": by_gap},
            "perturbations_from_trusted_pairs": perturbation,
            "rate_recording_level": {"all": bland_altman(rate_d, rate_e),
                                     "ecg_quality_passed": bland_altman(rate_d[ok_mask], rate_e[ok_mask]),
                                     "trusted_only": bland_altman(rate_d[trusted_mask], rate_e[trusted_mask]),
                                     "by_woman": bland_altman(woman_diff, np.zeros(len(woman_diff))),
                                     "within_5bpm": int(within.sum()), "within_5bpm_share": float(within.mean()),
                                     "abs_error_median_bpm": float(np.nanmedian(np.abs(rate_d - rate_e)))},
            "ecg_quality_passed_recordings": int(ok_mask.sum()),
            "beat_level_all_recordings": beat_level(chosen_items),
            "beat_level_ecg_quality_passed_recordings": beat_level([i for i in chosen_items if i["ecg_ok"]]),
            "beat_level_trusted_recordings": beat_level([i for i in chosen_items if i["trusted"]]),
            "delay_r_to_closure_ms_trusted": {"n": int(delay.size), "median": float(np.median(delay)) if delay.size else float("nan"),
                                              "min": float(delay.min()) if delay.size else float("nan"),
                                              "max": float(delay.max()) if delay.size else float("nan")},
            "cycle_start_ablation": {
                "closure_click": {"locked_share_median": float(np.nanmedian([p[0] for p in paired])), "abs_diff_median_ms": float(np.nanmedian([p[2] for p in paired]))},
                "wave_end": {"locked_share_median": float(np.nanmedian([p[1] for p in paired])), "abs_diff_median_ms": float(np.nanmedian([p[3] for p in paired]))},
                "recordings_where_click_is_better": int(sum(p[0] > p[1] for p in paired)), "n": len(paired)},
            "orientation": {
                "known_n": len(with_truth),
                "from_ecg_correct": int(ecg_correct), "from_ecg_ci": list(clopper_pearson(ecg_correct, len(with_truth))) if with_truth else [],
                "from_doppler_correct": int(doppler_correct), "from_doppler_ci": list(clopper_pearson(doppler_correct, len(with_truth))) if with_truth else [],
                "unknown_n": len(without_truth), "unknown_two_methods_agree": int(agree),
                "wrong_from_ecg": [r["record"] for r in with_truth if r["side_from_ecg"] != r["side_known"]],
                "wrong_from_doppler": [r["record"] for r in with_truth if r["side_from_doppler"] != r["side_known"]]},
        }

    test = metrics["by_split"]["test"]
    # Gate G3 as written in the plan: "on true pairs, limits of agreement within +-10 bpm for
    # recordings passing the quality check; AUROC >= 0.90 for detecting maternal substitution".
    # The quality check is the extractor's own (ecg_ok); agreement is beat to beat.
    beat = test["beat_level_ecg_quality_passed_recordings"]
    beat_limits = beat.get("rate_limits_plausible_pairs", {})
    recording_limits = test["rate_recording_level"]["ecg_quality_passed"]
    trusted_limits = test["rate_recording_level"]["trusted_only"]
    inside = lambda limits: bool(limits and limits["loa_low"] >= -10 and limits["loa_high"] <= 10)  # noqa: E731
    metrics["gate_G3"] = {
        "rule": "test women: beat-to-beat Bland-Altman limits within +-10 bpm in recordings that pass the ECG quality check, "
                "and AUROC >= 0.90 for maternal substitution",
        "beat_level_limits_bpm": [beat_limits.get("loa_low"), beat_limits.get("loa_high")],
        "beat_level_central_95_percent_bpm": [beat.get("rate_diff_percentiles_plausible_bpm", {}).get("2.5"),
                                               beat.get("rate_diff_percentiles_plausible_bpm", {}).get("97.5")],
        "beat_level_passed": inside(beat_limits),
        "auroc_maternal_test": test["auroc_vs_maternal"]["auroc"],
        "auroc_passed": bool(test["auroc_vs_maternal"]["auroc"] >= 0.90),
        "also_reported": {
            "recording_level_limits_bpm_quality_passed": [recording_limits["loa_low"], recording_limits["loa_high"]],
            "recording_level_within_10": inside(recording_limits),
            "recording_level_limits_bpm_trust_check_passed": [trusted_limits["loa_low"], trusted_limits["loa_high"]],
        },
    }
    metrics["gate_G3"]["passed"] = bool(metrics["gate_G3"]["beat_level_passed"] and metrics["gate_G3"]["auroc_passed"])
    metrics["n_perturbed_cases"] = len(perturb_rows)
    metrics["note_on_perturbations"] = ("shift, drop and jitter cases are real beat series altered by a stated rule; "
                                        "maternal and swap cases use unaltered real beat series")
    run.finish(metrics)

    for name in ("development", "test"):
        m = metrics["by_split"][name]
        print(f"\n== {name}: {m['n_recordings']} recordings, {m['n_women']} women ==")
        print(f"trusted {m['trusted_recordings']}/{m['n_recordings']}  CI {np.round(m['trusted_share_ci'], 2)}; not trusted: {[(x['record'], x['reasons']) for x in m['not_trusted']]}")
        print(f"locked share median {m['locked_share_median']:.2f} (min {m['locked_share_min']:.2f}); median |interval diff| {m['abs_diff_median_ms_median']:.1f} ms")
        print(f"AUROC vs maternal {m['auroc_vs_maternal']['auroc']:.3f} {np.round(m['auroc_vs_maternal']['ci'], 3)}; vs swap {m['auroc_vs_swap']['auroc']:.3f} {np.round(m['auroc_vs_swap']['ci'], 3)} (n swaps {m['auroc_vs_swap']['n_neg']})")
        print(f"rule: maternal flagged {m['maternal_flagged']['flagged']}/{m['maternal_flagged']['n']}; swaps flagged {m['swap_flagged']['flagged']}/{m['swap_flagged']['n']}; by rate gap {[(k, v['flagged'], v['n']) for k, v in m['swap_flagged']['by_rate_gap_bpm'].items()]}")
        for kind, levels in m["perturbations_from_trusted_pairs"].items():
            print(f"  {kind}: " + ", ".join(f"{level}: {v['flagged_share']:.2f}" for level, v in levels.items()))
        print("rate, recording level:", {k: (np.round([v['bias'], v['loa_low'], v['loa_high']], 2).tolist() if isinstance(v, dict) else v) for k, v in m["rate_recording_level"].items()})
        for key in ("beat_level_all_recordings", "beat_level_ecg_quality_passed_recordings", "beat_level_trusted_recordings"):
            b = m[key]
            if b.get("n_pairs"):
                print(f"{key}: n {b['n_pairs']}, median |d| {b['abs_diff_median_ms']:.1f} ms, within 10 ms {b['share_within_10ms']:.3f}, "
                      f"limits all {np.round([b['rate_limits_all_pairs']['loa_low'], b['rate_limits_all_pairs']['loa_high']], 1).tolist()}, "
                      f"plausible {np.round([b['rate_limits_plausible_pairs']['loa_low'], b['rate_limits_plausible_pairs']['loa_high']], 1).tolist()} bpm")
        print("delay R to closure (trusted):", m["delay_r_to_closure_ms_trusted"])
        print("cycle start ablation:", m["cycle_start_ablation"])
        print("orientation:", m["orientation"])
    print("\nGate G3:", metrics["gate_G3"])


if __name__ == "__main__":
    main()

"""Score one patient's tests and apply the decision rule. Research prototype.

    python scripts/predict_patient.py --ecg ECG_RECORD --doppler STRIP --gestation 25+3 --out tool_results/case.json
    python scripts/predict_patient.py --echo ECHO_FOLDER --cpu

Research prototype. Not validated on patients with all three tests. Not for clinical use.

Give any of --ecg, --doppler and --echo (at least one). Only the experts whose test was given
run. If the fetal ECG and the Doppler were recorded together (as in NInFEA), the ECG-Doppler
trust check runs as well. The decision is made by sfr.fusion.decision.decide, unchanged, and
the output says which conditions were checked and which were not.

The rhythm expert was fitted on NIFEADB. Calibrated there, it referred about half of the healthy
NInFEA fetuses (run wp6_fusion). Use --calibration with healthy scores from your own device and
setting (a CSV with a column "score", for example the "score" of the rhythm expert in this tool's
output for healthy recordings); the tool warns whenever the calibration comes from another
setting.

Patient files and results must never enter git: write the result outside the repository or into
tool_results/, which git ignores. The tool refuses any other place inside the repository.
"""

from __future__ import annotations

# On the development laptop PyTorch must be imported before NumPy work, or the process crashes.
try:
    import torch  # noqa: F401
    import open_clip  # noqa: F401
except ImportError:                                  # the ECG and Doppler experts do not need it
    torch = None

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.doppler.reference import conformal_p  # noqa: E402
from sfr.echo.fetalclip import FETALCLIP_SHA256  # noqa: E402
from sfr.fusion.conformal import p_value, smallest_p  # noqa: E402
from sfr.fusion.decision import EXPERTS, DecisionConfig, ExpertResult, decide  # noqa: E402
from sfr.provenance import code_sha256, git_commit  # noqa: E402
from sfrtool import ARTIFACTS, CONDITIONS, DISCLAIMER, ROOT, TESTS, TOOL_VERSION, tool_code_sha256  # noqa: E402
from sfrtool.artifacts import _sha256, artifact_sha256, load_artifact  # noqa: E402
from sfrtool.experts import (ECG_LAYOUTS, read_doppler, read_ecg, read_echo, rhythm_probability,  # noqa: E402
                             structure_probability, trust_check, function_score)
from sfrtool.labels import describe  # noqa: E402

PRIORS_NOTE = "the priors of function and rhythm findings in the decision rule are assumptions without a cited source"


def gestation_days(text: str) -> float:
    """'25+3' (weeks+days) or a number of days."""
    if "+" in text:
        weeks, days = text.split("+", 1)
        return float(int(weeks) * 7 + int(days))
    return float(text)


def check_output_path(path: Path) -> None:
    """Refuse a result file inside the repository unless git ignores it."""
    target = path.resolve()
    try:
        target.relative_to(ROOT)
    except ValueError:
        return
    ignored = subprocess.run(["git", "check-ignore", "-q", str(target)], cwd=ROOT).returncode == 0
    if not ignored:
        raise SystemExit(f"Refused: {target} is inside the repository and not ignored by git. Patient results must never be "
                         "committed; write to tool_results/ (ignored) or outside the repository.")


def rhythm_calibration(choice: str, arrays: dict, card: dict) -> tuple[np.ndarray, dict]:
    """Healthy scores for the rhythm p-value: a built-in set by name, or a CSV of the user's own healthy scores."""
    sets = card["calibration_sets"]
    if choice in sets:
        entry = sets[choice]
        return arrays[entry["array"]], {"kind": "published" if choice == "published" else "built-in", "name": choice,
                                        "n": int(entry["n"]), "setting": entry["setting"], "source": entry["source"]}
    path = Path(choice)
    if not path.exists():
        raise SystemExit(f"--calibration must be one of {sorted(sets)} or a CSV file; {path} does not exist")
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    header = [h.strip().lower() for h in rows[0]] if rows else []
    column = header.index("score") if "score" in header else 0
    body = rows[1:] if "score" in header else rows
    try:
        scores = np.array([float(row[column]) for row in body if row])
    except ValueError as error:
        raise SystemExit(f"{path}: every row must hold a number in the score column ({error})")
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise SystemExit(f"{path} holds no healthy scores")
    return scores, {"kind": "local", "name": path.name, "n": int(scores.size), "setting": "the user's own setting", "source": str(path.name)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ecg", type=Path, help="fetal ECG: WFDB record (.hea/.dat, path with or without the extension)")
    parser.add_argument("--ecg-layout", choices=ECG_LAYOUTS, default="ninfea", help="channel layout of the ECG record")
    parser.add_argument("--doppler", type=Path, help="pulsed-wave Doppler strip image (284 columns per second, NInFEA format)")
    parser.add_argument("--gestation", help="gestational age as weeks+days (25+3) or days; needed with --doppler")
    parser.add_argument("--echo", type=Path, help="folder with one image of each plane: Abdomen, 4CH, LVOT, RVOT, 3VT")
    parser.add_argument("--calibration", default="published",
                        help="healthy scores for the rhythm expert: 'published' (NIFEADB), 'ninfea', or a CSV with a column 'score'")
    parser.add_argument("--not-simultaneous", action="store_true", help="the ECG and the Doppler were not recorded together (no trust check)")
    parser.add_argument("--out", type=Path, help="write the full result as JSON here")
    parser.add_argument("--cpu", action="store_true", help="run FetalCLIP on the CPU even if a GPU is there")
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS, help="folder of the expert artifacts")
    args = parser.parse_args()
    if not (args.ecg or args.doppler or args.echo):
        parser.error("give at least one test: --ecg, --doppler or --echo")
    if args.doppler and args.gestation is None:
        parser.error("--doppler needs --gestation: the healthy reference depends on gestational age")
    if args.out:
        check_output_path(args.out)

    config = DecisionConfig()
    results: dict[str, ExpertResult] = {}
    experts: dict[str, dict] = {}
    warnings: list[str] = []
    artifacts: dict[str, str] = {}
    inputs: dict[str, dict] = {}
    ecg = doppler = None

    if args.ecg:
        arrays, card = load_artifact("rhythm", args.artifacts)
        artifacts["rhythm"] = artifact_sha256("rhythm", args.artifacts)
        healthy, calibration = rhythm_calibration(args.calibration, arrays, card)
        lowest = smallest_p(calibration["n"])
        if config.alpha["rhythm"] < lowest:
            raise SystemExit(f"Refused: with {calibration['n']} healthy calibration scores the smallest possible p-value is {lowest:.3f}, "
                             f"above the rhythm alarm level {config.alpha['rhythm']}, so the expert could never alarm. "
                             f"Give at least {int(np.ceil(1 / config.alpha['rhythm'])) - 1} healthy scores.")
        if calibration["kind"] == "local":
            calibration_text = f"local (n={calibration['n']})"
            warnings.append(f"rhythm expert calibrated on {calibration['n']} healthy scores you supplied: they must come from the same device "
                            "and setting as this ECG.")
        else:
            calibration_text = f"{calibration['kind']} ({calibration['setting'].upper()}, n={calibration['n']})"
            transport = card["transport"]
            if calibration["setting"] != args.ecg_layout:
                warnings.append(f"WARNING - domain shift: the rhythm expert is calibrated on {calibration['setting'].upper()} healthy recordings, "
                                f"not on the setting of this ECG ({args.ecg_layout.upper()} layout). Calibrated on NIFEADB it referred "
                                f"{transport['false_referral_share_on_ninfea_with_published_calibration']:.1%} of healthy NInFEA fetuses "
                                f"(run {transport['run']}). Use --calibration with healthy scores from your own device and setting.")
            else:
                warnings.append(f"rhythm expert calibrated on {calibration['setting'].upper()} healthy recordings ({calibration['n']}); "
                                "check that this ECG comes from the same device and setting.")
        ecg = read_ecg(args.ecg, args.ecg_layout)
        stem = args.ecg.with_suffix("") if args.ecg.suffix.lower() in (".hea", ".dat") else args.ecg
        inputs["ecg"] = {"layout": args.ecg_layout, "sha256": {s: _sha256(stem.with_suffix(s)) for s in (".hea", ".dat")}}
        entry = {"test": TESTS["rhythm"], "condition": CONDITIONS["rhythm"], "read": ecg.readable, "reason": ecg.reason,
                 "calibration": calibration_text, "calibration_detail": calibration, "alarm_level": config.alpha["rhythm"],
                 "smallest_possible_p": lowest, "duration_s": ecg.duration_s, "segments_used": ecg.segments_used}
        p = None
        if ecg.readable:
            score = rhythm_probability(ecg.features, arrays)
            p = p_value(score, healthy)
            entry.update({"score": score, "p_value": p, "alarm": p <= config.alpha["rhythm"], "features": ecg.features})
        experts["rhythm"] = entry
        results["rhythm"] = ExpertResult(present=True, quality_ok=ecg.readable, p_value=p)

    if args.doppler:
        arrays, card = load_artifact("function", args.artifacts)
        artifacts["function"] = artifact_sha256("function", args.artifacts)
        days = gestation_days(args.gestation)
        low, high = arrays["gestation_days_range"]
        if not low <= days <= high:
            warnings.append(f"gestational age {int(days) // 7}+{int(days) % 7} is outside the healthy reference "
                            f"({int(low) // 7}+{int(low) % 7} to {int(high) // 7}+{int(high) % 7}); the function expert extrapolates.")
        together = ecg is not None and not args.not_simultaneous and args.ecg_layout == "ninfea"
        doppler = read_doppler(args.doppler, r_s=ecg.r_s if together else None, ecg_ok=bool(together and ecg.extractor_ok))
        inputs["doppler"] = {"sha256": _sha256(args.doppler), "gestation_days": days}
        entry = {"test": TESTS["function"], "condition": CONDITIONS["function"], "read": doppler.readable, "reason": doppler.reason,
                 "calibration": f"published: healthy NInFEA recordings (n={arrays['calibration'].size})", "alarm_level": config.alpha["function"],
                 "smallest_possible_p": smallest_p(int(arrays["calibration"].size)), "ejection_side": doppler.side, "cycles": doppler.n_cycles}
        p = None
        if doppler.readable:
            score, z = function_score(doppler.measures, days, arrays, card)
            p = conformal_p(score, arrays["calibration"])
            entry.update({"score": score, "p_value": p, "alarm": p <= config.alpha["function"], "z": z,
                          "measures": {k: doppler.measures[k] for k in card["measures"]}})
        experts["function"] = entry
        results["function"] = ExpertResult(present=True, quality_ok=doppler.readable, p_value=p)

    trust_ok, trust = None, {"run": False}
    if ecg is not None and doppler is not None:
        if args.not_simultaneous:
            trust = {"run": False, "why": "the ECG and the Doppler were not recorded together"}
            warnings.append("trust check not run (--not-simultaneous): nothing verifies that the ECG and the Doppler show the same heart.")
        elif args.ecg_layout != "ninfea":
            trust = {"run": False, "why": "the trust check needs an ECG and a strip recorded together on one clock, as in NInFEA"}
            warnings.append("trust check not run: the ECG is not in the NInFEA layout, so it was not recorded together with the strip.")
        else:
            trust = {"run": True, **trust_check(ecg.r_s, doppler.c_s)}
            trust_ok = trust["trusted"]

    if args.echo:
        device = "cuda" if (torch is not None and not args.cpu and torch.cuda.is_available()) else "cpu"
        arrays, card = load_artifact("structure", args.artifacts, model={"name": "FetalCLIP", "sha256": FETALCLIP_SHA256})
        artifacts["structure"] = artifact_sha256("structure", args.artifacts)
        echo = read_echo(args.echo, device)
        inputs["echo"] = {"planes": echo.planes, "device": device}
        entry = {"test": TESTS["structure"], "condition": CONDITIONS["structure"], "read": echo.readable, "reason": echo.reason,
                 "calibration": f"published: normal DELTA Val cases (n={arrays['calibration'].size})", "alarm_level": config.alpha["structure"],
                 "smallest_possible_p": smallest_p(int(arrays["calibration"].size))}
        p = None
        if echo.readable:
            score = structure_probability(echo.embedding, arrays)
            p = p_value(score, arrays["calibration"])
            entry.update({"score": score, "p_value": p, "alarm": p <= config.alpha["structure"], "model": echo.model})
        experts["structure"] = entry
        results["structure"] = ExpertResult(present=True, quality_ok=echo.readable, p_value=p)
        warnings.append("structure expert: inputs with little or no cardiac anatomy also separated the DELTA classes; its output is an upper "
                        "bound on what heart structure contributes (see the artifact card).")

    decision = decide(results, trust_ok, config)
    labels = describe(decision, results, trust_ok)
    output = {
        "disclaimer": DISCLAIMER,
        "tool": {"version": TOOL_VERSION, "tool_code_sha256": tool_code_sha256(), "sfr_code_sha256": code_sha256(), "git_commit": git_commit()},
        "artifacts_sha256": artifacts,
        "alarm_levels": dict(config.alpha), "priors": dict(config.prior), "priors_note": PRIORS_NOTE,
        "false_referral_ceiling": float(sum(config.alpha[n] for n in results)),
        "inputs": inputs, "experts": experts, "trust_check": trust,
        "decision": {"action": decision.action, "reasons": decision.reasons, "alarms": decision.alarms, "repeat": decision.repeat,
                     "get": decision.get, **labels},
        "not_measured": "the performance of this tool on patients who had all three tests",
        "warnings": warnings,
    }
    print(DISCLAIMER + "\n")
    for name in EXPERTS:
        if name not in experts:
            print(f"  {TESTS[name]}: not given")
            continue
        e = experts[name]
        if not e["read"]:
            print(f"  {e['test']} ({e['condition']}): could not be read - {e['reason']}")
        else:
            print(f"  {e['test']} ({e['condition']}): p = {e['p_value']:.3f} at level {e['alarm_level']} -> "
                  f"{'ALARM' if e['alarm'] else 'no alarm'}  [calibration: {e['calibration']}]")
    if trust.get("run"):
        print(f"  trust check (ECG and Doppler show the same heart): {'passed' if trust['trusted'] else 'FAILED - ' + '; '.join(trust['reasons'])}")
    print(f"\nAction: {decision.action}\n{labels['display_label']}")
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v)), encoding="utf-8")
        print(f"\nResult written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

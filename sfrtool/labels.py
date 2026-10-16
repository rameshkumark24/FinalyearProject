"""Say exactly which conditions were checked, around the unchanged decision of ``sfr.fusion.decision``.

"ROUTINE" after an echo and a Doppler without a fetal ECG must not read like "the heart is
normal". ``describe`` adds three fields to a decision without changing it:

- ``covered``: conditions whose test was read (and, for ECG and Doppler, passed the trust check
  or alarmed - an alarm is never cancelled);
- ``not_covered``: every other condition, with the reason (not done, could not be read, not
  trusted);
- ``display_label``: plain words. Every label lists what was and was not checked. A label says
  "no alarm" only if at least one condition was checked, and then always adds that this is not a
  statement that the heart is normal.

The stored action keeps its name (REFER, REPEAT, GET_TEST, ROUTINE).
"""

from __future__ import annotations

from typing import Mapping

from sfr.fusion.decision import EXPERTS, PAIRED_BY_TRUST_CHECK, Decision, ExpertResult

from . import CONDITIONS, TESTS

NOT_NORMAL = "This is not a statement that the heart is normal."


def _named(names: list[str]) -> str:
    return ", ".join(f"{CONDITIONS[n]} ({TESTS[n]})" for n in names)


def describe(decision: Decision, results: Mapping[str, ExpertResult], trust_ok: bool | None) -> dict:
    results = {name: results.get(name, ExpertResult()) for name in EXPERTS}
    trusted = not (trust_ok is False and all(results[name].usable for name in PAIRED_BY_TRUST_CHECK))
    checked, why = [], {}
    for name in EXPERTS:
        result = results[name]
        if name in decision.alarms:
            checked.append(name)
        elif not result.present:
            why[name] = "not done"
        elif not result.usable:
            why[name] = "could not be read"
        elif name in PAIRED_BY_TRUST_CHECK and not trusted:
            why[name] = "not trusted: the fetal ECG and the Doppler do not show the same heart"
        else:
            checked.append(name)
    not_checked = [name for name in EXPERTS if name not in checked]
    quiet = [name for name in checked if name not in decision.alarms]
    missing_text = "; ".join(f"{CONDITIONS[n]} ({TESTS[n]}: {why[n]})" for n in not_checked)

    if decision.action == "REFER":
        parts = [f"Refer: alarm on {_named(list(decision.alarms))}."]
        if quiet:
            parts.append(f"Checked without alarm: {_named(quiet)}.")
        parts.append(f"NOT checked: {missing_text}." if not_checked else "All three tests were read.")
    elif decision.action == "REPEAT":
        parts = [f"Repeat: {'; '.join(decision.reasons)}."]
        if quiet:
            parts.append(f"Checked without alarm: {_named(quiet)}.")
        parts.append(f"NOT checked: {missing_text}.")
        parts.append("Nothing can be concluded about a test until it is repeated and read.")
    elif not checked:                                  # GET_TEST with nothing usable: no reassuring words at all
        parts = ["No test could be used, so nothing was checked.", f"NOT checked: {missing_text}.",
                 f"Get: {', '.join(TESTS[n] for n in decision.get)}."]
    else:                                               # GET_TEST or ROUTINE after at least one checked test
        parts = []
        if decision.action == "GET_TEST":
            parts.append(f"Get the missing test: {', '.join(TESTS[n] for n in decision.get)}.")
        parts.append(f"No alarm on the tests done ({_named(checked)}).")
        parts.append(f"NOT checked: {missing_text}." if not_checked else "All three tests were read.")
        parts.append(NOT_NORMAL)
    return {"covered": [CONDITIONS[n] for n in checked], "not_covered": [CONDITIONS[n] for n in not_checked],
            "not_covered_reasons": {CONDITIONS[n]: why[n] for n in not_checked}, "display_label": " ".join(parts)}

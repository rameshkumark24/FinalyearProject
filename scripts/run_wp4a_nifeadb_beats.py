"""WP4a - fetal beats of the 26 NIFEADB recordings, one-minute segment by segment.

The frozen WP1 extractor (maternal chest lead as reference) is run on consecutive 60-second
segments, the length it was tuned on. For every segment the script stores the tracked beats, the
raw threshold-only beats (used for rhythm analysis), the extractor's quality values and the
selected fetal source signal. Nothing here uses the arrhythmia label.

    python scripts/run_wp4a_nifeadb_beats.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.data.nifeadb import ALL_RECORDS, load_record, record_files  # noqa: E402
from sfr.ecg.extract import extract_fetal_ecg  # noqa: E402
from sfr.provenance import Run  # noqa: E402

ECG_FROZEN = dict(n_components=3, max_sources=8, n_tracked=3, smoothness=4.0, min_score=0.20, seed=0)
SEGMENT_S = 60.0
MIN_LAST_S = 30.0


def main() -> None:
    run = Run(
        "wp4a_nifeadb_beats",
        description="Fetal beat extraction on the 26 NIFEADB recordings in 60-second segments",
        script="scripts/run_wp4a_nifeadb_beats.py",
        config={"ecg": ECG_FROZEN, "segment_s": SEGMENT_S, "min_last_segment_s": MIN_LAST_S},
        inputs={"nifeadb": [p for name in ALL_RECORDS for p in record_files(name)]},
        n_real=len(ALL_RECORDS),
        seed=ECG_FROZEN["seed"],
    )
    rows, beats, sources = [], {}, {}
    started = time.time()
    for name in ALL_RECORDS:
        record = load_record(name)
        step = int(round(SEGMENT_S * record.fs))
        n_samples = record.abdominal.shape[1]
        edges = list(range(0, n_samples, step))
        segments = [(a, min(a + step, n_samples)) for a in edges if min(a + step, n_samples) - a >= MIN_LAST_S * record.fs]
        beats[name] = []
        signals = []
        for k, (a, b) in enumerate(segments):
            result = extract_fetal_ecg(record.abdominal[:, a:b], record.fs, maternal_reference=record.chest[a:b], **ECG_FROZEN)
            offset = a / record.fs
            quality = result.quality
            rows.append({
                "record": name, "label": record.label, "segment": k, "start_s": offset, "length_s": (b - a) / record.fs,
                "fs_original": record.fs, "n_channels": int(record.abdominal.shape[0]), "ok": bool(result.ok),
                "score": quality["score"], "rate_bpm": quality["rate_bpm"], "regularity_tracked": quality["regularity"],
                "regularity_raw": quality["regularity_raw"], "template_corr": quality["template_corr"],
                "maternal_bpm": quality["maternal_bpm"], "maternal_coincidence": quality["maternal_coincidence"],
                "maternal_phase_lock": quality["maternal_phase_lock"], "maternal_lock": bool(quality["maternal_lock"]),
                "source": result.source, "n_tracked": int(result.fetal_peaks.size), "n_raw": int(result.fetal_peaks_raw.size),
            })
            beats[name].append({
                "segment": k, "start_s": offset, "ok": bool(result.ok),
                "tracked_s": np.round(offset + result.fetal_peaks / result.fs, 4).tolist(),
                "raw_s": np.round(offset + result.fetal_peaks_raw / result.fs, 4).tolist(),
                "maternal_s": np.round(offset + result.maternal_peaks / result.fs, 4).tolist(),
            })
            signals.append(result.fetal_signal.astype(np.float32))
        sources[name] = signals
        print(f"{name}: {len(segments)} segments, {sum(r['ok'] for r in rows if r['record'] == name)} usable "
              f"({time.time() - started:.0f} s)", flush=True)

    run.write_csv("segments.csv", rows)
    run.write_json("beats.json", beats)
    np.savez_compressed(run.path / "fetal_sources.npz",
                        **{f"{name}__{k}": signal for name, signals in sources.items() for k, signal in enumerate(signals)})

    by_label = {}
    for label, text in ((1, "arrhythmia"), (0, "normal")):
        chosen = [r for r in rows if r["label"] == label]
        records = sorted({r["record"] for r in chosen})
        usable = [np.mean([r["ok"] for r in chosen if r["record"] == rec]) for rec in records]
        by_label[text] = {
            "n_records": len(records), "n_segments": len(chosen), "usable_segments": int(sum(r["ok"] for r in chosen)),
            "usable_share_per_record_median": float(np.median(usable)), "usable_share_per_record_min": float(np.min(usable)),
            "records_with_no_usable_segment": [rec for rec, u in zip(records, usable) if u == 0],
            "score_median": float(np.median([r["score"] for r in chosen])),
        }
    run.finish({"segment_s": SEGMENT_S, "n_segments": len(rows), "by_label": by_label,
                "fetal_source_sampling_hz": 1000,
                "note": "The usable share is reported by label because an extractor tuned for regular rhythms may fail more often on irregular ones."})
    print(by_label)


if __name__ == "__main__":
    main()

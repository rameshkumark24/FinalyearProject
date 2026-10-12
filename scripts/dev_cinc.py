"""Development helper: score the extraction pipeline on a list of CinC 2013 records."""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sfr.data.cinc2013 import load_record
from sfr.ecg.extract import extract_fetal_ecg
from sfr.ecg.metrics import beat_scores, heart_rate_bpm, follows_mother

def run(records, verbose=True, **kwargs):
    rows = []
    for name in records:
        x, fs, ref = load_record(name)
        t0 = time.time()
        ex = extract_fetal_ecg(x, fs, **kwargs)
        s = beat_scores(ex.fetal_peaks, ref, ex.fs)
        true_bpm = heart_rate_bpm(ref, fs); pred_bpm = heart_rate_bpm(ex.fetal_peaks, ex.fs)
        lock = follows_mother(pred_bpm, true_bpm, ex.quality["maternal_bpm"])
        rows.append(dict(rec=name, f1=s["f1"], se=s["sensitivity"], ppv=s["ppv"], true=true_bpm, pred=pred_bpm,
                         mat=ex.quality["maternal_bpm"], lock=lock, ok=ex.ok, src=ex.source, score=ex.quality["score"],
                         tc=ex.quality["template_corr"], reg=ex.quality["regularity_raw"], msrc=ex.quality["maternal_source"], t=time.time()-t0))
        if verbose:
            r = rows[-1]
            print(f"{name} F1={r['f1']:.3f} Se={r['se']:.3f} PPV={r['ppv']:.3f} true={r['true']:.0f} pred={r['pred']:.0f} mat={r['mat']:.0f} "
                  f"{'LOCK ' if lock else ''}{'' if ex.ok else 'FAILED-FLAG '}src={r['src']} score={r['score']:.2f} tc={r['tc']:.2f} reg={r['reg']:.2f} msrc={r['msrc']} {r['t']:.1f}s", flush=True)
    f1 = np.array([r["f1"] for r in rows])
    print(f"n={len(rows)} mean F1={f1.mean():.3f} median F1={np.median(f1):.3f} | F1>=0.9: {(f1>=0.9).sum()} | F1<0.5: {(f1<0.5).sum()} | follows mother: {sum(r['lock'] for r in rows)} | flagged failed: {sum(not r['ok'] for r in rows)}")
    return rows

if __name__ == "__main__":
    first, last = int(sys.argv[1]), int(sys.argv[2])
    run([f"a{i:02d}" for i in range(first, last + 1)])

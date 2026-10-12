"""WP4d - a small 1D convolutional network trained from scratch on extracted fetal ECG (NIFEADB).

Two questions:
1. Leave-one-subject-out: how well does a network that is really trained separate arrhythmia from
   normal rhythm? One score per recording = mean of its window probabilities.
2. Leakage on real data (R4): the same network scored once with windows of a recording on both
   sides of a 10-fold split and once with the recording held out.

Everything is fixed in advance (architecture, 20 epochs, learning rate, no early stopping, no
tuning), because with 26 subjects there is no data to tune on.

    python scripts/run_wp4d_cnn.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from scipy import signal
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.data.nifeadb import ALL_RECORDS, label_of, record_files  # noqa: E402
from sfr.eval.stats import auroc_ci  # noqa: E402
from sfr.paths import RUNS  # noqa: E402
from sfr.provenance import Run, load_run  # noqa: E402
from sfr.rhythm.evaluate import summarise  # noqa: E402

SOURCE_RUN = "wp4a_nifeadb_beats"
FS, WINDOW_S = 250, 10
EPOCHS, BATCH, LEARNING_RATE, WEIGHT_DECAY, SEED = 20, 64, 1e-3, 1e-4, 0
SEEDS = (0, 1, 2)            # the first is the reported run; the others show how much training noise there is


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        def block(n_in: int, n_out: int, stride: int, pool: bool) -> list[nn.Module]:
            layers: list[nn.Module] = [nn.Conv1d(n_in, n_out, kernel_size=7, stride=stride, padding=3), nn.BatchNorm1d(n_out), nn.ReLU()]
            return layers + ([nn.MaxPool1d(2)] if pool else [])

        self.features = nn.Sequential(*block(1, 16, 2, False), *block(16, 32, 2, True), *block(32, 64, 2, True), *block(64, 64, 1, False))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).mean(-1)).squeeze(-1)


def train_and_score(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, device: str, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    model = SmallCNN().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    positive = float(y_train.sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y_train) - positive) / max(positive, 1.0), device=device))
    xt = torch.from_numpy(x_train).unsqueeze(1).to(device)
    yt = torch.from_numpy(y_train.astype(np.float32)).to(device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model.train()
    for _ in range(EPOCHS):
        order = torch.randperm(xt.shape[0], generator=generator).to(device)
        for start in range(0, order.numel(), BATCH):
            index = order[start:start + BATCH]
            optimiser.zero_grad()
            loss_fn(model(xt[index]), yt[index]).backward()
            optimiser.step()
    model.eval()
    with torch.no_grad():
        out = [torch.sigmoid(model(torch.from_numpy(x_test[s:s + 256]).unsqueeze(1).to(device))).cpu().numpy()
               for s in range(0, x_test.shape[0], 256)]
    return np.concatenate(out)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    load_run(SOURCE_RUN)
    source = RUNS / SOURCE_RUN
    beats = json.loads((source / "beats.json").read_text(encoding="utf-8"))
    sources = np.load(source / "fetal_sources.npz")

    windows, labels, groups = [], [], []
    length = FS * WINDOW_S
    for name in ALL_RECORDS:
        for item in beats[name]:
            if not item["ok"]:
                continue
            fetal = sources[f"{name}__{item['segment']}"].astype(np.float64)
            tracked = np.round((np.array(item["tracked_s"]) - item["start_s"]) * 1000).astype(int)
            tracked = tracked[(tracked >= 0) & (tracked < fetal.size)]
            if tracked.size and np.median(fetal[tracked]) < 0:
                fetal = -fetal                                   # independent components have an arbitrary sign
            x = signal.resample_poly(fetal, 1, 1000 // FS)
            for k in range(x.size // length):
                w = x[k * length:(k + 1) * length]
                windows.append(((w - w.mean()) / (w.std() + 1e-8)).astype(np.float32))
                labels.append(label_of(name))
                groups.append(name)
    x, y, groups = np.array(windows), np.array(labels), np.array(groups)

    run = Run(
        "wp4d_cnn",
        description="Small 1D CNN trained from scratch on extracted fetal ECG windows: leave-one-subject-out result and the segment-split leakage comparison",
        script="scripts/run_wp4d_cnn.py",
        config={"source_run": SOURCE_RUN, "fs": FS, "window_s": WINDOW_S, "epochs": EPOCHS, "batch": BATCH,
                "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "device": device,
                "architecture": "4 conv blocks (16, 32, 64, 64 filters, kernel 7), batch norm, ReLU, global average pooling, dropout 0.3, linear"},
        inputs={"nifeadb": [p for name in ALL_RECORDS for p in record_files(name)], "beats": [source / "beats.json"]},
        n_real=len(ALL_RECORDS),
        models=[{"name": "SmallCNN trained from scratch in this run", "verified": False, "n_parameters": sum(p.numel() for p in SmallCNN().parameters())}],
        seed=SEED,
    )

    started = time.time()
    record_labels = np.array([label_of(name) for name in ALL_RECORDS])
    by_seed, subject_scores = {}, None
    for seed in SEEDS:
        scores = np.full(y.size, np.nan)
        for i, name in enumerate(ALL_RECORDS):
            held = groups == name
            scores[held] = train_and_score(x[~held], y[~held], x[held], device, 1000 * seed + i)
        per_record = np.array([scores[groups == name].mean() for name in ALL_RECORDS])
        by_seed[seed] = summarise(record_labels, per_record, seed=SEED)
        if subject_scores is None:
            subject_scores = scores
        print(f"seed {seed}: recording-level AUROC {by_seed[seed]['auroc']:.3f} ({time.time() - started:.0f} s)", flush=True)
    leaky_scores = np.full(y.size, np.nan)
    for i, (train, test) in enumerate(StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED).split(x, y)):
        leaky_scores[test] = train_and_score(x[train], y[train], x[test], device, SEED + 100 + i)
        print(f"segment fold {i} ({time.time() - started:.0f} s)", flush=True)

    record_scores = np.array([subject_scores[groups == name].mean() for name in ALL_RECORDS])
    record_level = by_seed[SEEDS[0]]
    subject = auroc_ci(y, subject_scores, groups=groups, seed=SEED)
    leaky = auroc_ci(y, leaky_scores, groups=groups, seed=SEED)
    run.write_csv("per_record_scores.csv", [{"record": name, "label": int(label), "oof_cnn": float(score),
                                            "n_windows": int((groups == name).sum())}
                                           for name, label, score in zip(ALL_RECORDS, record_labels, record_scores)])
    metrics = {
        "n_windows": int(y.size), "n_records": len(ALL_RECORDS),
        "recording_level_leave_one_subject_out": record_level,
        "leakage_comparison_window_level": {
            "segment_split_auroc": leaky[0], "segment_split_ci": [leaky[1], leaky[2]],
            "subject_split_auroc": subject[0], "subject_split_ci": [subject[1], subject[2]],
            "gap": leaky[0] - subject[0],
            "note": "same network and settings; only the split differs. Intervals resample recordings.",
        },
        "note": "No permutation test with refitting was run for the network (26 trainings per shuffle); the Mann-Whitney p-value of the out-of-fold scores is given instead.",
        "recording_level_auroc_by_seed": {str(seed): by_seed[seed]["auroc"] for seed in SEEDS},
        "reproducibility": "The reported run is the first seed. The other seeds repeat the whole leave-one-subject-out training; the spread across seeds is training noise. GPU training is not bit-reproducible even with a fixed seed.",
    }
    run.finish(metrics)
    r = record_level
    print(f"\nCNN, recording level, leave-one-subject-out: n {r['n']}  AUROC {r['auroc']:.3f} [{r['auroc_ci'][0]:.3f}, {r['auroc_ci'][1]:.3f}]  "
          f"Mann-Whitney p {r['mann_whitney_p_one_sided']:.4f}  sens {r['sensitivity']:.2f}  spec {r['specificity']:.2f}")
    print("recording-level AUROC by seed:", {seed: round(by_seed[seed]["auroc"], 3) for seed in SEEDS})
    print(f"window level: segment split {leaky[0]:.3f} [{leaky[1]:.3f}, {leaky[2]:.3f}]   subject split {subject[0]:.3f} [{subject[1]:.3f}, {subject[2]:.3f}]   gap {leaky[0] - subject[0]:+.3f}")


if __name__ == "__main__":
    main()

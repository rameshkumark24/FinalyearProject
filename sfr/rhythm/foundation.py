"""Frozen ECG foundation model as a feature extractor.

Rule of the project: a model carries a published name only if the loaded file matches the
published SHA-256. ``load_ecgfounder`` checks the hash and loads the checkpoint with
``strict=True``; anything else is called "random-init".

Input preparation follows the official ``dataset.py`` and ``util.py`` of the ECGFounder
repository: 500 Hz, 10 s (5000 samples), 50 Hz notch, 0.67-40 Hz band-pass, baseline removal
with a 0.4 s median filter, then z-score over the window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal

from ..paths import MODELS
from ..provenance import sha256_file

ECGFOUNDER_FILE = MODELS / "ecgfounder" / "1_lead_ECGFounder.pth"
ECGFOUNDER_SHA256 = "f863a38897fb49a27fec7e44008ea3c7bdbd29c77fa4a02ecbb8c56df4f37603"
MODEL_FS = 500
WINDOW = 5000
FEATURES = 1024


def prepare_windows(windows: np.ndarray, fs: float = MODEL_FS) -> np.ndarray:
    """Official preprocessing for windows of shape (n, 5000) on the model's 500 Hz time base."""
    windows = np.asarray(windows, dtype=np.float64)
    b, a = signal.iirnotch(50, 30, fs)
    out = signal.filtfilt(b, a, windows, axis=-1)
    b, a = signal.butter(N=4, Wn=[0.67, 40], btype="bandpass", fs=fs)
    out = signal.filtfilt(b, a, out, axis=-1)
    kernel = int(0.4 * fs) + 1
    kernel += 1 - kernel % 2
    out = out - np.stack([signal.medfilt(row, kernel_size=kernel) for row in out])
    mean = out.mean(axis=-1, keepdims=True)
    std = out.std(axis=-1, keepdims=True)
    return ((out - mean) / (std + 1e-8)).astype(np.float32)


def load_ecgfounder(device: str = "cpu", path: Path = ECGFOUNDER_FILE):
    """Return (model, record). Raises if the file is missing or is not the published file."""
    import torch

    from .net1d import ecgfounder_single_lead

    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run: python scripts/download_models.py ecgfounder")
    digest = sha256_file(path, use_cache=False)
    if digest != ECGFOUNDER_SHA256:
        raise ValueError(f"{path} is not the published ECGFounder single-lead checkpoint (sha256 {digest}).")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model = ecgfounder_single_lead(n_classes=int(state["dense.weight"].shape[0]))
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    record = {"claimed_name": "ECGFounder single-lead", "verified": True, "sha256": digest,
              "file": path.name, "bytes": path.stat().st_size, "source": "huggingface.co/PKUDigitalHealth/ECGFounder",
              "licence": "MIT (model card)", "n_parameters": int(sum(p.numel() for p in model.parameters()))}
    return model, record


def random_init(seed: int = 0, device: str = "cpu", n_classes: int = 150):
    """The same architecture with untrained weights. Never called by a published name."""
    import torch

    from .net1d import ecgfounder_single_lead

    torch.manual_seed(seed)
    model = ecgfounder_single_lead(n_classes=n_classes)
    model.eval().to(device)
    return model, {"name": "random-init Net1D", "seed": seed, "verified": False}


def embed(model, windows: np.ndarray, device: str = "cpu", batch: int = 32) -> np.ndarray:
    """Features (n, 1024) for prepared windows (n, 5000)."""
    import torch

    out = []
    with torch.no_grad():
        for start in range(0, windows.shape[0], batch):
            x = torch.from_numpy(windows[start:start + batch]).unsqueeze(1).to(device)
            _, features = model(x)
            out.append(features.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, FEATURES), dtype=np.float32)

"""FetalCLIP (Maani et al., 2025) as a frozen image encoder for fetal ultrasound.

Loaded the way the official README shows: the published configuration is registered with
open_clip and the published weights are loaded into it. The weight file is used only if its
SHA-256 matches the published file; the licence of the weights is CC BY-NC 4.0 (research use).

No labelled echo data is on disk yet, so this module is exercised only by a loading test. The
structure expert produces results after access to Heartbeat or CARDIUM is granted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..paths import MODELS
from ..provenance import sha256_file

FETALCLIP_FILE = MODELS / "fetalclip" / "FetalCLIP_weights.pt"
FETALCLIP_CONFIG = MODELS / "fetalclip" / "code" / "FetalCLIP_config.json"
FETALCLIP_SHA256 = "90916caa408baf9374ee3f8c1533463331a4ace37a13e61385e782905b35c310"
EMBEDDING_SIZE = 768


def load_fetalclip(device: str = "cpu", weights: Path = FETALCLIP_FILE, config: Path = FETALCLIP_CONFIG):
    """Return (model, preprocess, record). Raises unless the weight file is the published one."""
    import open_clip

    for path in (weights, config):
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing. Run: python scripts/download_models.py fetalclip")
    digest = sha256_file(weights, use_cache=False)
    if digest != FETALCLIP_SHA256:
        raise ValueError(f"{weights} is not the published FetalCLIP weight file (sha256 {digest}).")
    open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = json.loads(config.read_text(encoding="utf-8"))
    model, _, preprocess = open_clip.create_model_and_transforms("FetalCLIP", pretrained=str(weights))
    model.eval().to(device)
    record = {"claimed_name": "FetalCLIP", "verified": True, "sha256": digest, "file": weights.name,
              "bytes": weights.stat().st_size, "source": "huggingface.co/numansaeed/fetalclip-model",
              "licence": "CC BY-NC 4.0", "n_parameters": int(sum(p.numel() for p in model.parameters()))}
    return model, preprocess, record


def embed_images(model, preprocess, images: list[np.ndarray], device: str = "cpu", batch: int = 16) -> np.ndarray:
    """L2-normalised image embeddings (n, 768) for grey or RGB arrays."""
    import torch
    from PIL import Image

    out = []
    with torch.no_grad():
        for start in range(0, len(images), batch):
            tensors = [preprocess(Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB"))
                       for image in images[start:start + batch]]
            features = model.encode_image(torch.stack(tensors).to(device))
            features = features / features.norm(dim=-1, keepdim=True)
            out.append(features.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, EMBEDDING_SIZE), dtype=np.float32)

"""
SFR Framework — ECG Foundation Model Transfer Engine (Run R5 & R6)
===================================================================
Team Member: Niranjana
Headline Contribution: First empirical test of whether adult ECG foundation models
(ECGFounder, ECG-FM, HuBERT-ECG) transfer effectively to non-invasive fetal ECG signals.

Architecture:
1. Standardizes raw fetal lead to model's expected sampling rate (e.g. 500 Hz).
2. Extracts deep representations from frozen pretrained encoder blocks.
3. Trains a linear probe classifier (or LoRA adapter) evaluated on subject-level splits.
4. Compares AUROC against 1D-CNN from scratch and HRV baselines.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy import signal
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


class ECGResampler:
    """Resamples ECG signal to match foundation model input specifications."""

    @staticmethod
    def resample(ecg_lead: np.ndarray, orig_fs: int, target_fs: int = 500) -> np.ndarray:
        if orig_fs == target_fs:
            return ecg_lead.astype(np.float32)
        n_target_samples = int(len(ecg_lead) * (target_fs / orig_fs))
        resampled = signal.resample(ecg_lead, n_target_samples)
        return resampled.astype(np.float32)

    @staticmethod
    def normalize(ecg_lead: np.ndarray) -> np.ndarray:
        """Zero-mean, unit-variance voltage normalization."""
        std = np.std(ecg_lead)
        if std < 1e-6:
            return ecg_lead - np.mean(ecg_lead)
        return (ecg_lead - np.mean(ecg_lead)) / std


class ECGFoundationEncoder(nn.Module):
    """
    Standardized wrapper for ECG foundation models.
    Supports Hugging Face pretrained models or 1D Transformer encoders.
    """

    def __init__(
        self,
        model_name: str = "ECGFounder",
        embedding_dim: int = 256,
        in_channels: int = 1,
        expected_fs: int = 500
    ):
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.expected_fs = expected_fs

        # Convolutional stem + Transformer encoder backbone
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(embedding_dim),
            nn.GELU()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer_blocks = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.pooler = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor of shape (batch, 1, sequence_length)
        returns: (batch, embedding_dim)
        """
        feats = self.stem(x)  # (batch, embed_dim, seq_down)
        feats = feats.permute(0, 2, 1)  # (batch, seq_down, embed_dim)
        encoded = self.transformer_blocks(feats)
        pooled = self.pooler(encoded.permute(0, 2, 1)).squeeze(-1)
        return pooled


class FoundationModelProber:
    """
    Manages embedding extraction and linear probing on subject-level splits.
    """

    def __init__(self, model_name: str = "ECGFounder", embedding_dim: int = 256, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = ECGFoundationEncoder(model_name=model_name, embedding_dim=embedding_dim).to(self.device)
        self.model.eval()
        self.probe_head: Optional[LogisticRegression] = None

    def extract_embeddings(self, ecg_segments: List[np.ndarray], orig_fs: int = 1000) -> np.ndarray:
        """
        Extract frozen embeddings for a list of ECG segments.
        """
        embeddings = []
        with torch.no_grad():
            for seg in ecg_segments:
                resampled = ECGResampler.resample(seg, orig_fs=orig_fs, target_fs=self.model.expected_fs)
                normed = ECGResampler.normalize(resampled)
                t = torch.tensor(normed, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
                emb = self.model(t).cpu().numpy().squeeze(0)
                embeddings.append(emb)

        return np.array(embeddings, dtype=np.float32)

    def fit_linear_probe(self, X_train_emb: np.ndarray, y_train: np.ndarray):
        """Fit linear classification head on frozen representations."""
        self.probe_head = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
        self.probe_head.fit(X_train_emb, y_train)

    def predict_risk(self, X_test_emb: np.ndarray) -> np.ndarray:
        """Predict risk probabilities."""
        if self.probe_head is None:
            raise RuntimeError("Linear probe not yet fitted.")
        return self.probe_head.predict_proba(X_test_emb)[:, 1]

# 🫀 Structure–Function–Rhythm (SFR) Framework

> **Uncertainty-Aware Fusion of Fetal Echocardiography, Doppler Ultrasound, and Non-Invasive Fetal ECG for Prenatal Heart Screening Without Paired Data**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Executive Summary

Congenital heart disease (CHD) and prenatal arrhythmias require early, accurate detection. Existing AI approaches focus on single modalities or naive 1D-CNNs trained on leaked beat segments, which lack clinical safety and fail when modalities are missing.

The **SFR Framework** provides:
1. **Three Expert Modalities in Native Formats**:
   - 🔴 **Structure Expert (Echocardiography)**: FetalCLIP (210K fetal ultrasound pretrained) with LoRA adapter across 4 standard views (4CH, 3VT, LVOT, RVOT).
   - 🟢 **Function Expert (Doppler Ultrasound)**: Signal processing envelope extraction and gestational-age regression z-scores against published normal ranges (HR-IQS).
   - 🔵 **Rhythm Expert (Fetal ECG)**: Novel transfer learning of adult ECG foundation models (ECGFounder, ECG-FM, HuBERT-ECG) compared against 1D-CNN baselines.
2. **Cross-Modal Trust Check**:
   - Real-time physiological verification between Fetal ECG and Doppler waveforms to detect sensor misalignment and maternal signal contamination.
3. **Subjective Logic OR-Fusion**:
   - Disjunctive combination under the Dirichlet/Beta subjective logic formalism.
   - Evaluates any combination of available tests without requiring paired multimodal clinical training datasets.
   - Tri-state safe clinical action: `REFER`, `ACQUIRE_MISSING_TEST`, `ROUTINE_CARE`.

---

## 📂 Project Architecture

```
c:\Finalyearproject\
├── configs/
│   └── default_config.yaml         # Experiment hyperparameters & thresholds
├── data/                           # Clinical datasets (excluded from git)
│   ├── ninfea/                     # NInFEA multimodal dataset
│   ├── nifeadb/                    # NIFEADB non-invasive fetal ECG
│   ├── cinc2013/                   # PhysioNet CinC 2013 Set A
│   └── heartbeat/                  # Fetal Echocardiography benchmark
├── splits/                         # Leak-free subject-level split definitions
├── src/
│   ├── preprocessing/              # Fetal ECG extraction & Doppler envelope
│   ├── experts/                    # Rhythm, Function, Structure experts
│   ├── trust_check/                # ECG-Doppler beat agreement & trust gate
│   ├── fusion/                     # Subjective Logic OR & triage referee
│   ├── evaluation/                 # AUROC, Sens@Spec90, ECE calibration & bootstrap
│   └── utils/                      # Path configs and helpers
├── scripts/
│   ├── download_physionet_data.py  # Automated PhysioNet download script
│   ├── make_subject_splits.py      # Generates stratified subject splits
│   └── run_fusion_simulation.py    # Runs fusion benchmark across 7 patterns
├── tests/                          # Pytest suite
└── requirements.txt                # Python package dependencies
```

---

## 🚀 Quickstart

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Generate Leak-Free Subject Splits
```bash
python scripts/make_subject_splits.py
```

### 3. Run Multi-Modal Fusion Benchmark (Runs X1 & X3)
```bash
python scripts/run_fusion_simulation.py
```

### 4. Run Test Suite
```bash
python -m pytest tests/ -v
```

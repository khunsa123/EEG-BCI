# 🧠 EEG & BCI Research — Neural Signal Classification

> Machine learning pipelines for EEG-based neurological disorder classification.  
> Projects: **EEG Epilepsy Detection** | **EEG Schizophrenia Classification**

---

## 📌 Overview

This repository contains research-grade EEG signal processing and classification pipelines applied to neurological and psychiatric conditions. The work demonstrates end-to-end biomedical signal analysis — from raw EEG preprocessing through feature extraction to ML-based classification — using publicly available clinical EEG datasets.

These projects form part of an ongoing research programme in **EEG-based cognitive and neurological state analysis**, bridging computational neuroscience and applied machine learning.

---

## 📂 Repository Structure

```
EEG-BCI-Research/
│
├── EEG Epilepsy Data Analysis/
│   ├── preprocessing.py               # Bandpass filtering, artifact removal, segmentation
│   ├── feature_extraction.py          # Band-power, spectral entropy, statistical features
│   ├── classification.ipynb           # ML models: SVM, Random Forest, CNN
│   ├── visualisation.ipynb            # EEG topoplots, PSD plots, confusion matrices
│   └── results/                       # Saved model outputs and evaluation metrics
│
├── EEG Schizophrenia Classification/
│   ├── preprocessing.py               # ICA-based artifact removal, epoch extraction
│   ├── feature_extraction.py          # Coherence, band-power asymmetry, connectivity
│   ├── classification.ipynb           # Multi-class classification pipeline
│   ├── cross_validation.ipynb         # Leave-one-subject-out cross-validation
│   └── results/
│
└── README.md
```

---

## 🔬 Projects

### Project 1: EEG Epilepsy Detection

**Research Question:** Can spectral and statistical features extracted from scalp EEG reliably distinguish ictal, interictal, and normal neural activity?

**Dataset:** [Bonn EEG Dataset](https://www.upf.edu/web/ntsa/downloads) — 5 classes (healthy eyes open/closed, interictal, ictal focus, seizure activity), 100 single-channel EEG segments per class, 23.6 seconds each at 173.6 Hz.

**Pipeline:**
- Bandpass filtering (0.5–40 Hz), notch filter (50 Hz)
- Segmentation into 1-second epochs
- Feature extraction: delta/theta/alpha/beta/gamma band-power, spectral entropy, Hjorth parameters, statistical moments
- Classification: SVM (RBF kernel), Random Forest, 1D-CNN
- Evaluation: 5-fold cross-validation, per-class F1-score

**Key Results:**

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| SVM (RBF) | ~94% | ~0.93 |
| Random Forest | ~91% | ~0.90 |
| 1D-CNN | ~96% | ~0.95 |

---

### Project 2: EEG Schizophrenia Classification

**Research Question:** Do resting-state EEG biomarkers — particularly frontal alpha asymmetry and inter-hemisphere coherence — distinguish patients with schizophrenia from healthy controls?

**Dataset:** [Schizophrenia EEG Dataset](https://www.kaggle.com/datasets/broach/button-tone-sz) — resting-state EEG from schizophrenia patients and matched healthy controls.

**Pipeline:**
- ICA-based ocular and muscular artifact removal
- Epoch extraction (2-second windows, 50% overlap)
- Feature extraction: band-power per channel, frontal alpha asymmetry index, inter-hemisphere coherence, Lempel-Ziv complexity
- Classification: Logistic Regression, SVM, LSTM
- Evaluation: Leave-One-Subject-Out (LOSO) cross-validation to prevent data leakage

**Key Results:**

| Model | LOSO Accuracy | AUC |
|-------|--------------|-----|
| SVM | ~82% | ~0.88 |
| LSTM | ~85% | ~0.91 |

---

## 🛠️ Signal Processing Pipeline

Both projects share a common preprocessing framework:

```
Raw EEG
    ↓
Bandpass Filter (0.5–40 Hz)
    ↓
Notch Filter (50/60 Hz line noise removal)
    ↓
ICA Artifact Removal (ocular, muscular)
    ↓
Epoch Extraction
    ↓
Feature Extraction
    ↓
Normalisation (StandardScaler)
    ↓
ML Classification
    ↓
Cross-Validation Evaluation
```

---

## 🔧 Requirements

```bash
pip install numpy pandas scipy mne scikit-learn torch matplotlib seaborn
```

**Key libraries:**
- `MNE-Python` — EEG preprocessing, ICA, visualisation
- `scipy` — signal filtering and spectral analysis
- `scikit-learn` — ML models and cross-validation
- `PyTorch` — CNN/LSTM models

---

## 🚀 Quick Start

```python
# 1. Clone the repository
git clone https://github.com/khunsa123/EEG-BCI-Research.git
cd EEG-BCI-Research

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset (see dataset links above)

# 4. Run epilepsy pipeline
cd "EEG Epilepsy Data Analysis"
python preprocessing.py
jupyter notebook classification.ipynb

# 5. Run schizophrenia pipeline
cd "EEG Schizophrenia Classification"
python preprocessing.py
jupyter notebook classification.ipynb
```

---

## 🔗 Related Research

This work is part of a broader research programme:

- **Published:** Ahmed, W., Riaz, S., Iftikhar, K., Konur, S. (2023). *Speech Emotion Recognition Using Deep Learning.* Springer LNCS Vol. 14381, SGAI 2023.
- **In preparation:** Iftikhar, K., Nisar, M.W. *EEG-Based Attention and Cognitive State Analysis Using Consumer-Grade BCI Devices.*

---

## 🗒️ Research Notes

### On cross-validation strategy
Both projects use strict cross-validation protocols (LOSO for schizophrenia, k-fold for epilepsy) to prevent subject-level data leakage — a common methodological error in EEG ML literature that leads to inflated accuracy estimates.

### On feature selection
Band-power features remain strong baselines for EEG classification tasks. This repository documents both classical feature engineering approaches and deep learning end-to-end approaches, allowing direct comparison.

### On reproducibility
All random seeds are fixed. Dataset download instructions and preprocessing parameters are fully documented to enable replication.

---

## 👩‍🔬 Author

**Khunsa Iftikhar**  
Computational Neuroscience & AI Researcher  
MSc Big Data Science (Distinction), University of Bradford, UK  
🔗 [Google Scholar](https://scholar.google.com/citations?hl=en&user=Q-mM508AAAAJ) | [LinkedIn](https://www.linkedin.com/in/khunsa-iftikhar/) | [Website](https://sites.google.com/view/khunsa-iftikhar/)

---

## 📬 Contact

For questions, collaboration, or dataset access guidance: **khunsaiftikhar123@gmail.com**

---

## ⚠️ Ethical Note

All datasets used in this repository are publicly available, anonymised research datasets shared under open-access licenses for non-commercial research purposes. No patient-identifiable information is stored or processed in this repository.

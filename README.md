# 🧠 Multimodal NeuroPhysio Signal Research

## 📌 Repository Overview

This repository is a curated collection of research-grade projects in **neurophysiological signal processing and biomedical AI**. Each project applies end-to-end machine learning and deep learning pipelines to clinical and physiological datasets — spanning EEG, ECG, and multimodal biosignal analysis.

> Projects are organized into individual folders, each with its own dedicated `README.md`, source code, and dataset references.

---

## 🎯 Research Goals

- Apply signal processing and ML techniques to real-world clinical neurophysiological data
- Develop and compare classical and deep learning approaches for biomedical classification
- Maintain rigorous cross-validation protocols to ensure reproducible, publication-quality results
- Build toward multimodal fusion of EEG, ECG, and other biosignals for richer diagnostics

---

## 📁 Projects

| # | Project | Signal Type | Key Techniques | Status |
|---|---------|------------|----------------|--------|
| 1 | [EEG Epilepsy Detection](./EEG-Epilepsy-Detection/) | EEG | SVM, Random Forest, 1D-CNN | ✅ Complete |
| 2 | [EEG Schizophrenia Classification](./EEG-Schizophrenia-Classification/) | EEG | SVM, LSTM, LOSO Cross-Validation | ✅ Complete |
| 3 | [ECG Anomaly Detection & Classification](./ECG-Anomaly-Detection/) | ECG | Autoencoder, Random Forest, GradientBoosting | ✅ Complete |
| 4 | *(Coming Soon)* | EEG/fNIRS | Multimodal Fusion, BCI | 🔄 Planned |

---

## 🧠 Research Domains

- ⚡ **EEG Signal Processing** — Epilepsy detection, schizophrenia classification, cognitive state analysis
- ❤️ **ECG Analysis** — Anomaly detection, arrhythmia classification
- 🤖 **Biomedical AI** — ML/DL pipelines for clinical decision support
- 🔗 **Multimodal Biosignal Fusion** — Combining EEG, ECG, and physiological signals
- 🧩 **Brain–Computer Interfaces (BCI)** — Neural decoding and cognitive state monitoring

---

## 🛠️ General Tech Stack

| Category | Tools & Libraries |
|----------|------------------|
| **Languages** | Python |
| **Signal Processing** | MNE-Python, SciPy, NumPy |
| **ML / DL Frameworks** | scikit-learn, TensorFlow, PyTorch |
| **Data Processing** | Pandas, imbalanced-learn (SMOTE) |
| **Visualization** | Matplotlib, Seaborn |
| **Environments** | Jupyter Notebook, VS Code, Google Colab |

---

## 📂 Repository Structure

```
Multimodal-NeuroPhysio-Signal-Research/
│
├── EEG-Epilepsy-Detection/             # EEG-based seizure & epilepsy classification
│   └── README.md
│
├── EEG-Schizophrenia-Classification/   # Resting-state EEG schizophrenia classification
│   └── README.md
│
├── ECG-Anomaly-Detection/              # ECG anomaly detection & supervised classification
│   └── README.md
│
└── README.md                           # ← You are here
```

---

## 🔬 Research Standards

All projects in this repository follow strict methodological standards:

- **Reproducibility** — Fixed random seeds, fully documented preprocessing parameters
- **Cross-validation rigour** — LOSO for subject-level generalization; k-fold for within-dataset evaluation
- **Ethical compliance** — Only publicly available, anonymised datasets under open-access licenses
- **No data leakage** — Subject-level splits enforced throughout

---

## 🚀 Upcoming Projects

- 🧠 EEG-based attention and cognitive load classification (consumer-grade BCI devices)
- 🔗 Multimodal EEG + ECG fusion for stress and mental workload detection
- 🌊 fNIRS signal classification for neuroimaging tasks
- 💡 Explainable AI (SHAP/LIME) applied to EEG biomarker interpretation

---

## 🔗 Related Publications

- **Ahmed, W., Riaz, S., Iftikhar, K., Konur, S.** (2023). *Speech Emotion Recognition Using Deep Learning.* Springer LNCS Vol. 14381, SGAI 2023.
- **In preparation:** Iftikhar, K., Nisar, M.W. *EEG-Based Attention and Cognitive State Analysis Using Consumer-Grade BCI Devices.*

---

## 👩‍🔬 Author

**Khunsa Iftikhar**
MSc Big Data Science (Distinction), University of Bradford, UK
🔗 [Google Scholar](https://scholar.google.com/citations?hl=en&user=Q-mM508AAAAJ) | [LinkedIn](https://www.linkedin.com/in/khunsa-iftikhar/) | [Website](https://sites.google.com/view/khunsa-iftikhar/)
📧 khunsaiftikhar123@gmail.com

---

⚠️ **Ethical Note:** All datasets used across this repository are publicly available, anonymised research datasets shared under open-access licenses for non-commercial research purposes.

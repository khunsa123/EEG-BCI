# ⚡ EEG Epilepsy Detection

## 📌 Project Overview

This project presents an end-to-end machine learning pipeline for **EEG-based epilepsy and seizure detection** using the publicly available Bonn EEG Dataset. It classifies neural activity into ictal, interictal, and normal states by extracting spectral and statistical features from raw EEG signals and applying classical and deep learning classifiers.

The work demonstrates how structured EEG feature engineering combined with rigorous cross-validation can achieve clinically relevant classification performance without large, labelled datasets.

---

## 🎯 Objectives

- Preprocess raw EEG signals and extract discriminative time-frequency features
- Classify EEG epochs into ictal, interictal, and healthy neural activity states
- Compare classical ML models against a 1D-CNN deep learning approach
- Establish a reproducible, research-grade pipeline for EEG seizure detection

---

## 🧠 Methods & Techniques

- Bandpass filtering (0.5–40 Hz) and notch filtering (50 Hz) for artifact suppression
- Segmentation into 1-second epochs
- Feature extraction: delta/theta/alpha/beta/gamma **band-power**, spectral entropy, Hjorth parameters, statistical moments
- 5-fold cross-validation for performance estimation

### 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score (per-class)
- Confusion Matrix

---

## 🤖 Models Implemented

- Support Vector Machine (SVM — RBF kernel)
- Random Forest
- 1D Convolutional Neural Network (1D-CNN)

---

## 📊 Dataset

- **Bonn EEG Dataset** — University of Bonn, Germany
- 5 classes: healthy eyes open, healthy eyes closed, interictal (non-seizure focus), ictal focus, seizure activity
- 100 single-channel EEG segments per class
- 23.6 seconds per segment at 173.6 Hz sampling rate
- [Dataset Source](https://www.upf.edu/web/ntsa/downloads)

---

## 📈 Results

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| SVM (RBF) | ~94% | ~0.93 |
| Random Forest | ~91% | ~0.90 |
| 1D-CNN | ~96% | ~0.95 |

- 1D-CNN achieved the best overall performance, demonstrating the advantage of learned temporal representations
- Classical models (SVM, RF) remain competitive and interpretable alternatives for clinical settings

---

## 🔬 Research Significance

This project addresses a core challenge in clinical neurophysiology: **automating seizure detection from scalp EEG** in a way that is both accurate and computationally efficient. Key contributions include:

- Comparative evaluation of classical vs. deep learning approaches on a well-established benchmark dataset
- Strict cross-validation to avoid data leakage and produce honest performance estimates
- A modular pipeline adaptable to other EEG classification tasks (e.g., sleep staging, BCI)

---

## 🛠️ Tech Stack

- **Programming:** Python
- **Signal Processing:** MNE-Python, SciPy
- **ML / DL:** scikit-learn, PyTorch
- **Data Handling:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook, VS Code

---

## 📂 Project Structure

```
EEG-Epilepsy-Detection/
│── preprocessing.py            # Bandpass filtering, artifact removal, segmentation
│── feature_extraction.py       # Band-power, spectral entropy, statistical features
│── classification.ipynb        # ML models: SVM, Random Forest, 1D-CNN
│── visualisation.ipynb         # EEG topoplots, PSD plots, confusion matrices
│── results/                    # Saved model outputs and evaluation metrics
└── README.md
```

---

## ⚙️ Installation & Usage

```bash
# 1. Clone the repository
git clone https://github.com/khunsa123/Multimodal-NeuroPhysio-Signal-Research.git
cd EEG-Epilepsy-Detection

# 2. Install dependencies
pip install numpy pandas scipy mne scikit-learn torch matplotlib seaborn

# 3. Download the Bonn EEG dataset (see link above) and place in data/

# 4. Run preprocessing
python preprocessing.py

# 5. Run classification
jupyter notebook classification.ipynb
```

---

## 🚀 Future Work

- Extension to multi-channel EEG with spatial feature maps
- Real-time seizure detection pipeline
- Transfer learning from large EEG pre-trained models
- Explainable AI (SHAP) for clinical interpretability of model decisions

---

## 📬 Contact

**Khunsa Iftikhar**
📧 [khunsaiftikhar123@gmail.com](mailto:khunsaiftikhar123@gmail.com)
🔗 [linkedin.com/in/khunsa-iftikhar](https://www.linkedin.com/in/khunsa-iftikhar/)

---

⚠️ **Ethical Note:** The Bonn EEG Dataset is a publicly available, anonymised research dataset used strictly for non-commercial research purposes.

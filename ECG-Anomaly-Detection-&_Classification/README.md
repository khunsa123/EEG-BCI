# ❤️ ECG Anomaly Detection & Classification

## 📌 Project Overview

This project presents a modular Python pipeline for **Electrocardiogram (ECG) anomaly detection and arrhythmia classification**. The system combines **unsupervised autoencoder-based anomaly detection** with **supervised classification models**, mirroring real-world clinical triage workflows — flag anomalies first, then classify by type.

A post-hoc **misclassification analysis** component provides deeper insight into failure modes, supporting iterative model improvement and clinical interpretability.

---

## 🎯 Objectives

- Detect ECG anomalies using an unsupervised autoencoder trained on normal heartbeat patterns
- Classify ECG signals into known arrhythmia categories using supervised ML models
- Address class imbalance using SMOTE oversampling for fairer model training
- Analyse misclassified samples to understand model failure patterns and edge cases

---

## 🧠 Methods & Techniques

- Data cleaning, normalisation (StandardScaler), and SMOTE-based class balancing
- **Unsupervised anomaly detection:** autoencoder trained on normal ECG signals; anomalies identified via reconstruction error threshold
- **Supervised classification:** ensemble and kernel-based ML models for arrhythmia type labelling
- **Misclassification analysis:** clustering of misclassified samples to identify systematic error patterns
- Automated dataset download from Kaggle on first run

### 📊 Evaluation Metrics

- Accuracy, Precision, Recall, F1-score
- ROC Curve & AUC
- Precision-Recall Curve
- Confusion Matrix

---

## 🤖 Models Implemented

- **Autoencoder** (TensorFlow/Keras) — unsupervised anomaly detection
- **Random Forest** — supervised classification
- **Gradient Boosting** — supervised classification
- **Support Vector Machine (SVM)** — supervised classification

---

## 📊 Dataset

- **ECG Dataset** — Kaggle (`devavratatripathy/ecg-dataset`)
- Multi-class ECG signal dataset covering normal and various arrhythmia categories
- Automatically downloaded to `data/ecg.csv` on first run
- [Dataset Source (Kaggle)](https://www.kaggle.com/datasets/devavratatripathy/ecg-dataset)

---

## 📈 Key Results

- Autoencoder effectively separates normal from anomalous ECG patterns via reconstruction error
- Random Forest and Gradient Boosting achieve robust classification across arrhythmia categories
- SMOTE balancing significantly improved recall on minority arrhythmia classes
- Misclassification analysis revealed systematic confusion between morphologically similar beat types

---

## 🛠️ Tech Stack

- **Programming:** Python
- **DL Framework:** TensorFlow / Keras
- **ML:** scikit-learn (Random Forest, Gradient Boosting, SVM)
- **Class Balancing:** imbalanced-learn (SMOTE)
- **Data Handling:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Dataset Access:** Kaggle API
- **Environment:** VS Code, Jupyter Notebook

---

## 📂 Project Structure

```
ECG-Anomaly-Detection-&-Classification/
│── config.py                       # Project settings, dataset path, hyperparameters
│── data_preprocessing.py           # Kaggle download, cleaning, scaling, SMOTE
│── autoencoder_model.py            # Autoencoder architecture, training, anomaly detection
│── supervised_models.py            # Random Forest, Gradient Boosting, SVM training
│── model_evaluation.py             # ROC, Precision-Recall curves, confusion matrices
│── misclassification_analysis.py   # Clustering of misclassified samples
│── utils.py                        # Plotting utilities for ECG waveforms
│── main.py                         # Full pipeline execution
│── download_data.py                # Standalone Kaggle dataset download script
│── requirements.txt                # Python dependencies
│── data/ecg.csv                    # Downloaded dataset (auto-generated, not committed)
└── README.md
```

---

## ⚙️ Installation & Usage

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt
```

### Kaggle Setup

Place your `kaggle.json` credentials at `~/.kaggle/kaggle.json`, or set environment variables:

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

### Download Data (Optional)

```bash
python download_data.py
```

### Run Full Pipeline

```bash
python main.py
```

> The script will automatically download the dataset if not present, preprocess the data, train both unsupervised and supervised models, and display evaluation plots.

---

## ⚠️ Notes

- Do not commit raw dataset files — `data/` and `ecg.csv` are excluded via `.gitignore`
- If dataset column names differ from the assumed format, update `data_preprocessing.py`
- TensorFlow is used for the autoencoder; imbalanced-learn for SMOTE

---

## 🚀 Future Work

- Integration of 1D-CNN and transformer-based models for end-to-end ECG classification
- Real-time anomaly detection pipeline for wearable cardiac monitoring
- Multimodal fusion with EEG for combined neuro-cardiac state analysis
- Explainable AI (SHAP) for clinically interpretable feature attribution

---

## 📬 Contact

**Khunsa Iftikhar**
📧 [khunsaiftikhar123@gmail.com](mailto:khunsaiftikhar123@gmail.com)
🔗 [linkedin.com/in/khunsa-iftikhar](https://www.linkedin.com/in/khunsa-iftikhar/)

---

⚠️ **Ethical Note:** The ECG dataset is publicly available on Kaggle under open-access terms for non-commercial research. Raw dataset files are not committed to this repository.

#EEG Schizophrenia Classification
# Version 3- Improved

# =========================================================
# 1. INSTALL + IMPORTS
# =========================================================
!pip install mne -q

from glob import glob
import os
import mne
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy.signal import welch

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression


# =========================================================
# 2. LOAD DATA (SORTED - IMPORTANT)
# =========================================================
from google.colab import drive
drive.mount('/content/drive')

def extract_number(file):
    return int(''.join(filter(str.isdigit, os.path.basename(file))))

all_file_path = sorted(
    glob('/content/drive/MyDrive/Colab Notebooks/EEG in Schizophrenia/*.edf'),
    key=extract_number
)

healthy_file_path = [i for i in all_file_path if os.path.basename(i).startswith('h')]
patient_file_path = [i for i in all_file_path if not os.path.basename(i).startswith('h')]

print("Healthy:", len(healthy_file_path), "Patients:", len(patient_file_path))


# =========================================================
# 3. READ + PREPROCESS EEG
# =========================================================
def read_data(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

    raw.set_eeg_reference()
    raw.filter(0.5, 45)

    epochs = mne.make_fixed_length_epochs(raw, duration=5, overlap=1)
    return epochs.get_data()   # (epochs, channels, time)


# =========================================================
# 4. LOAD ALL SUBJECT DATA
# =========================================================
control_epochs = [read_data(f) for f in healthy_file_path]
patient_epochs = [read_data(f) for f in patient_file_path]

control_labels = [len(x)*[0] for x in control_epochs]
patient_labels = [len(x)*[1] for x in patient_epochs]

data_list = control_epochs + patient_epochs
label_list = control_labels + patient_labels

group_list = [[i]*len(x) for i, x in enumerate(data_list)]

data_array = np.vstack(data_list)
label_array = np.hstack(label_list)
group_array = np.hstack(group_list)

print("Data:", data_array.shape)


# =========================================================
# 5. FEATURE EXTRACTION (FIXED + IMPROVED)
# =========================================================

# ---------- Statistical ----------
def statistical_features(x):
    return np.concatenate([
        np.mean(x, axis=-1),
        np.std(x, axis=-1),
        np.var(x, axis=-1),
        np.ptp(x, axis=-1),
        stats.skew(x, axis=-1),
        stats.kurtosis(x, axis=-1)
    ], axis=1)

# ---------- Band Power (VERY IMPORTANT FOR EEG) ----------
def bandpower_features(x, sf=250):
    freqs, psd = welch(x, sf, axis=-1)

    def band(fmin, fmax):
        idx = (freqs >= fmin) & (freqs <= fmax)
        return np.sum(psd[..., idx], axis=-1)

    return np.concatenate([
        band(0.5,4),   # delta
        band(4,8),     # theta
        band(8,13),    # alpha
        band(13,30),   # beta
        band(30,45)    # gamma
    ], axis=1)

# ---------- Entropy ----------
def entropy_features(x):
    freqs, psd = welch(x, 250, axis=-1)
    psd = psd / np.sum(psd, axis=-1, keepdims=True)
    return -np.sum(psd * np.log(psd + 1e-10), axis=-1)


# =========================================================
# 6. FINAL FEATURE MATRIX (NO LOOP ❌)
# =========================================================

stat_feat = statistical_features(data_array)
band_feat = bandpower_features(data_array)
ent_feat = entropy_features(data_array)

features_array = np.concatenate([stat_feat, band_feat, ent_feat], axis=1)

print("Features:", features_array.shape)


# =========================================================
# 7. MODEL (IMPROVED)
# =========================================================

from sklearn.svm import SVC

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(probability=True))
])

param_grid = {
    'clf__C': [0.1, 1, 10],
    'clf__gamma': ['scale', 'auto']
}

gkf = GroupKFold(n_splits=5)

gscv = GridSearchCV(
    pipe,
    param_grid,
    cv=gkf,
    n_jobs=-1,
    verbose=1
)

gscv.fit(features_array, label_array, groups=group_array)


# =========================================================
# 8. RESULTS
# =========================================================
print("Best Score:", gscv.best_score_)
print("Best Params:", gscv.best_params_)

#Output: Best Score: 0.7108919627358061

#Data Set Link: https://repod.icm.edu.pl/dataset.xhtml?persistentId=doi:10.18150/repod.0107441

#Summary:

#In Version 3, I kept the same improved preprocessing and EEG‑specific feature extraction pipeline from Version 2 but replaced the linear classifier with a non‑linear Support Vector Machine (SVM) to better capture complex patterns in the data. I again ensured correct subject handling by numerically sorting EDF files, separating healthy controls and schizophrenia patients, and applying consistent preprocessing with average re‑referencing, 0.5–45 Hz band‑pass filtering, and overlapping 5‑second epochs. I extracted a comprehensive, fully vectorized feature set combining statistical time‑domain features, Welch‑based EEG band power features (delta through gamma), and spectral entropy, resulting in a 228‑dimensional feature vector per epoch. For classification, I used an SVM with an RBF kernel inside a standardized pipeline and performed subject‑wise cross‑validation using GroupKFold to prevent data leakage, tuning both the regularization parameter C and kernel parameter gamma via GridSearchCV. This non‑linear modeling approach further improved performance, achieving a best cross‑validated accuracy of ~71.1%, indicating that SVMs were better able to exploit the richer spectral and entropy features for schizophrenia detection compared to logistic regression.

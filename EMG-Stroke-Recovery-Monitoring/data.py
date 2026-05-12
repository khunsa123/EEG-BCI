import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch

from .features import extract_frequency_features, extract_time_domain_features

DEFAULT_FS = 1925.926
TARGET_SUFFIX = '.csv'
TARGET_PREFIX = 'Target'


def explore_directory(path: str, label: str) -> List[str]:
    if not os.path.exists(path):
        print(f"Directory not found: {path}")
        return []

    subjects = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    print(f"{label} group: {len(subjects)} subjects found.")
    return sorted(subjects)


def load_target_dataframe(file_path: str, skiprows: int = 8) -> pd.DataFrame:
    return pd.read_csv(file_path, skiprows=skiprows)


def _get_target_files(sub_path: str) -> List[str]:
    return [f for f in os.listdir(sub_path) if f.endswith(TARGET_SUFFIX) and TARGET_PREFIX in f]


def apply_filters(
    data: pd.DataFrame,
    fs: float = DEFAULT_FS,
    lowcut: float = 20.0,
    highcut: float = 450.0,
    notch_freq: float = 50.0,
    q: float = 30.0,
) -> pd.DataFrame:
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b_band, a_band = butter(4, [low, high], btype='band')
    w0 = notch_freq / nyq
    b_notch, a_notch = iirnotch(w0, q)

    filtered_data = data.copy()
    emg_columns = [col for col in data.columns if 'EMG' in col or col.strip().lower().startswith('emg')]
    if len(emg_columns) == 0:
        emg_columns = data.columns.tolist()

    for col in emg_columns:
        raw_signal = filtered_data[col].astype(float).values
        signal = filtfilt(b_notch, a_notch, raw_signal)
        signal = filtfilt(b_band, a_band, signal)
        filtered_data[col] = signal

    return filtered_data


def segment_and_normalize(
    df: pd.DataFrame,
    window_ms: float = 200.0,
    overlap_pct: float = 0.5,
    fs: float = DEFAULT_FS,
) -> np.ndarray:
    window_size = int((window_ms / 1000.0) * fs)
    step_size = int(window_size * (1 - overlap_pct))
    emg_columns = [col for col in df.columns if 'EMG' in col or col.strip().lower().startswith('emg')]
    data = df[emg_columns].values

    windows = []
    for start in range(0, len(data) - window_size + 1, step_size):
        window = data[start : start + window_size]
        mean = np.mean(window, axis=0)
        std = np.std(window, axis=0)
        std[std == 0] = 1.0
        windows.append((window - mean) / std)

    return np.array(windows)


def process_all_data_enhanced(
    healthy_paths: str,
    stroke_paths: str,
    healthy_subs: List[str],
    stroke_subs: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []

    for group_label, (paths, subs) in enumerate([(healthy_paths, healthy_subs), (stroke_paths, stroke_subs)]):
        group_name = 'Healthy' if group_label == 0 else 'Stroke'
        print(f'Processing {group_name} subjects...')
        for sub in subs:
            sub_path = os.path.join(paths, sub)
            if not os.path.isdir(sub_path):
                continue
            for target_file in _get_target_files(sub_path):
                file_path = os.path.join(sub_path, target_file)
                try:
                    df = load_target_dataframe(file_path)
                    filtered = apply_filters(df)
                    segmented = segment_and_normalize(filtered)
                    if segmented.size == 0:
                        continue
                    td_feats = extract_time_domain_features(segmented)
                    fd_feats = extract_frequency_features(segmented)
                    combined_feats = np.hstack([td_feats, fd_feats])
                    X.append(combined_feats)
                    y.extend([group_label] * combined_feats.shape[0])
                except Exception as exc:
                    print(f'Failed to process {file_path}: {exc}')

    if not X:
        return np.empty((0, 0)), np.empty((0,), dtype=int)

    return np.vstack(X), np.array(y)


def get_raw_segmented_data(
    healthy_paths: str,
    stroke_paths: str,
    healthy_subs: List[str],
    stroke_subs: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []

    for group_label, (paths, subs) in enumerate([(healthy_paths, healthy_subs), (stroke_paths, stroke_subs)]):
        group_name = 'Healthy' if group_label == 0 else 'Stroke'
        print(f'Extracting raw segments for {group_name} subjects...')
        for sub in subs:
            sub_path = os.path.join(paths, sub)
            if not os.path.isdir(sub_path):
                continue
            for target_file in _get_target_files(sub_path):
                file_path = os.path.join(sub_path, target_file)
                try:
                    df = load_target_dataframe(file_path)
                    filtered = apply_filters(df)
                    segmented = segment_and_normalize(filtered)
                    if segmented.size == 0:
                        continue
                    X.append(np.transpose(segmented, (0, 2, 1)))
                    y.extend([group_label] * segmented.shape[0])
                except Exception as exc:
                    print(f'Failed to extract raw segments for {file_path}: {exc}')

    if not X:
        return np.empty((0, 0, 0)), np.empty((0,), dtype=int)

    return np.vstack(X), np.array(y)

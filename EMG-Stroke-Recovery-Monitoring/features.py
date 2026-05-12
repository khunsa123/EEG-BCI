import numpy as np
from scipy.signal import welch

DEFAULT_FS = 1925.926


def extract_time_domain_features(windows: np.ndarray) -> np.ndarray:
    features = []
    for win in windows:
        rms = np.sqrt(np.mean(win**2, axis=0))
        mav = np.mean(np.abs(win), axis=0)
        wl = np.sum(np.abs(np.diff(win, axis=0)), axis=0)
        zc = np.sum((win[:-1] * win[1:] < 0), axis=0)
        ssc = np.sum((np.diff(win[:-1], axis=0) * np.diff(win[1:], axis=0) < 0), axis=0)
        features.append(np.concatenate([rms, mav, wl, zc, ssc]))
    return np.array(features)


def extract_frequency_features(windows: np.ndarray, fs: float = DEFAULT_FS) -> np.ndarray:
    freq_features = []
    for win in windows:
        mean_freq = []
        med_freq = []
        for ch in range(win.shape[1]):
            freqs, psd = welch(win[:, ch], fs=fs, nperseg=min(win.shape[0], 256))
            mnf = np.sum(freqs * psd) / np.sum(psd)
            cumulative_psd = np.cumsum(psd)
            med_idx = np.where(cumulative_psd >= cumulative_psd[-1] / 2)[0][0]
            mdf = freqs[med_idx]
            mean_freq.append(mnf)
            med_freq.append(mdf)
        freq_features.append(np.concatenate([mean_freq, med_freq]))
    return np.array(freq_features)

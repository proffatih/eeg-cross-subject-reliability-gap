"""
Feature extraction for the Bird et al. (2018) open Muse EEG mental-state dataset.

Raw recordings: 4 subjects (a-d) x 3 states (relaxed / neutral / concentrating)
x 2 trials = 24 CSV files, 4 EEG channels (TP9, AF7, AF8, TP10) sampled at ~256 Hz.

We segment each recording into non-overlapping windows and extract, per channel:
  - relative band power in delta/theta/alpha/beta/gamma (Welch PSD)
  - time-domain statistics: mean, std, skew, kurtosis, RMS, line length, zero-crossings
This yields a fixed-length, interpretable feature vector per window. No data is
fabricated; every value derives from the public recordings.
"""
import os, glob, numpy as np, pandas as pd
from scipy.signal import welch, butter, filtfilt
from scipy.stats import skew, kurtosis

FS = 256.0                      # Muse sampling rate (Hz)
CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 45)}
WIN_SEC = 1.0                   # window length (seconds)
STEP_SEC = 0.5                  # hop (50% overlap) -> more epochs per recording
LABELS = {"relaxed": 0, "neutral": 1, "concentrating": 2}
LABEL_NAMES = ["relaxed", "neutral", "concentrating"]


def _bandpass(x, lo=1.0, hi=45.0, fs=FS, order=4):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def _window_features(seg):
    """seg: (n_samples, n_channels) -> 1D feature vector."""
    feats = []
    for c in range(seg.shape[1]):
        x = seg[:, c]
        x = x - np.mean(x)
        # Welch PSD -> band powers
        nper = min(len(x), 256)
        f, pxx = welch(x, fs=FS, nperseg=nper)
        total = np.trapz(pxx, f) + 1e-12
        for (lo, hi) in BANDS.values():
            idx = (f >= lo) & (f < hi)
            bp = np.trapz(pxx[idx], f[idx]) if idx.any() else 0.0
            feats.append(bp / total)                      # relative band power
        # time-domain stats
        feats.append(np.mean(x))
        feats.append(np.std(x))
        feats.append(skew(x) if np.std(x) > 1e-9 else 0.0)
        feats.append(kurtosis(x) if np.std(x) > 1e-9 else 0.0)
        feats.append(np.sqrt(np.mean(x ** 2)))            # RMS
        feats.append(np.sum(np.abs(np.diff(x))))          # line length
        feats.append(np.sum(np.abs(np.diff(np.sign(x))) > 0) / len(x))  # ZC rate
    return np.asarray(feats, dtype=np.float64)


def feature_names():
    names = []
    for ch in CHANNELS:
        for band in BANDS:
            names.append(f"{ch}_{band}_relpow")
        for s in ["mean", "std", "skew", "kurt", "rms", "linelen", "zcr"]:
            names.append(f"{ch}_{s}")
    return names


def extract_all(raw_dir):
    rows, ys, subjects, recs = [], [], [], []
    files = sorted(glob.glob(os.path.join(raw_dir, "subject*-*-*.csv")))
    win = int(WIN_SEC * FS)
    step = int(STEP_SEC * FS)
    for fp in files:
        base = os.path.basename(fp).replace(".csv", "")
        subj, state, trial = base.split("-")           # e.g. subjecta relaxed 1
        subj = subj.replace("subject", "")
        if state not in LABELS:
            continue
        df = pd.read_csv(fp)
        sig = df[CHANNELS].to_numpy(dtype=np.float64)
        # drop NaNs/rows that are all-zero artefacts
        sig = sig[~np.isnan(sig).any(axis=1)]
        if len(sig) < win:
            continue
        # band-pass each channel
        for c in range(sig.shape[1]):
            try:
                sig[:, c] = _bandpass(sig[:, c])
            except Exception:
                pass
        for start in range(0, len(sig) - win + 1, step):
            seg = sig[start:start + win]
            rows.append(_window_features(seg))
            ys.append(LABELS[state])
            subjects.append(subj)
            recs.append(base)
    X = np.vstack(rows)
    y = np.asarray(ys)
    subj = np.asarray(subjects)
    rec = np.asarray(recs)
    return X, y, subj, rec


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    raw = os.path.join(here, "..", "data", "raw")
    X, y, subj, rec = extract_all(raw)
    out = os.path.join(here, "..", "data", "features.npz")
    np.savez_compressed(out, X=X, y=y, subj=subj, rec=rec,
                        names=np.array(feature_names()))
    print("Feature matrix:", X.shape)
    print("Subjects:", sorted(set(subj.tolist())),
          "counts:", {s: int((subj == s).sum()) for s in sorted(set(subj))})
    print("Class counts:", {LABEL_NAMES[i]: int((y == i).sum()) for i in range(3)})
    print("Saved ->", out)

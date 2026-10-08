"""
Feature extraction for the Bird et al. (2018) open Muse EEG mental-state dataset.

Raw recordings: 4 subjects (a-d) x 3 states (relaxed / neutral / concentrating)
x 2 trials = 24 CSV files, 4 EEG channels (TP9, AF7, AF8, TP10) sampled at ~256 Hz.

Each channel is notch-filtered at 50 Hz (mains pickup) and band-passed 1-45 Hz,
then segmented into 1 s windows with 50% overlap. Per channel we extract:
  - relative band power in delta/theta/alpha/beta/gamma (Welch PSD)
  - time-domain statistics: std, skew, kurtosis, line length, zero-crossing rate
(the window mean is always zero after centring and RMS equals std, so both
were dropped).
This yields a fixed-length, interpretable feature vector per window. No data is
fabricated; every value derives from the public recordings.
"""
import os, glob, warnings, numpy as np, pandas as pd
from scipy.signal import welch, butter, filtfilt, iirnotch
from scipy.stats import skew, kurtosis

FS = 256.0                      # Muse sampling rate (Hz)
CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 45)}
WIN_SEC = 1.0                   # window length (seconds)
STEP_SEC = 0.5                  # hop (50% overlap) -> more epochs per recording
LABELS = {"relaxed": 0, "neutral": 1, "concentrating": 2}
LABEL_NAMES = ["relaxed", "neutral", "concentrating"]


MAINS_HZ = 50.0                 # mains frequency of the recording site
# Narrow lines to remove. Phase 2 also found a line series at 256/12 Hz
# (21.33, 42.67 Hz, ...) mostly in the concentrating recordings; add
# 256 / 12 and 2 * 256 / 12 here to remove it as well.
NOTCH_HZ = (MAINS_HZ,)
NOTCH_Q = 30.0                  # notch quality factor (~1.7 Hz wide at 50 Hz)


def _bandpass(x, lo=1.0, hi=45.0, fs=FS, order=4, notch=None, q=NOTCH_Q):
    """Zero-phase notch(es) at NOTCH_HZ (each skipped when above Nyquist)
    followed by a zero-phase order-4 Butterworth band-pass."""
    for f0 in (NOTCH_HZ if notch is None else notch):
        if f0 < fs / 2:
            bn, an = iirnotch(f0, q, fs=fs)
            x = filtfilt(bn, an, x)
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


GAP_S = 0.1                     # a timestamp jump larger than this splits a recording


def _segments(df, channels=CHANNELS):
    """Split a recording at NaN rows and timestamp gaps, so that filtering and
    windowing never run across a discontinuity (one recording, subjectb-relaxed-2,
    consists of ten 3-4.5 s pieces separated by gaps of up to 700 s)."""
    sig = df[channels].to_numpy(dtype=np.float64)
    t = df["timestamps"].to_numpy(dtype=np.float64)
    idx = np.flatnonzero(~np.isnan(sig).any(axis=1))
    if len(idx) == 0:
        return []
    brk = np.flatnonzero((np.diff(idx) > 1) | (np.diff(t[idx]) > GAP_S)) + 1
    return [sig[i] for i in np.split(idx, brk)]


def _window_features(seg):
    """seg: (n_samples, n_channels) -> 1D feature vector."""
    feats = []
    for c in range(seg.shape[1]):
        x = seg[:, c]
        x = x - np.mean(x)
        # Welch PSD -> band powers
        nper = min(len(x), 256)
        f, pxx = welch(x, fs=FS, nperseg=nper)
        total = np.trapezoid(pxx, f) + 1e-12
        for (lo, hi) in BANDS.values():
            idx = (f >= lo) & (f < hi)
            bp = np.trapezoid(pxx[idx], f[idx]) if idx.any() else 0.0
            feats.append(bp / total)                      # relative band power
        # time-domain stats
        feats.append(np.std(x))
        feats.append(skew(x) if np.std(x) > 1e-9 else 0.0)
        feats.append(kurtosis(x) if np.std(x) > 1e-9 else 0.0)
        feats.append(np.sum(np.abs(np.diff(x))))          # line length
        feats.append(np.sum(np.abs(np.diff(np.sign(x))) > 0) / len(x))  # ZC rate
    return np.asarray(feats, dtype=np.float64)


def feature_names():
    names = []
    for ch in CHANNELS:
        for band in BANDS:
            names.append(f"{ch}_{band}_relpow")
        for s in ["std", "skew", "kurt", "linelen", "zcr"]:
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
        for sig in _segments(df):              # continuous pieces only
            if len(sig) < win:
                continue
            for c in range(sig.shape[1]):      # notch + band-pass each channel
                try:
                    sig[:, c] = _bandpass(sig[:, c])
                except Exception as e:
                    warnings.warn(f"{base} ch{c}: band-pass failed ({e}); left unfiltered")
            for start in range(0, len(sig) - win + 1, step):
                rows.append(_window_features(sig[start:start + win]))
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

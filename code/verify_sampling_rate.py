"""Verify the Muse sampling rate from the recorded timestamps (not the nominal
256 Hz): per recording, the effective rate is (N - 1) / (t_last - t_first);
we report the median and interquartile range across the 24 recordings, and
list every recording that contains timestamp gaps longer than 0.1 s (these
are split into continuous pieces by features._segments)."""
import os, glob, numpy as np, pandas as pd
RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
GAP_S = 0.1
rates = []
for fp in sorted(glob.glob(os.path.join(RAW, "subject*-*-*.csv"))):
    t = pd.read_csv(fp)["timestamps"].to_numpy(float)
    rates.append((len(t) - 1) / (t[-1] - t[0]))
    dt = np.diff(t)
    gaps = dt[dt > GAP_S]
    if len(gaps):
        edges = np.r_[0, np.flatnonzero(dt > GAP_S) + 1, len(t)]
        pieces = [t[edges[i + 1] - 1] - t[edges[i]] for i in range(len(edges) - 1)]
        print(f"{os.path.basename(fp)}: {len(gaps)} gaps > {GAP_S} s (largest {gaps.max():.1f} s), "
              f"{len(t) / 256:.1f} s of data over {t[-1] - t[0]:.0f} s; "
              f"{len(pieces)} pieces of {min(pieces):.1f}-{max(pieces):.1f} s")
rates = np.asarray(rates)
q1, med, q3 = np.percentile(rates, [25, 50, 75])
print(f"recordings={len(rates)}  median={med:.2f} Hz  IQR={q3 - q1:.3f} Hz  "
      f"min={rates.min():.2f}  max={rates.max():.2f}")

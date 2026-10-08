"""
Checks for non-neural, state-dependent content in the recordings (LOSO; GBM
unless stated otherwise).

  notch      : the corrected chain with and without the 50 Hz notch, per subject
               (GBM, and the MLP averaged over five seeds)
  controls   : can features that carry no brain information predict the state?
               50 Hz power only (positive control); DC offset only and 55-128 Hz
               broadband power with every narrow line excluded (negative
               controls). Chance level is 1/3.
  lines      : prominence of the 50 Hz line and of the 256/12 Hz line series
               (21.33 and 42.67 Hz) over the local floor, by subject and state
  extranotch : corrected chain plus notches at 21.33 and 42.67 Hz
  causal     : corrected chain with causal (single-pass) filters
  band20     : corrected chain limited to 1-20 Hz

  python confound_checks.py [notch|controls|lines|extranotch|causal|band20|all]
"""
import os, sys, glob, warnings
import numpy as np, pandas as pd
from scipy.signal import welch, iirnotch, filtfilt, lfilter, butter, sosfilt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as F
from experiment import evaluate, RESULTS

RAW = os.path.join(HERE, "..", "data", "raw")
LINE_HZ = 256.0 / 12                   # 21.33 Hz line series seen in the concentrating recordings


def recordings():
    for fp in sorted(glob.glob(os.path.join(RAW, "subject*-*-*.csv"))):
        s, st, _ = os.path.basename(fp)[:-4].split("-")
        yield s.replace("subject", ""), st, F._segments(pd.read_csv(fp), F.CHANNELS)


def windows(seg, fs=256):
    w, h = int(fs), int(fs / 2)
    for i in range(0, len(seg) - w + 1, h):
        yield seg[i:i + w]


def run_chain(filt):
    """Features with a custom per-channel filter; returns X, y, subj."""
    X, ys, ss = [], [], []
    for s, st, segs in recordings():
        for seg in segs:
            if len(seg) < 256:
                continue
            seg = np.column_stack([filt(seg[:, c]) for c in range(seg.shape[1])])
            for w in windows(seg):
                X.append(F._window_features(w)); ys.append(F.LABELS[st]); ss.append(s)
    return np.asarray(X), np.asarray(ys), np.asarray(ss)


_EXTRA = []


def _save_extra(name):
    """Persist the robustness checks that otherwise only printed, so the numbers
    quoted in the manuscript have a committed artefact like every other result."""
    if _EXTRA:
        pd.DataFrame(_EXTRA).to_csv(
            os.path.join(RESULTS, f"confound_{name}.csv"), index=False)


def _report(tag, X, y, subj, model="GBM"):
    r = evaluate(X, y, subj, model)
    tag = tag if model == "GBM" else f"{tag} [{model}]"
    _EXTRA.append(dict(check=tag, model=model, acc=r["acc"], f1=r.get("f1"),
                       ece=r["ece"],
                       **{f"subj_{k}": v for k, v in r.get("per_subject", {}).items()}))
    print(f"  {tag:<50} acc={r['acc']:.3f}  ece={r['ece']:.3f}  "
          + " ".join(f"{g}={a:.3f}" for g, a in r["per_group"].items()), flush=True)
    return dict(check=tag, model=model, acc=r["acc"], acc_sd=r["acc_sd"], f1=r["f1"], ece=r["ece"],
                **{f"subj_{g}": a for g, a in r["per_group"].items()})


def notch():
    """GBM (deterministic) and MLP (mean of five seeds), with and without the notch."""
    withn = run_chain(lambda x: F._bandpass(x))
    without = run_chain(lambda x: F._bandpass(x, notch=()))
    rows = [_report(tag, *data, model=m) for m in ["GBM", "MLP"]
            for tag, data in [("corrected chain (50 Hz notch)", withn), ("same chain without the notch", without)]]
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "confound_notch.csv"), index=False)


def controls():
    feats = {k: [] for k in ["mains", "dc", "hf"]}
    ys, ss = [], []
    for s, st, segs in recordings():
        for seg in segs:
            for w in windows(seg):                     # raw, unfiltered windows
                f, p = welch(w - w.mean(0), fs=256, nperseg=256, axis=0)
                keep = (f >= 55) & (f <= 128)
                for f0 in list(np.arange(1, 7) * LINE_HZ) + [100.0]:
                    keep &= np.abs(f - f0) > 1.5
                feats["mains"].append(np.log10(p[(f >= 49) & (f <= 51)].sum(0) + 1e-12))
                feats["dc"].append(w.mean(0))
                feats["hf"].append(np.log10(p[keep].sum(0) + 1e-12))
                ys.append(F.LABELS[st]); ss.append(s)
    y, subj = np.asarray(ys), np.asarray(ss)
    rows = [_report(tag, np.asarray(feats[k]), y, subj) for k, tag in
            [("mains", "50 Hz power only (positive control)"),
             ("dc", "DC offset only (negative control)"),
             ("hf", "55-128 Hz broadband, lines excluded (negative)")]]
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "confound_controls.csv"), index=False)


def lines():
    fr = np.arange(1, 100.01, 0.5)
    acc = {}
    for s, st, segs in recordings():
        for seg in segs:
            if len(seg) < 512:
                continue
            for c in range(seg.shape[1]):
                f, p = welch(seg[:, c] - seg[:, c].mean(), fs=256, nperseg=512)
                acc.setdefault((s, st), []).append(np.interp(fr, f, p))

    def prom(p, f0):                                   # line power over local floor, dB
        on = np.abs(fr - f0) <= 0.5
        off = (np.abs(fr - f0) >= 2) & (np.abs(fr - f0) <= 4)
        return 10 * np.log10(p[on].mean() / p[off].mean())
    rows = []
    for f0 in [LINE_HZ, 2 * LINE_HZ, 50.0]:
        for s in "abcd":
            for st in ["relaxed", "neutral", "concentrating"]:
                rows.append(dict(line_hz=round(f0, 2), subject=s, state=st,
                                 prominence_db=prom(np.mean(acc[(s, st)], 0), f0)))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(RESULTS, "confound_lines.csv"), index=False)
    for f0, g in d.groupby("line_hz"):
        print(f"\n  {f0} Hz line, dB over local floor:\n"
              + g.pivot(index="subject", columns="state", values="prominence_db").round(1).to_string())


def extranotch():
    nts = [iirnotch(f0, 30.0, fs=256.0) for f0 in (LINE_HZ, 2 * LINE_HZ)]

    def filt(x):
        for b, a in nts:
            x = filtfilt(b, a, x)
        return F._bandpass(x)
    _report("plus notches at 21.33 and 42.67 Hz", *run_chain(filt))


def causal():
    bn, an = iirnotch(50.0, 30.0, fs=256.0)
    sos = butter(4, [1 / 128, 45 / 128], btype="band", output="sos")
    _report("causal chain (single-pass notch + band-pass)", *run_chain(lambda x: sosfilt(sos, lfilter(bn, an, x))))


def band20():
    _report("chain limited to 1-20 Hz", *run_chain(lambda x: F._bandpass(x, hi=20.0)))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    checks = dict(notch=notch, controls=controls, lines=lines, extranotch=extranotch,
                  causal=causal, band20=band20)
    for name, fn in checks.items():
        if what in (name, "all"):
            print(name, flush=True)
            _EXTRA.clear()
            fn()
            if name in ("extranotch", "causal", "band20"):
                _save_extra(name)

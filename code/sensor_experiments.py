"""
Sensor-level characterisation on the corrected chain (50 Hz notch, 1-45 Hz
band-pass, recordings split at timestamp gaps).

  s1   electrode subsets : all 15 subsets of {TP9, AF7, AF8, TP10}; GBM and MLP
  s2   noise and faults  : white noise, 50 Hz mains, baseline wander, contact
                           degradation and contact dropout. Noise has ONE absolute
                           level for every recording (set against the median
                           in-band power of the data set) and the decoder is
                           trained on clean recordings and tested on degraded
                           ones. GBM.
  s3   converter rate    : 256, 200, 150, 128 and 112 Hz, all of which carry the
                           full 1-45 Hz band with 50 Hz below Nyquist, in the
                           device-like order resample -> notch -> band-pass. GBM.
  s3w  decision window   : 2, 1 and 0.5 s at 256 and 112 Hz. GBM.
  s4   compute budget    : feature-extraction and inference time, model size. GBM.
  s3c  (optional, not part of "all") band-controlled rate test: every rate gets
       identical content, filtered once at 256 Hz before resampling.

  python sensor_experiments.py [s1|s2|s3|s3w|s4|s3c|all]      (default: all)
"""
import os, sys, json, time, pickle, itertools, warnings, glob
from fractions import Fraction
import numpy as np, pandas as pd
from scipy.signal import butter, filtfilt, firwin, kaiserord, iirnotch, resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as F
from experiment import evaluate, RESULTS

RAW = os.path.join(HERE, "..", "data", "raw")
FS0 = F.FS                            # recorded rate, 256 Hz
CH = F.CHANNELS                       # ["TP9", "AF7", "AF8", "TP10"]
N_FEAT_PER_CH = 10                    # 5 band powers + 5 waveform statistics
N_REP = 3                             # noise realisations per stochastic degradation
NOISE_FLOOR_UV = 0.1                  # open-contact noise floor, independent of the signal
RATES = [256.0, 200.0, 150.0, 128.0, 112.0]


def load_base():
    d = np.load(os.path.join(HERE, "..", "data", "features.npz"), allow_pickle=True)
    return d["X"], d["y"], d["subj"]


def channel_cols(subset):
    """Feature columns of an electrode subset (10 consecutive columns per contact)."""
    cols = []
    for ch in subset:
        i = CH.index(ch)
        cols.extend(range(i * N_FEAT_PER_CH, (i + 1) * N_FEAT_PER_CH))
    return np.asarray(cols)


# --------------------------------------------------------------------------
# signal chain with optional fault, rate and window
# --------------------------------------------------------------------------
def _resample(x, fs):
    r = Fraction(int(fs), int(FS0))                   # e.g. 200/256 = 25/32
    return x if r == 1 else resample_poly(x, r.numerator, r.denominator)


def _steep_lowpass(pass_hz, stop_hz, fs=FS0, atten_db=80.0):
    ntaps, beta = kaiserord(atten_db, (stop_hz - pass_hz) / (fs / 2))
    return firwin(ntaps | 1, (pass_hz + stop_hz) / 2, window=("kaiser", beta), fs=fs)


def extract(perturb=None, fs=FS0, win_sec=1.0, seed=0, common_band=None):
    """Re-run the feature chain on every recording.
    perturb(sig, rng): fault applied to the raw 256 Hz signal of a recording,
        once per recording, before any filtering, as a front-end fault would act.
    fs, win_sec: converter rate and decision-window length.
    common_band=(pass_hz, stop_hz): band-controlled chain (S3c only)."""
    rng = np.random.default_rng(seed)
    if common_band is not None:
        lp = _steep_lowpass(*common_band)
        bh, ah = butter(4, 1.0 / (FS0 / 2), btype="high")
    rows, ys, subjects = [], [], []
    for fp in sorted(glob.glob(os.path.join(RAW, "subject*-*-*.csv"))):
        subj, state, _ = os.path.basename(fp)[:-4].split("-")
        if state not in F.LABELS:
            continue
        segs = F._segments(pd.read_csv(fp), CH)
        if not segs:
            continue
        if perturb is not None:                       # one fault draw per recording
            lens = [len(s) for s in segs]
            segs = np.split(perturb(np.vstack(segs), rng), np.cumsum(lens)[:-1])
        for seg in segs:
            if len(seg) < int(FS0):
                continue
            cols = []
            for c in range(seg.shape[1]):
                x = seg[:, c]
                if common_band is None:               # device-like order
                    x = F._bandpass(_resample(x, fs), hi=min(45.0, 0.45 * fs), fs=fs)
                else:                                 # identical content at every rate
                    for f0 in F.NOTCH_HZ:
                        bn, an = iirnotch(f0, F.NOTCH_Q, fs=FS0)
                        x = filtfilt(bn, an, x)
                    x = filtfilt(bh, ah, x)
                    x = filtfilt(lp, [1.0], x, padlen=min(3 * len(lp), len(x) - 1))
                    x = _resample(x, fs)
                cols.append(x)
            seg = np.column_stack(cols)
            win, step = int(win_sec * fs), int(0.5 * win_sec * fs)
            old = F.FS
            F.FS = fs                                 # Welch at the new rate
            try:
                for st in range(0, len(seg) - win + 1, step):
                    rows.append(F._window_features(seg[st:st + win]))
                    ys.append(F.LABELS[state])
                    subjects.append(subj.replace("subject", ""))
            finally:
                F.FS = old
    return np.vstack(rows), np.asarray(ys), np.asarray(subjects)


# --------------------------------------------------------------------------
# S1  electrode subsets
# --------------------------------------------------------------------------
def s1_electrode_subsets(sizes=(1, 2, 3, 4), out="sensor_electrode_subsets.csv"):
    X, y, subj = load_base()
    rows, per = [], []
    for k in sizes:
        for subset in itertools.combinations(CH, k):
            Xs = X[:, channel_cols(subset)]
            row = dict(n_ch=k, electrodes="+".join(subset))
            for model in ["GBM", "MLP"]:
                r = evaluate(Xs, y, subj, model)
                row.update({f"{model}_{m}": r[m] for m in ["acc", "acc_sd", "f1", "ece"]})
                per += [dict(electrodes=row["electrodes"], model=model, subject=g, acc=a)
                        for g, a in r["per_group"].items()]
            rows.append(row)
            print(f"  S1 {row['electrodes']:<20} GBM={row['GBM_acc']:.3f}  "
                  f"MLP={row['MLP_acc']:.3f}+-{row['MLP_acc_sd']:.3f}", flush=True)
    df = pd.DataFrame(rows).sort_values(["n_ch", "GBM_acc"], ascending=[False, False])
    if out:
        df.to_csv(os.path.join(RESULTS, out), index=False)
        pd.DataFrame(per).to_csv(os.path.join(RESULTS, out.replace(".csv", "_per_subject.csv")), index=False)
    return df


def best_pair():
    """Best two-electrode subset for the GBM, from the S1 results."""
    f = os.path.join(RESULTS, "sensor_electrode_subsets.csv")
    d = pd.read_csv(f) if os.path.exists(f) else s1_electrode_subsets()
    return d[d.n_ch == 2].sort_values("GBM_acc", ascending=False).iloc[0]["electrodes"].split("+")


# --------------------------------------------------------------------------
# S2  noise and contact faults (trained clean, tested degraded)
# --------------------------------------------------------------------------
def noise_reference():
    """Median in-band (notched, 1-45 Hz) variance over every recording and
    contact: one absolute reference, so the injected noise carries no
    information about the recording it is added to."""
    v = []
    for fp in sorted(glob.glob(os.path.join(RAW, "subject*-*-*.csv"))):
        for seg in F._segments(pd.read_csv(fp), CH):
            if len(seg) >= int(FS0):
                v += [np.var(F._bandpass(seg[:, c])) for c in range(seg.shape[1])]
    return float(np.median(v))


def white_noise(sd):
    return lambda sig, rng: sig + rng.normal(0, sd, sig.shape)


def mains(sd, f0=50.0):
    def f(sig, rng):
        t = np.arange(len(sig)) / FS0
        ph = rng.uniform(0, 2 * np.pi, sig.shape[1])
        return sig + sd * np.sqrt(2) * np.sin(2 * np.pi * f0 * t[:, None] + ph[None, :])
    return f


def wander(sd):
    """Low-frequency baseline wander (0.1-0.5 Hz): electrode movement."""
    def f(sig, rng):
        t = np.arange(len(sig)) / FS0
        f0 = rng.uniform(0.1, 0.5, sig.shape[1])
        ph = rng.uniform(0, 2 * np.pi, sig.shape[1])
        return sig + sd * np.sqrt(2) * np.sin(2 * np.pi * f0[None, :] * t[:, None] + ph[None, :])
    return f


def contact(level):
    """Degraded contact on ONE random electrode: attenuation to `level` of the
    amplitude plus the low-pass of a rising skin-electrode impedance."""
    def f(sig, rng):
        out = sig.copy()
        c = rng.integers(0, out.shape[1])
        b, a = butter(2, min((8.0 + 30.0 * level) / (FS0 / 2), 0.99), btype="low")
        out[:, c] = level * filtfilt(b, a, out[:, c])
        return out
    return f


def dropout():
    """Open contact on ONE random electrode: replaced by a fixed noise floor."""
    def f(sig, rng):
        out = sig.copy()
        c = rng.integers(0, out.shape[1])
        out[:, c] = rng.normal(0, NOISE_FLOOR_UV, len(out))
        return out
    return f


def s2_noise_faults(names=None, out="sensor_noise_faults.csv"):
    p = noise_reference()
    sd = lambda snr: np.sqrt(p / 10 ** (snr / 10.0))
    grid = [("white noise", snr, f"{snr} dB SNR", white_noise(sd(snr))) for snr in [20, 10, 5, 0]]
    grid += [("50 Hz mains", snr, f"{snr} dB SNR", mains(sd(snr))) for snr in [20, 10, 5, 0]]
    grid += [("baseline wander", snr, f"{snr} dB SNR", wander(sd(snr))) for snr in [10, 0]]
    grid += [("contact degradation", np.nan, f"{int(lv * 100)}% amplitude", contact(lv)) for lv in [0.5, 0.2]]
    grid += [("contact dropout", np.nan, "random contact", dropout())]
    Xc, yc, sc = load_base()
    rows = []
    if names is None or "none" in names:
        r = evaluate(Xc, yc, sc, "GBM")
        rows.append(dict(degradation="none", level="clean", snr_db=np.nan, n_rep=1,
                         acc=r["acc"], acc_sd=0.0, f1=r["f1"], ece=r["ece"], ece_sd=0.0))
    for name, snr, level, fn in grid:
        if names is not None and name not in names:
            continue
        reps = [evaluate(Xc, yc, sc, "GBM", test=extract(perturb=fn, seed=rep)) for rep in range(N_REP)]
        rows.append(dict(degradation=name, level=level, snr_db=snr, n_rep=N_REP,
                         acc=np.mean([r["acc"] for r in reps]), acc_sd=np.std([r["acc"] for r in reps]),
                         f1=np.mean([r["f1"] for r in reps]), ece=np.mean([r["ece"] for r in reps]),
                         ece_sd=np.std([r["ece"] for r in reps])))
        print(f"  S2 {name:<20} {level:<15} acc={rows[-1]['acc']:.3f}+-{rows[-1]['acc_sd']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    if out:
        df.to_csv(os.path.join(RESULTS, out), index=False)
        json.dump({"p_ref_uV2": p}, open(os.path.join(RESULTS, "noise_reference.json"), "w"))
    return df


# --------------------------------------------------------------------------
# S3  converter rate and decision window
# --------------------------------------------------------------------------
def _rate_rows(cond, out):
    rows = []
    for kw in cond:
        X, y, subj = extract(**kw)
        r = evaluate(X, y, subj, "GBM")
        rows.append(dict(fs_hz=kw.get("fs", FS0), win_s=kw.get("win_sec", 1.0), n_windows=len(X),
                         acc=r["acc"], f1=r["f1"], ece=r["ece"]))
        print(f"  fs={rows[-1]['fs_hz']:>5.0f} Hz  win={rows[-1]['win_s']:.1f} s  "
              f"acc={r['acc']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    if out:
        df.to_csv(os.path.join(RESULTS, out), index=False)
    return df


def s3_rate(rates=RATES, out="sensor_rate.csv"):
    return _rate_rows([dict(fs=fs) for fs in rates], out)


def s3_window(rates=(256.0, 112.0), wins=(2.0, 1.0, 0.5), out="sensor_window.csv"):
    return _rate_rows([dict(fs=fs, win_sec=w) for fs in rates for w in wins], out)


def s3c_rate_band_controlled(rates=RATES, out="sensor_rate_band_controlled.csv"):
    return _rate_rows([dict(fs=fs, common_band=(45.0, 47.0)) for fs in rates], out)


# --------------------------------------------------------------------------
# S4  compute budget
# --------------------------------------------------------------------------
def loso_custom(X, y, subj, make_clf):
    """LOSO accuracy for an arbitrary scikit-learn classifier factory."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    preds, trues = [], []
    for s in sorted(set(subj.tolist())):
        te, tr = subj == s, subj != s
        sc_ = StandardScaler().fit(X[tr])
        clf = make_clf().fit(sc_.transform(X[tr]), y[tr])
        preds.append(clf.predict(sc_.transform(X[te])))
        trues.append(y[te])
    return accuracy_score(np.concatenate(trues), np.concatenate(preds))


def _feat_latency(n_ch, reps=300):
    """Per-window feature-extraction latency (ms) for an n_ch array."""
    seg = np.random.default_rng(0).normal(0, 20, (int(F.WIN_SEC * F.FS), n_ch))
    for _ in range(10):
        F._window_features(seg)
    t0 = time.perf_counter()
    for _ in range(reps):
        F._window_features(seg)
    return (time.perf_counter() - t0) / reps * 1e3


def s4_compute_budget(pair):
    """Feature-extraction time scales with the electrode count; inference is
    timed in batch and divided by the number of windows. Single-threaded on
    the host CPU: a relative budget between array sizes, not an MCU figure."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        X, y, subj = load_base()
        out = {"window_s": F.WIN_SEC, "pair": "+".join(pair),
               "feature_extraction_ms": {f"{n}-electrode": _feat_latency(n) for n in (4, len(pair))},
               "decoders": {}}
        configs = [("full", dict(max_iter=300, learning_rate=0.08)),
                   ("compact", dict(max_iter=60, learning_rate=0.2, max_depth=3, max_leaf_nodes=8))]
        for cfg_name, kw in configs:
            for tag, Xs in [("4-electrode", X), (f"{len(pair)}-electrode", X[:, channel_cols(pair)])]:
                s = sorted(set(subj.tolist()))[0]
                te, tr = subj == s, subj != s
                sc_ = StandardScaler().fit(Xs[tr])
                clf = HistGradientBoostingClassifier(random_state=0, **kw).fit(sc_.transform(Xs[tr]), y[tr])
                Z = sc_.transform(Xs[te])
                clf.predict_proba(Z[:64])
                t0 = time.perf_counter()
                for _ in range(20):
                    clf.predict_proba(Z)
                out["decoders"][f"{cfg_name} / {tag}"] = dict(
                    model_kb=len(pickle.dumps(clf)) / 1024.0,
                    inference_ms_per_window=(time.perf_counter() - t0) / (20 * len(Z)) * 1e3,
                    n_features=int(Xs.shape[1]),
                    loso_acc=loso_custom(Xs, y, subj, lambda: HistGradientBoostingClassifier(random_state=0, **kw)))
                print(f"  S4 {cfg_name:<8} {tag:<12} {out['decoders'][f'{cfg_name} / {tag}']}", flush=True)
    with open(os.path.join(RESULTS, "sensor_compute_budget.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("s1", "all"):
        print("S1 electrode subsets", flush=True); s1_electrode_subsets()
    if what in ("s2", "all"):
        print("S2 noise and contact faults", flush=True); s2_noise_faults()
    if what in ("s3", "all"):
        print("S3 converter rate", flush=True); s3_rate()
    if what in ("s3w", "all"):
        print("S3w decision window", flush=True); s3_window()
    if what in ("s4", "all"):
        print("S4 compute budget", flush=True); s4_compute_budget(best_pair())
    if what == "s3c":
        print("S3c band-controlled rate test", flush=True); s3c_rate_band_controlled()

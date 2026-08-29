"""
Sensor-level characterisation of a four-channel dry-electrode affective-EEG
headband (Bird et al. 2018 open Muse recordings, 4 subjects x 3 mental states).

Everything here is measured on the real recordings; nothing is simulated except
the injected sensor degradations, which are applied to the real signal.

S1  ELECTRODE COUNT   how far can the electrode array be reduced before
                      cross-subject (LOSO) accuracy and calibration collapse?
                      All 15 non-empty subsets of {TP9, AF7, AF8, TP10}.

S2  NOISE / INTERFERENCE RESILIENCE
                      degradations injected into the raw signal *before* the
                      analogue-equivalent band-pass, so they propagate through
                      the whole chain exactly as a real sensor fault would:
                        - additive white noise at a controlled SNR
                        - 50 Hz mains interference
                        - baseline wander (motion / electrode movement)
                        - contact-impedance degradation (attenuation + LPF)
                        - single-electrode dropout
S3  SAMPLING RATE / WINDOW LENGTH
                      the sensor node's power-latency budget: decimated fs and
                      shorter decision windows.
S4  EDGE COST         feature-extraction and inference latency per window, and
                      model footprint, measured on this machine.

Outputs -> ../results/sensor_*.csv|json
"""
import os, sys, json, time, pickle, itertools
import numpy as np, pandas as pd
from scipy.signal import butter, filtfilt, decimate

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as F
from experiment import ece, fit_predict, run_loso  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
RAW = os.path.join(HERE, "..", "data", "raw")
os.makedirs(RESULTS, exist_ok=True)
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

SEED = 0
N_REP = 3          # independent noise realisations per stochastic degradation
N_CLS = 3
CH = F.CHANNELS                       # ["TP9", "AF7", "AF8", "TP10"]
N_FEAT_PER_CH = 12                    # 5 band powers + 7 time-domain stats


# --------------------------------------------------------------------------
# generic evaluation
# --------------------------------------------------------------------------
def loso_scores(X, y, subj, model="GBM", seed=SEED):
    P, Y = run_loso(X, y, subj, model, seed, N_CLS)
    pred = P.argmax(1)
    return dict(acc=accuracy_score(Y, pred),
                f1=f1_score(Y, pred, average="macro"),
                ece=ece(P, Y))


def loso_custom(X, y, subj, make_clf):
    """LOSO accuracy for an arbitrary scikit-learn classifier factory."""
    from sklearn.preprocessing import StandardScaler
    preds, trues = [], []
    for s in sorted(set(subj.tolist())):
        te, tr = subj == s, subj != s
        sc_ = StandardScaler().fit(X[tr])
        clf = make_clf().fit(sc_.transform(X[tr]), y[tr])
        preds.append(clf.predict(sc_.transform(X[te])))
        trues.append(y[te])
    return accuracy_score(np.concatenate(trues), np.concatenate(preds))


def load_base():
    d = np.load(os.path.join(HERE, "..", "data", "features.npz"), allow_pickle=True)
    return d["X"], d["y"], d["subj"]


# --------------------------------------------------------------------------
# S1  electrode-count reduction  (re-uses the base feature matrix: the feature
#     vector is a plain concatenation of per-channel blocks)
# --------------------------------------------------------------------------
def channel_cols(subset):
    cols = []
    for ch in subset:
        i = CH.index(ch)
        cols.extend(range(i * N_FEAT_PER_CH, (i + 1) * N_FEAT_PER_CH))
    return np.asarray(cols)


def s1_electrode_count():
    X, y, subj = load_base()
    rows = []
    for k in range(1, len(CH) + 1):
        for subset in itertools.combinations(CH, k):
            sc = loso_scores(X[:, channel_cols(subset)], y, subj)
            rows.append(dict(n_ch=k, electrodes="+".join(subset), **sc))
            print(f"  S1 {k}ch {'+'.join(subset):<20} acc={sc['acc']:.3f} "
                  f"ECE={sc['ece']:.3f}", flush=True)
    df = pd.DataFrame(rows).sort_values(["n_ch", "acc"], ascending=[True, False])
    df.to_csv(os.path.join(RESULTS, "sensor_electrode_ablation.csv"), index=False)
    return df


# --------------------------------------------------------------------------
# S2  sensor degradations, injected into the raw signal
# --------------------------------------------------------------------------
def _snr_scale(x, snr_db):
    """Noise std giving the requested SNR against this channel's own power."""
    p_sig = np.mean(x ** 2)
    return np.sqrt(p_sig / (10 ** (snr_db / 10.0)))


def perturb_white(sig, rng, snr_db):
    out = sig.copy()
    for c in range(out.shape[1]):
        out[:, c] += rng.normal(0, _snr_scale(out[:, c], snr_db), len(out))
    return out


def perturb_mains(sig, rng, snr_db, f0=50.0):
    """50 Hz mains pickup, random phase per channel."""
    t = np.arange(len(sig)) / F.FS
    out = sig.copy()
    for c in range(out.shape[1]):
        a = _snr_scale(out[:, c], snr_db) * np.sqrt(2.0)     # sine RMS -> amplitude
        out[:, c] += a * np.sin(2 * np.pi * f0 * t + rng.uniform(0, 2 * np.pi))
    return out


def perturb_wander(sig, rng, snr_db):
    """Low-frequency baseline wander: electrode movement / motion artefact."""
    t = np.arange(len(sig)) / F.FS
    out = sig.copy()
    for c in range(out.shape[1]):
        a = _snr_scale(out[:, c], snr_db) * np.sqrt(2.0)
        f0 = rng.uniform(0.1, 0.5)
        out[:, c] += a * np.sin(2 * np.pi * f0 * t + rng.uniform(0, 2 * np.pi))
    return out


def perturb_contact(sig, rng, level):
    """Contact-impedance degradation on ONE electrode: the skin-electrode
    interface becomes a stronger low-pass and attenuates the signal.
    level = fraction of the original amplitude retained."""
    out = sig.copy()
    c = rng.integers(0, out.shape[1])
    fc = 8.0 + 30.0 * level          # worse contact -> lower corner frequency
    b, a = butter(2, min(fc / (F.FS / 2), 0.99), btype="low")
    out[:, c] = level * filtfilt(b, a, out[:, c])
    return out


def perturb_dropout(sig, rng, which):
    """A single electrode loses contact entirely (flat + sensor noise floor)."""
    out = sig.copy()
    c = CH.index(which)
    out[:, c] = rng.normal(0, 1e-3 * np.std(sig[:, c]) + 1e-6, len(out))
    return out


def extract_with(perturb=None, win_sec=F.WIN_SEC, fs=F.FS, seed=SEED):
    """Re-run the whole feature pipeline with an optional raw-signal
    perturbation and/or a different sampling rate / window length."""
    import glob
    rng = np.random.default_rng(seed)
    rows, ys, subjects = [], [], []
    old_fs, old_win = F.FS, F.WIN_SEC
    F.FS, F.WIN_SEC = fs, win_sec
    try:
        win = int(win_sec * fs)
        step = int(0.5 * win_sec * fs)
        for fp in sorted(glob.glob(os.path.join(RAW, "subject*-*-*.csv"))):
            base = os.path.basename(fp).replace(".csv", "")
            subj, state, _ = base.split("-")
            if state not in F.LABELS:
                continue
            df = pd.read_csv(fp)
            sig = df[CH].to_numpy(dtype=np.float64)
            sig = sig[~np.isnan(sig).any(axis=1)]
            if fs != old_fs:                       # decimate to the target rate
                q = int(round(old_fs / fs))
                if q > 1:
                    sig = np.column_stack([decimate(sig[:, c], q, ftype="fir",
                                                    zero_phase=True)
                                           for c in range(sig.shape[1])])
            if perturb is not None:
                sig = perturb(sig, rng)
            if len(sig) < win:
                continue
            hi = min(45.0, 0.45 * fs)              # stay below Nyquist when decimated
            for c in range(sig.shape[1]):          # analogue-equivalent band-pass
                try:
                    sig[:, c] = F._bandpass(sig[:, c], hi=hi, fs=fs)
                except Exception:
                    pass
            for start in range(0, len(sig) - win + 1, step):
                rows.append(F._window_features(sig[start:start + win]))
                ys.append(F.LABELS[state])
                subjects.append(subj.replace("subject", ""))
    finally:
        F.FS, F.WIN_SEC = old_fs, old_win
    return np.vstack(rows), np.asarray(ys), np.asarray(subjects)


def s2_noise_resilience():
    rows = []
    clean = loso_scores(*load_base())
    rows.append(dict(degradation="none", snr_db=np.nan, level="clean",
                     n_rep=1, acc=clean["acc"], acc_sd=0.0, f1=clean["f1"],
                     ece=clean["ece"], ece_sd=0.0))
    print(f"  S2 clean            acc={clean['acc']:.3f} ECE={clean['ece']:.3f}",
          flush=True)

    # Every stochastic degradation is repeated over independent noise
    # realisations, otherwise a single draw can invert the ordering of two
    # nearby interference levels.
    grid = []
    for snr in [20, 10, 5, 0]:
        grid.append(("white noise", snr, f"{snr} dB SNR",
                     lambda s, r, q=snr: perturb_white(s, r, q)))
    for snr in [20, 10, 5, 0]:
        grid.append(("50 Hz mains", snr, f"{snr} dB SNR",
                     lambda s, r, q=snr: perturb_mains(s, r, q)))
    for snr in [10, 0]:
        grid.append(("baseline wander", snr, f"{snr} dB SNR",
                     lambda s, r, q=snr: perturb_wander(s, r, q)))
    for lv in [0.5, 0.2]:
        grid.append(("contact degradation", np.nan, f"{int(lv*100)}% amplitude",
                     lambda s, r, q=lv: perturb_contact(s, r, q)))
    for ch in CH:
        grid.append(("electrode dropout", np.nan, ch,
                     lambda s, r, q=ch: perturb_dropout(s, r, q)))

    for name, snr, level, fn in grid:
        n_rep = 1 if name == "electrode dropout" else N_REP
        accs, f1s, eces = [], [], []
        for rep in range(n_rep):
            X, y, subj = extract_with(perturb=fn, seed=SEED + rep)
            sc = loso_scores(X, y, subj)
            accs.append(sc["acc"]); f1s.append(sc["f1"]); eces.append(sc["ece"])
        rows.append(dict(degradation=name, snr_db=snr, level=level, n_rep=n_rep,
                         acc=np.mean(accs), acc_sd=np.std(accs),
                         f1=np.mean(f1s), ece=np.mean(eces),
                         ece_sd=np.std(eces)))
        print(f"  S2 {name:<20} {level:<14} acc={np.mean(accs):.3f}"
              f"+-{np.std(accs):.3f} ECE={np.mean(eces):.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "sensor_noise_resilience.csv"), index=False)
    return df


# --------------------------------------------------------------------------
# S3  sampling rate and decision-window length
# --------------------------------------------------------------------------
def s3_rate_window():
    rows = []
    for fs in [256.0, 128.0, 64.0]:
        for win in [2.0, 1.0, 0.5]:
            if win * fs < 32:
                continue
            X, y, subj = extract_with(win_sec=win, fs=fs)
            sc = loso_scores(X, y, subj)
            rows.append(dict(fs_hz=fs, win_s=win, n_windows=len(X), **sc))
            print(f"  S3 fs={fs:>5.0f} Hz win={win:.2f} s  acc={sc['acc']:.3f} "
                  f"ECE={sc['ece']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "sensor_rate_window.csv"), index=False)
    return df


def s3b_feature_group_vs_rate():
    """Why does decimation cost accuracy? Split the feature vector into the
    5 relative band powers and the 7 waveform-shape statistics per channel and
    score each group separately at each ADC rate."""
    bp, td = [], []
    for c in range(len(CH)):
        base = c * N_FEAT_PER_CH
        bp += list(range(base, base + 5))
        td += list(range(base + 5, base + N_FEAT_PER_CH))
    rows = []
    for fs in [256.0, 128.0, 64.0]:
        X, y, subj = extract_with(fs=fs)
        rows.append(dict(fs_hz=fs,
                         acc_all=loso_scores(X, y, subj)["acc"],
                         acc_bandpower=loso_scores(X[:, bp], y, subj)["acc"],
                         acc_waveform=loso_scores(X[:, td], y, subj)["acc"]))
        print(f"  S3b fs={fs:>5.0f} Hz  all={rows[-1]['acc_all']:.3f} "
              f"bandpower={rows[-1]['acc_bandpower']:.3f} "
              f"waveform={rows[-1]['acc_waveform']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "sensor_feature_group_vs_rate.csv"),
              index=False)
    return df


# --------------------------------------------------------------------------
# S4  edge cost
# --------------------------------------------------------------------------
def _feat_latency(n_ch, reps=300):
    """Per-window feature-extraction latency for an n_ch array."""
    seg = np.random.default_rng(0).normal(0, 20, (int(F.WIN_SEC * F.FS), n_ch))
    for _ in range(10):
        F._window_features(seg)
    t0 = time.perf_counter()
    for _ in range(reps):
        F._window_features(seg)
    return (time.perf_counter() - t0) / reps * 1e3


def s4_edge_cost(best_subset):
    """Compute budget of the sensor node.

    Two things matter on a headband-class node: the per-window cost of turning
    raw samples into features (which scales with the electrode count) and the
    footprint of the decoder. Inference is timed in batch and divided by the
    number of windows, because a single-sample predict_proba call in scikit-learn
    is dominated by Python-side overhead and would not represent the arithmetic.
    A deliberately compact decoder is also trained and scored so the footprint
    figure is tied to a real accuracy, not quoted in isolation.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    X, y, subj = load_base()
    cols = channel_cols(best_subset)
    out = {"window_s": F.WIN_SEC,
           "feature_extraction_ms": {f"{n}-electrode": _feat_latency(n)
                                     for n in (4, len(best_subset))}}

    configs = [("full", dict(max_iter=300, learning_rate=0.08)),
               ("compact", dict(max_iter=60, learning_rate=0.2, max_depth=3,
                                max_leaf_nodes=8))]
    arrays = [("4-electrode", X), (f"{len(best_subset)}-electrode", X[:, cols])]

    out["decoders"] = {}
    for cfg_name, kw in configs:
        for tag, Xs in arrays:
            # footprint and latency from one LOSO fold; accuracy from full LOSO
            s = sorted(set(subj.tolist()))[0]
            te, tr = subj == s, subj != s
            sc_ = StandardScaler().fit(Xs[tr])
            clf = HistGradientBoostingClassifier(random_state=SEED, **kw)
            clf.fit(sc_.transform(Xs[tr]), y[tr])
            Z = sc_.transform(Xs[te])
            clf.predict_proba(Z[:64])
            t0 = time.perf_counter()
            for _ in range(20):
                clf.predict_proba(Z)
            t_inf = (time.perf_counter() - t0) / (20 * len(Z)) * 1e3   # ms/window

            acc = loso_custom(Xs, y, subj,
                              lambda: HistGradientBoostingClassifier(
                                  random_state=SEED, **kw))
            out["decoders"][f"{cfg_name} / {tag}"] = dict(
                model_kb=len(pickle.dumps(clf)) / 1024.0,
                inference_ms_per_window=t_inf,
                n_features=int(Xs.shape[1]),
                loso_acc=acc)
            print(f"  S4 {cfg_name:<8} {tag:<12} {out['decoders'][f'{cfg_name} / {tag}']}",
                  flush=True)

    out["note"] = ("single-threaded on the host CPU (x86-64); quoted as a "
                   "relative compute budget between array sizes, not an MCU figure")
    with open(os.path.join(RESULTS, "sensor_edge_cost.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    print("S1 electrode-count reduction", flush=True)
    d1 = s1_electrode_count()
    best2 = d1[d1.n_ch == 2].iloc[0]["electrodes"].split("+")
    print("  best 2-electrode subset:", best2, flush=True)

    print("S2 noise / interference resilience", flush=True)
    s2_noise_resilience()

    print("S3 sampling rate and window length", flush=True)
    s3_rate_window()

    print("S3b feature group vs ADC rate", flush=True)
    s3b_feature_group_vs_rate()

    print("S4 edge cost", flush=True)
    s4_edge_cost(best2)
    print("done", flush=True)

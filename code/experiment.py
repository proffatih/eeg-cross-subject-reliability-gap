"""
Subject-aware evaluation of EEG mental-state (affective) classification on the
open Bird et al. (2018) Muse dataset (4 subjects, 3 states).

Two protocols:
  WITHIN  : stratified group K-fold pooling all subjects, but with epochs grouped
            by recording so that windows from the same recording never straddle
            the train/test split (prevents within-recording leakage). This is the
            optimistic "subjects seen in training" setting.
  LOSO    : leave-one-subject-out; the test subject is entirely unseen. This is the
            honest cross-subject generalization setting.

Models:
  GBM : HistGradientBoostingClassifier (strong gradient-boosted tabular baseline)
  MLP : multilayer perceptron (scikit-learn MLPClassifier, two hidden layers)
        on standardized features

Metrics (per protocol, per model, averaged over SEEDS):
  accuracy, macro-F1, and Expected Calibration Error (ECE, 15 bins).
We also dump a pooled confusion matrix and reliability-curve data for the figures.
"""
import os, json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
N_BINS = 15
LABEL_NAMES = ["relaxed", "neutral", "concentrating"]


# ----------------------------- metrics -------------------------------------
def ece(probs, labels, n_bins=N_BINS):
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            e += (m.mean()) * abs(acc[m].mean() - conf[m].mean())
    return e


def reliability(probs, labels, n_bins=N_BINS):
    conf = probs.max(1); pred = probs.argmax(1)
    acc = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys, ws = [], [], []
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            xs.append(conf[m].mean()); ys.append(acc[m].mean()); ws.append(m.mean())
    return np.array(xs), np.array(ys), np.array(ws)


# ----------------------------- models --------------------------------------
def _align(probs, classes_, n_cls, n):
    """Map a classifier's predict_proba columns onto the full class set."""
    full = np.zeros((n, n_cls))
    for j, c in enumerate(classes_):
        full[:, c] = probs[:, j]
    return full


def fit_predict(model_name, Xtr, ytr, Xte, seed, n_cls):
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    if model_name == "GBM":
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                             random_state=seed)
    else:  # MLP
        clf = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                            alpha=1e-3, batch_size=128, learning_rate_init=1e-3,
                            max_iter=400, early_stopping=True, n_iter_no_change=20,
                            random_state=seed)
    clf.fit(Xtr_s, ytr)
    return _align(clf.predict_proba(Xte_s), clf.classes_, n_cls, len(Xte_s))


# ----------------------------- protocols -----------------------------------
def run_within(X, y, rec, model_name, seed, n_cls):
    """Stratified group K-fold; groups = recordings (no within-recording leak)."""
    gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    all_p, all_y = [], []
    for tr, te in gkf.split(X, y, groups=rec):
        p = fit_predict(model_name, X[tr], y[tr], X[te], seed, n_cls)
        all_p.append(p); all_y.append(y[te])
    P = np.vstack(all_p); Y = np.concatenate(all_y)
    return P, Y


def run_loso(X, y, subj, model_name, seed, n_cls):
    all_p, all_y = [], []
    for s in sorted(set(subj.tolist())):
        te = subj == s; tr = ~te
        p = fit_predict(model_name, X[tr], y[tr], X[te], seed, n_cls)
        all_p.append(p); all_y.append(y[te])
    P = np.vstack(all_p); Y = np.concatenate(all_y)
    return P, Y


# ----------------------------- main ----------------------------------------
def main():
    d = np.load(os.path.join(HERE, "..", "data", "features.npz"), allow_pickle=True)
    X, y, subj, rec = d["X"], d["y"], d["subj"], d["rec"]
    n_cls = len(LABEL_NAMES)
    print(f"Loaded X={X.shape}, subjects={sorted(set(subj.tolist()))}")

    rows = []
    # store pooled preds (seed 0) for confusion + reliability figures
    store = {}
    for model_name in ["GBM", "MLP"]:
        for proto, fn in [("within", run_within), ("loso", run_loso)]:
            accs, f1s, eces = [], [], []
            for seed in SEEDS:
                if proto == "within":
                    P, Y = run_within(X, y, rec, model_name, seed, n_cls)
                else:
                    P, Y = run_loso(X, y, subj, model_name, seed, n_cls)
                pred = P.argmax(1)
                accs.append(accuracy_score(Y, pred))
                f1s.append(f1_score(Y, pred, average="macro"))
                eces.append(ece(P, Y))
                if seed == 0:
                    store[(model_name, proto)] = (P, Y)
            rows.append(dict(model=model_name, protocol=proto,
                             acc_mean=np.mean(accs), acc_std=np.std(accs),
                             f1_mean=np.mean(f1s), f1_std=np.std(f1s),
                             ece_mean=np.mean(eces), ece_std=np.std(eces),
                             n_seeds=len(SEEDS)))
            print(f"{model_name:4s} {proto:7s} acc={np.mean(accs):.3f}"
                  f"+-{np.std(accs):.3f}  f1={np.mean(f1s):.3f}  ece={np.mean(eces):.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "metrics.csv"), index=False)

    # generalization gap table
    gap = []
    for model_name in ["GBM", "MLP"]:
        w = df[(df.model == model_name) & (df.protocol == "within")].iloc[0]
        l = df[(df.model == model_name) & (df.protocol == "loso")].iloc[0]
        gap.append(dict(model=model_name,
                        within_acc=w.acc_mean, loso_acc=l.acc_mean,
                        acc_gap=w.acc_mean - l.acc_mean,
                        within_f1=w.f1_mean, loso_f1=l.f1_mean,
                        f1_gap=w.f1_mean - l.f1_mean))
    pd.DataFrame(gap).to_csv(os.path.join(RESULTS, "generalization_gap.csv"), index=False)

    # per-subject LOSO accuracy (best model = GBM, seed 0..4 averaged)
    persubj = []
    for s in sorted(set(subj.tolist())):
        for model_name in ["GBM", "MLP"]:
            a = []
            for seed in SEEDS:
                te = subj == s; tr = ~te
                p = fit_predict(model_name, X[tr], y[tr], X[te], seed, n_cls)
                a.append(accuracy_score(y[te], p.argmax(1)))
            persubj.append(dict(subject=s, model=model_name,
                                loso_acc_mean=np.mean(a), loso_acc_std=np.std(a)))
    pd.DataFrame(persubj).to_csv(os.path.join(RESULTS, "per_subject_loso.csv"), index=False)

    # dump arrays for figures (confusion + reliability), seed 0
    np.savez_compressed(os.path.join(RESULTS, "fig_arrays.npz"),
                        **{f"{m}_{p}_P": store[(m, p)][0] for m in ["GBM", "MLP"] for p in ["within", "loso"]},
                        **{f"{m}_{p}_Y": store[(m, p)][1] for m in ["GBM", "MLP"] for p in ["within", "loso"]})

    # confusion matrices (GBM, both protocols)
    cms = {}
    for proto in ["within", "loso"]:
        P, Y = store[("GBM", proto)]
        cm = confusion_matrix(Y, P.argmax(1), labels=[0, 1, 2])
        cms[proto] = cm.tolist()
    with open(os.path.join(RESULTS, "confusion_gbm.json"), "w") as f:
        json.dump(cms, f, indent=2)

    print("\nSaved results to", RESULTS)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

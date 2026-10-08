"""
Evaluation core shared by every script: the two decoders, the two evaluation
protocols and the metrics.

Decoders
  GBM : histogram-based gradient-boosted trees (300 iterations, learning rate
        0.08). At this data size it has no random element (no early stopping,
        no feature or bin subsampling), so it is run once.
  MLP : multilayer perceptron, two hidden layers of 128 and 64 ReLU units and a
        softmax output, Adam, early stopping on a 10% validation split. Its
        results are averaged over the fixed seeds in SEEDS, which set the initial
        weights, the validation split and the mini-batch order.

Protocols
  LOSO    : leave-one-subject-out; the test subject is entirely unseen.
  SESSION : within-subject, leave-one-session-out. Each of the 8 folds holds out
            one session of one subject (its relaxed, neutral and concentrating
            recordings) and trains on everything else, including that subject's
            other session. The folds are fixed, so the result is identical on
            every machine.

Metrics: accuracy, macro-F1 and Expected Calibration Error (ECE, 15 bins).

  python experiment.py   ->  results/metrics.csv, results/per_group.csv,
                             results/confusion_gbm.json
"""
import os, json
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]          # fixed in advance; the best seed is never selected
N_BINS = 15
N_CLS = 3
LABEL_NAMES = ["relaxed", "neutral", "concentrating"]


# ----------------------------- metrics -------------------------------------
def ece(probs, labels, n_bins=N_BINS):
    """Expected Calibration Error with equal-width confidence bins."""
    conf = probs.max(1)
    acc = (probs.argmax(1) == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e


def scores(P, Y):
    pred = P.argmax(1)
    return dict(acc=accuracy_score(Y, pred), f1=f1_score(Y, pred, average="macro"),
                ece=float(ece(P, Y)))


# ----------------------------- decoders ------------------------------------
def _align(probs, classes_, n_cls, n):
    """Map a classifier's predict_proba columns onto the full class set."""
    full = np.zeros((n, n_cls))
    for j, c in enumerate(classes_):
        full[:, c] = probs[:, j]
    return full


def fit_predict(model_name, Xtr, ytr, Xte, seed, n_cls=N_CLS):
    """Standardise with training statistics, fit, return test probabilities."""
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
def run_loso(X, y, subj, model_name, seed, n_cls=N_CLS):
    """Leave-one-subject-out; predictions pooled in sorted-subject order."""
    all_p, all_y = [], []
    for s in sorted(set(subj.tolist())):
        te = subj == s
        all_p.append(fit_predict(model_name, X[~te], y[~te], X[te], seed, n_cls))
        all_y.append(y[te])
    return np.vstack(all_p), np.concatenate(all_y)


def loso_groups(subj):
    """Subject label of every pooled LOSO prediction (same order as run_loso)."""
    return np.concatenate([np.full(int((subj == s).sum()), s) for s in sorted(set(subj.tolist()))])


def run_loso_train_test(Xtr, ytr, str_, Xte, yte, ste, model_name, seed, n_cls=N_CLS):
    """LOSO with separate training and test sets (e.g. trained on clean
    recordings, tested on degraded ones): fold s trains on Xtr without subject s
    and tests on Xte of subject s."""
    all_p, all_y = [], []
    for s in sorted(set(str_.tolist())):
        all_p.append(fit_predict(model_name, Xtr[str_ != s], ytr[str_ != s],
                                 Xte[ste == s], seed, n_cls))
        all_y.append(yte[ste == s])
    return np.vstack(all_p), np.concatenate(all_y)


def run_session(X, y, subj, rec, model_name, seed, n_cls=N_CLS):
    """Within-subject leave-one-session-out (8 fixed folds)."""
    sess = np.array([r.split("-")[-1] for r in rec])
    all_p, all_y, groups = [], [], []
    for s in sorted(set(subj.tolist())):
        for k in sorted(set(sess.tolist())):
            te = (subj == s) & (sess == k)
            all_p.append(fit_predict(model_name, X[~te], y[~te], X[te], seed, n_cls))
            all_y.append(y[te]); groups.append(np.full(int(te.sum()), f"{s}{k}"))
    return np.vstack(all_p), np.concatenate(all_y), np.concatenate(groups)


def evaluate(X, y, subj, model_name, protocol="loso", rec=None, test=None):
    """Accuracy, macro-F1 and ECE (mean and s.d. over seeds) and per-group
    accuracy. GBM is deterministic here and is run once (seed 0); the MLP is
    run for every seed in SEEDS. test=(Xte, yte, ste) trains on (X, y, subj)
    and tests on the given set, subject by subject."""
    seeds = [0] if model_name == "GBM" else SEEDS
    runs, per = [], []
    for seed in seeds:
        if test is not None:
            P, Y = run_loso_train_test(X, y, subj, *test, model_name, seed)
            G = loso_groups(test[2])
        elif protocol == "loso":
            P, Y = run_loso(X, y, subj, model_name, seed)
            G = loso_groups(subj)
        else:
            P, Y, G = run_session(X, y, subj, rec, model_name, seed)
        runs.append(scores(P, Y))
        per.append({g: accuracy_score(Y[G == g], P[G == g].argmax(1)) for g in sorted(set(G.tolist()))})
    out = {k: float(np.mean([r[k] for r in runs])) for k in ["acc", "f1", "ece"]}
    out.update({k + "_sd": float(np.std([r[k] for r in runs])) for k in ["acc", "f1", "ece"]})
    out["n_seeds"] = len(seeds)
    out["per_group"] = pd.DataFrame(per).mean().to_dict()
    return out


# ----------------------------- main ----------------------------------------
def main():
    d = np.load(os.path.join(HERE, "..", "data", "features.npz"), allow_pickle=True)
    X, y, subj, rec = d["X"], d["y"], d["subj"], d["rec"]
    print(f"Loaded X={X.shape}, subjects={sorted(set(subj.tolist()))}")
    rows, per = [], []
    for model in ["GBM", "MLP"]:
        for protocol in ["session", "loso"]:
            r = evaluate(X, y, subj, model, protocol, rec)
            rows.append(dict(model=model, protocol=protocol, n_seeds=r["n_seeds"],
                             **{k: r[k] for k in ["acc", "acc_sd", "f1", "f1_sd", "ece", "ece_sd"]}))
            per += [dict(model=model, protocol=protocol, group=g, acc=a) for g, a in r["per_group"].items()]
            print(f"{model} {protocol:<7} acc={r['acc']:.3f}+-{r['acc_sd']:.3f}  "
                  f"f1={r['f1']:.3f}  ece={r['ece']:.3f}")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "metrics.csv"), index=False)
    pd.DataFrame(per).to_csv(os.path.join(RESULTS, "per_group.csv"), index=False)
    cms = {}
    for protocol in ["session", "loso"]:
        out = run_loso(X, y, subj, "GBM", 0) if protocol == "loso" else run_session(X, y, subj, rec, "GBM", 0)
        cms[protocol] = confusion_matrix(out[1], out[0].argmax(1), labels=[0, 1, 2]).tolist()
    with open(os.path.join(RESULTS, "confusion_gbm.json"), "w") as f:
        json.dump(cms, f, indent=2)
    print("Saved results to", os.path.abspath(RESULTS))


if __name__ == "__main__":
    main()

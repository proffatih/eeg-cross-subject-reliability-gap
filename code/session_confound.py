"""Is the recording session, rather than the mental state, what is decoded?

The reviewers asked whether the apparent state effect is confounded with
recording order. The raw timestamps answer that directly: in three of the four
participants the concentrating recordings were made on a different day from the
relaxed and neutral recordings, in a separate session with a separate donning of
the headband. Any session-specific signature is therefore a shortcut to the
label for those participants.

This script runs the analyses that separate the two explanations.

  sessions   recording day and clock time of all 24 recordings, from the
             timestamp column, and which participants have the day split
  split      three-class LOSO accuracy per participant against whether that
             participant has the split
  within     the decisive test: relaxed versus neutral only. Both classes were
             recorded on the same day for every participant, so this comparison
             is free of the day confound. If the decoder carries neural
             information, it should still separate them.
  timeonly   a baseline that sees nothing but the recording's position in time
             (no EEG at all), as a measure of how much the label can be read off
             the session structure alone

  python session_confound.py [sessions|split|within|timeonly|all]
"""
import os, sys, glob, json, warnings
import numpy as np, pandas as pd
import datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as F
from experiment import evaluate, RESULTS

RAW = os.path.join(HERE, "..", "data", "raw")


def _recording_table():
    rows = []
    for fp in sorted(glob.glob(os.path.join(RAW, "subject*-*-*.csv"))):
        s, st, tr = os.path.basename(fp)[:-4].split("-")
        t = pd.read_csv(fp, usecols=["timestamps"]).timestamps
        t0, t1 = float(t.iloc[0]), float(t.iloc[-1])
        rows.append(dict(subject=s.replace("subject", ""), state=st, trial=tr,
                         t0=t0, duration_s=round(t1 - t0, 1),
                         day=dt.datetime.fromtimestamp(t0, dt.UTC).strftime("%Y-%m-%d"),
                         clock=dt.datetime.fromtimestamp(t0, dt.UTC).strftime("%H:%M")))
    return pd.DataFrame(rows).sort_values(["subject", "t0"]).reset_index(drop=True)


def split_subjects(tab):
    """Participants whose concentrating recordings sit on their own day."""
    out = {}
    for s, g in tab.groupby("subject"):
        days = g.groupby("state").day.apply(lambda x: set(x))
        conc = days.get("concentrating", set())
        other = set().union(*[days[k] for k in days.index if k != "concentrating"])
        out[s] = bool(conc) and not (conc & other)
    return out


def sessions():
    tab = _recording_table()
    tab.to_csv(os.path.join(RESULTS, "session_recording_table.csv"), index=False)
    sp = split_subjects(tab)
    pd.DataFrame([dict(subject=s, separate_session_for_concentrating=v)
                  for s, v in sp.items()]).to_csv(
        os.path.join(RESULTS, "session_split.csv"), index=False)
    for s, g in tab.groupby("subject"):
        base = g.t0.min()
        print(f"  {s}: split={sp[s]}")
        for _, r in g.iterrows():
            print(f"      +{r.t0-base:8.0f}s  {r.day} {r.clock}  {r.state:<13} {r.duration_s:6.1f}s")
    return tab, sp


def _load():
    d = np.load(os.path.join(HERE, "..", "data", "features.npz"), allow_pickle=True)
    return d["X"], d["y"], d["subj"], d["rec"]


def split_vs_accuracy():
    """Per-participant three-class LOSO accuracy against the day split."""
    tab = _recording_table(); sp = split_subjects(tab)
    X, y, subj, _ = _load()
    rows = []
    for model in ("GBM", "MLP"):
        r = evaluate(X, y, subj, model)
        for s, acc in r["per_group"].items():
            rows.append(dict(model=model, subject=s, separate_session=sp[s],
                             loso_acc=acc))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "session_split_vs_accuracy.csv"), index=False)
    print(df.to_string(index=False))
    for model, g in df.groupby("model"):
        a = g[g.separate_session].loso_acc.mean()
        b = g[~g.separate_session].loso_acc.mean()
        print(f"  {model}: split {a:.3f}  vs  no split {b:.3f}  (chance 0.333)")
    return df


def within_session():
    """relaxed vs neutral only: same day for every participant, so the day
    confound cannot contribute. Chance is 0.5."""
    X, y, subj, _ = _load()
    keep = np.isin(y, [F.LABELS["relaxed"], F.LABELS["neutral"]])
    Xk, yk, sk = X[keep], y[keep], subj[keep]
    yk = (yk == F.LABELS["neutral"]).astype(int)
    rows = []
    for model in ("GBM", "MLP"):
        r = evaluate(Xk, yk, sk, model)
        rows.append(dict(model=model, n_windows=int(len(yk)),
                         loso_acc=r["acc"], loso_acc_sd=r["acc_sd"],
                         loso_f1=r["f1"], ece=r["ece"],
                         **{f"subj_{k}": v for k, v in r["per_group"].items()}))
        print(f"  {model}: relaxed-vs-neutral LOSO acc={r['acc']:.3f} "
              f"(chance 0.500)  per subject {r['per_group']}")
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS, "within_session_relaxed_vs_neutral.csv"), index=False)


def time_only():
    """No EEG at all: the only feature is where the window sits in time, as the
    recording's start offset within that participant's first session."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score
    tab = _recording_table()
    key = {(r.subject, r.state, r.trial): r.t0 for r in tab.itertuples()}
    base = tab.groupby("subject").t0.min().to_dict()
    X, y, subj, rec = _load()
    # rec is "<state>-<trial>" per window
    def _key(s, r):
        parts = str(r).split("-")          # rec is the full basename
        return (s, parts[-2], parts[-1])
    t = np.array([key[_key(s, r)] - base[s] for s, r in zip(subj, rec)])
    T = t.reshape(-1, 1)
    P, Y = [], []
    for s in sorted(set(subj.tolist())):
        te, tr = subj == s, subj != s
        clf = HistGradientBoostingClassifier(random_state=0).fit(T[tr], y[tr])
        P.append(clf.predict(T[te])); Y.append(y[te])
    acc = accuracy_score(np.concatenate(Y), np.concatenate(P))
    print(f"  time-only baseline (no EEG): LOSO acc={acc:.3f}  (chance 0.333)")
    json.dump(dict(time_only_loso_acc=float(acc), chance=1/3),
              open(os.path.join(RESULTS, "time_only_baseline.json"), "w"), indent=2)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in dict(sessions=sessions, split=split_vs_accuracy,
                         within=within_session, timeonly=time_only).items():
        if what in (name, "all"):
            print(name, flush=True); fn()

"""Second round of follow-ups: the checks a co-author's audit asked for, plus
the two reviewer requests the first revision answered only in words.

Four things are settled here.

  timedefs   : the 50 Hz-versus-time correlation under three definitions of
               "time". The first revision reported one of them. The sign is not
               stable across the three, so the manuscript must say which one it
               means and lead with the defensible one.
  permtest   : is the relaxed-versus-neutral contrast above chance *in each
               participant*? Windows overlap and are serially correlated, so the
               unit that may be permuted is the recording, not the window.
  baselines  : the relaxed-versus-neutral contrast scored by a clock-only and an
               artifact-only decoder, the same controls the three-class analysis
               already carries. Without them the contrast is not shown to be
               neural either.
  covariate  : mains level regressed on state with acquisition time and
               participant as covariates (reviewer 2's "include time as a
               covariate").

  python followups_round2.py [timedefs|permtest|baselines|covariate|all]
"""
import os, sys, json, itertools
import numpy as np, pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as F
from experiment import RESULTS
from session_confound import _recording_table, _load
from reviewer_followups import _artifact_and_time_features


# --------------------------------------------------------------------------
def timedefs():
    """The same correlation under three definitions of time."""
    d = pd.read_csv(os.path.join(RESULTS, "mains_vs_time.csv"))
    d1 = d[~d.second_day].copy()
    y = d1.mains_log10_uV2.values
    out = {"n_recordings_day1": int(len(d1))}

    r, p = spearmanr(d1.t0.values, y)
    out["absolute_clock"] = dict(rho=float(r), p=float(p))

    r, p = spearmanr(d1.hours_from_first.values, y)
    out["elapsed_since_participants_own_first"] = dict(rho=float(r), p=float(p))

    rank, val = [], []
    for s, g in d1.groupby("subject"):
        g = g.sort_values("t0")
        rank += list(range(len(g))); val += list(g.mains_log10_uV2.values)
    r, p = spearmanr(rank, val)
    out["within_participant_rank"] = dict(rho=float(r), p=float(p))

    # the recording whose windows are spliced across a long gap
    out["note_spliced_recording"] = (
        "subject b, relaxed, trial 2: 10572 samples spanning 957.7 s with nine "
        "gaps totalling 916 s, i.e. 41.5 s of data. Its windows do not sit at a "
        "single point in time.")
    for k, v in out.items():
        if isinstance(v, dict) and "rho" in v:
            print(f"  {k:42s} rho={v['rho']:+.3f}  p={v['p']:.4f}")
    json.dump(out, open(os.path.join(RESULTS, "mains_time_definitions.json"), "w"),
              indent=2)
    return out


# --------------------------------------------------------------------------
def _within_arrays():
    """relaxed/neutral windows with their recording identity."""
    X, y, subj, rec = _load()
    keep = np.isin(y, [F.LABELS["relaxed"], F.LABELS["neutral"]])
    yk = (y[keep] == F.LABELS["neutral"]).astype(int)
    r = np.array([f"{s}|{str(q).split('-')[-2]}-{str(q).split('-')[-1]}"
                  for s, q in zip(subj[keep], rec[keep])])
    return X[keep], yk, subj[keep], r


def _loso_predictions(X, y, subj, model="GBM", seed=0):
    """Window-by-window LOSO predictions from the manuscript's own pipeline, so
    these numbers are directly comparable with the rest of the paper."""
    from experiment import run_loso, loso_groups
    P, Y = run_loso(X, y, subj, model, seed)
    G = loso_groups(subj)
    return P.argmax(1), Y, G


def permtest():
    """Per-participant significance of relaxed vs neutral, permuting the unit
    that is actually exchangeable: the recording."""
    X, y, subj, rec = _within_arrays()
    pred, Y, G = _loso_predictions(X, y, subj)
    # run_loso returns rows grouped by held-out subject; realign rec to it
    order = np.concatenate([np.where(subj == g)[0]
                            for g in sorted(set(subj.tolist()))])
    rec, y = rec[order], Y
    subj = G
    rows = []
    for s in sorted(set(subj.tolist())):
        m = subj == s
        recs = sorted(set(rec[m].tolist()))
        lab = {r: int(y[m][rec[m] == r][0]) for r in recs}
        n_pos = sum(lab.values())
        obs = float((pred[m] == y[m]).mean())
        accs = []
        for combo in itertools.combinations(range(len(recs)), n_pos):
            assign = {r: (1 if i in combo else 0) for i, r in enumerate(recs)}
            yy = np.array([assign[r] for r in rec[m]])
            accs.append(float((pred[m] == yy).mean()))
        accs = np.array(accs)
        p = float((accs >= obs).sum() / len(accs))
        rows.append(dict(subject=s, n_recordings=len(recs), n_windows=int(m.sum()),
                         acc=obs, n_permutations=len(accs),
                         p_recording_level=p, min_attainable_p=1.0 / len(accs)))
        print(f"  {s}: acc={obs:.3f}  {len(recs)} recordings  "
              f"p={p:.3f} (smallest attainable {1.0/len(accs):.3f})")
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS, "within_session_permutation.csv"), index=False)
    return rows


# --------------------------------------------------------------------------
def baselines():
    """Clock-only and artifact-only decoders on the relaxed-vs-neutral contrast."""
    from sklearn.metrics import accuracy_score
    A, T, ya, sa = _artifact_and_time_features()
    keep = np.isin(ya, [F.LABELS["relaxed"], F.LABELS["neutral"]])
    yk = (ya[keep] == F.LABELS["neutral"]).astype(int)
    sk = sa[keep]
    Xf, yf, sf, _ = _within_arrays()
    assert len(yk) == len(yf) and (yk == yf).all() and (sk == sf).all(), \
        "artifact and in-band window orders disagree"
    rows = []
    for name, M in [("in-band EEG (full decoder)", Xf),
                    ("artifact only (50 Hz + DC + 55-128 Hz)", A[keep]),
                    ("clock only (no EEG)", T[keep])]:
        pred, Y, G = _loso_predictions(M, yk, sk)
        per = {s: float((pred[G == s] == Y[G == s]).mean())
               for s in sorted(set(G.tolist()))}
        acc = float(accuracy_score(Y, pred))
        rows.append(dict(model=name, acc=acc, **{f"subj_{k}": v for k, v in per.items()}))
        print(f"  {name:42s} acc={acc:.3f}  {per}")
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS, "within_session_baselines.csv"), index=False)
    return rows


# --------------------------------------------------------------------------
def covariate():
    """Mains level with time and participant as covariates (reviewer 2)."""
    import statsmodels.formula.api as smf
    d = pd.read_csv(os.path.join(RESULTS, "mains_vs_time.csv"))
    out = {}
    for tag, sub in [("day1_only", d[~d.second_day]), ("all_recordings", d)]:
        f = "mains_log10_uV2 ~ C(state, Treatment('neutral')) + hours_from_first + C(subject)"
        m = smf.ols(f, data=sub).fit()
        out[tag] = {
            "n": int(len(sub)),
            "params": {k: float(v) for k, v in m.params.items()},
            "pvalues": {k: float(v) for k, v in m.pvalues.items()},
            "r2": float(m.rsquared)}
        print(f"\n  [{tag}] n={len(sub)}  R2={m.rsquared:.3f}")
        for k in m.params.index:
            if "state" in k or "hours" in k:
                print(f"    {k:52s} b={m.params[k]:+.3f}  p={m.pvalues[k]:.4f}")
    json.dump(out, open(os.path.join(RESULTS, "mains_covariate_model.json"), "w"),
              indent=2)
    return out


if __name__ == "__main__":
    want = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in dict(timedefs=timedefs, permtest=permtest,
                         baselines=baselines, covariate=covariate).items():
        if want in ("all", name):
            print(f"\n=== {name} ===")
            fn()

"""Follow-up analyses requested by the reviewers at the first round.

Reviewer 2 asked for three things that the raw files can actually answer and
that the first revision did not yet report; Reviewer 1's point about the
high-frequency control is sharpened by the third of them.

  mainstime  : 50 Hz power of every recording against when it was recorded,
               by state (R2.1, "plot 50 Hz power against time by state" and
               "correlate 50 Hz with time"). Separates a slow within-session
               drift from a step at the day boundary.
  agreement  : the full in-band decoder, an artifact-only decoder (50 Hz power,
               DC offset and 55-128 Hz power, no in-band EEG) and a decoder that
               sees only the clock, scored on the same windows in the same LOSO
               folds, and compared to each other prediction by prediction
               (R2.3, "compare confusion matrices/predictions between in-band
               and artifact-only models").
  persubject : the artifact baselines per participant (R2.3, "report per-subject
               artifact baselines"), from the committed control results.

  python reviewer_followups.py [mainstime|agreement|persubject|all]
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
from scipy.signal import welch
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix, cohen_kappa_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as F
from experiment import evaluate, run_loso, loso_groups, RESULTS, LABEL_NAMES
from confound_checks import windows, LINE_HZ, run_chain
from session_confound import _recording_table, split_subjects


# ---------------------------------------------------------------- R2.1 -----
def mains_vs_time():
    """50 Hz power per recording against acquisition time, by state."""
    tab = _recording_table().set_index(["subject", "state", "trial"])
    rows = []
    import glob
    for fp in sorted(glob.glob(os.path.join(HERE, "..", "data", "raw",
                                            "subject*-*-*.csv"))):
        s, st, tr = os.path.basename(fp)[:-4].split("-")
        s = s.replace("subject", "")
        segs = F._segments(pd.read_csv(fp), F.CHANNELS)
        p50 = []
        for seg in segs:
            for w in windows(seg):
                f, p = welch(w - w.mean(0), fs=256, nperseg=256, axis=0)
                p50.append(p[(f >= 49) & (f <= 51)].sum(0))
        if not p50:
            continue
        r = tab.loc[(s, st, tr)]
        rows.append(dict(subject=s, state=st, trial=tr, day=r.day, clock=r.clock,
                         t0=r.t0, n_windows=len(p50),
                         mains_log10_uV2=float(np.log10(np.mean(p50, 0).mean() + 1e-12))))
    d = pd.DataFrame(rows)
    base = d.groupby("subject").t0.min()
    d["hours_from_first"] = [(r.t0 - base[r.subject]) / 3600.0 for r in d.itertuples()]
    d["second_day"] = d.hours_from_first > 12
    d = d.sort_values(["subject", "t0"]).reset_index(drop=True)
    d.to_csv(os.path.join(RESULTS, "mains_vs_time.csv"), index=False)

    # (i) within the first day only: does the line level drift with time?
    day1 = d[~d.second_day]
    rho1, p1 = spearmanr(day1.hours_from_first, day1.mains_log10_uV2)
    # within participant, first day (ranks computed inside each participant)
    z = day1.copy()
    z["r_t"] = z.groupby("subject").hours_from_first.rank()
    z["r_m"] = z.groupby("subject").mains_log10_uV2.rank()
    rho1w, p1w = spearmanr(z.r_t, z.r_m)
    # (ii) the step between the two days, in the three participants that have one
    sp = split_subjects(_recording_table())
    step = []
    for s, g in d.groupby("subject"):
        if not sp[s]:
            continue
        step.append(g[g.second_day].mains_log10_uV2.mean()
                    - g[~g.second_day].mains_log10_uV2.mean())
    # (iii) same-day state difference in participant c, who has no day split
    c = d[d.subject == "c"]
    c_conc = c[c.state == "concentrating"].mains_log10_uV2.mean()
    c_rest = c[c.state != "concentrating"].mains_log10_uV2.mean()

    out = dict(
        spearman_time_vs_mains_day1=dict(rho=float(rho1), p=float(p1), n=int(len(day1))),
        spearman_time_vs_mains_day1_within_participant=dict(rho=float(rho1w), p=float(p1w)),
        day_step_log10_uV2=dict(per_participant=[float(x) for x in step],
                                mean=float(np.mean(step)),
                                in_dB=float(10 * np.mean(step))),
        participant_c_same_day=dict(concentrating=float(c_conc), other=float(c_rest),
                                    difference_in_dB=float(10 * (c_conc - c_rest))))
    json.dump(out, open(os.path.join(RESULTS, "mains_vs_time_stats.json"), "w"), indent=2)
    print("  day 1 only, time vs 50 Hz power: "
          f"rho={rho1:.3f} (p={p1:.3f}, n={len(day1)}); "
          f"within participant rho={rho1w:.3f} (p={p1w:.3f})")
    print(f"  step at the day boundary: {10*np.mean(step):+.1f} dB "
          f"(per participant {[round(10*x,1) for x in step]})")
    print(f"  participant c, same day, concentrating vs rest: "
          f"{10*(c_conc-c_rest):+.1f} dB")
    return d


# ---------------------------------------------------------------- R2.3 -----
def _artifact_and_time_features():
    """Artifact-only features and the clock feature, window by window, in the
    same order as the in-band chain."""
    tab = _recording_table()
    key = {(r.subject, r.state, r.trial): r.t0 for r in tab.itertuples()}
    base = tab.groupby("subject").t0.min().to_dict()
    import glob
    A, T, ys, ss = [], [], [], []
    for fp in sorted(glob.glob(os.path.join(HERE, "..", "data", "raw",
                                            "subject*-*-*.csv"))):
        s, st, tr = os.path.basename(fp)[:-4].split("-")
        s = s.replace("subject", "")
        for seg in F._segments(pd.read_csv(fp), F.CHANNELS):
            for w in windows(seg):
                f, p = welch(w - w.mean(0), fs=256, nperseg=256, axis=0)
                keep = (f >= 55) & (f <= 128)
                for f0 in list(np.arange(1, 7) * LINE_HZ) + [100.0]:
                    keep &= np.abs(f - f0) > 1.5
                A.append(np.concatenate([
                    np.log10(p[(f >= 49) & (f <= 51)].sum(0) + 1e-12),
                    w.mean(0),
                    np.log10(p[keep].sum(0) + 1e-12)]))
                T.append([key[(s, st, tr)] - base[s]])
                ys.append(F.LABELS[st]); ss.append(s)
    return np.asarray(A), np.asarray(T), np.asarray(ys), np.asarray(ss)


def agreement():
    """Do the in-band decoder and a decoder that sees no in-band EEG make the
    same predictions?"""
    Xf, yf, sf = run_chain(lambda x: F._bandpass(x))          # corrected chain
    A, T, ya, sa = _artifact_and_time_features()
    assert len(Xf) == len(A) and (yf == ya).all() and (sf == sa).all(), \
        "window order differs between the two feature builders"
    y, subj = yf, sf
    P = {}
    for name, X in [("in-band EEG (full decoder)", Xf),
                    ("artifact only (50 Hz + DC + 55-128 Hz)", A),
                    ("clock only (no EEG)", T)]:
        probs, Y = run_loso(X, y, subj, "GBM", 0)
        P[name] = probs.argmax(1)
    G = loso_groups(subj)
    Ytrue = np.concatenate([y[subj == s] for s in sorted(set(subj.tolist()))])

    rows = []
    for name, pred in P.items():
        rows.append(dict(model=name, acc=float((pred == Ytrue).mean()),
                         **{f"subj_{s}": float((pred[G == s] == Ytrue[G == s]).mean())
                            for s in sorted(set(G.tolist()))}))
    acc = pd.DataFrame(rows)
    names = list(P)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = P[names[i]], P[names[j]]
            pairs.append(dict(model_a=names[i], model_b=names[j],
                              agreement=float((a == b).mean()),
                              kappa=float(cohen_kappa_score(a, b))))
    chance = float(sum((Ytrue == k).mean() ** 2 for k in [0, 1, 2]))
    pair = pd.DataFrame(pairs)
    pair["chance_agreement"] = round(chance, 6)
    acc.to_csv(os.path.join(RESULTS, "artifact_model_accuracy.csv"), index=False)
    pair.to_csv(os.path.join(RESULTS, "artifact_model_agreement.csv"), index=False)
    cms = {name: confusion_matrix(Ytrue, pred, labels=[0, 1, 2]).tolist()
           for name, pred in P.items()}
    cms["in-band vs artifact-only (predictions)"] = confusion_matrix(
        P[names[0]], P[names[1]], labels=[0, 1, 2]).tolist()
    cms["label_order"] = LABEL_NAMES
    json.dump(cms, open(os.path.join(RESULTS, "artifact_model_confusion.json"), "w"),
              indent=2)
    print(acc.to_string(index=False))
    print(pair.to_string(index=False))
    print(f"  chance agreement of two independent decoders: {chance:.3f}")


# ---------------------------------------------------------------- R2.3 -----
def persubject():
    """Artifact baselines per participant, from the committed control run."""
    d = pd.read_csv(os.path.join(RESULTS, "confound_controls.csv"))
    sp = split_subjects(_recording_table())
    cols = [c for c in d.columns if c.startswith("subj_")]
    out = d.melt(id_vars=["check"], value_vars=cols,
                 var_name="subject", value_name="loso_acc")
    out["subject"] = out.subject.str.replace("subj_", "", regex=False)
    out["separate_session"] = out.subject.map(sp)
    out = out.pivot(index="check", columns="subject", values="loso_acc").round(3)
    out.to_csv(os.path.join(RESULTS, "artifact_baselines_per_subject.csv"))
    print(out.to_string())
    print("  participants with the day split:",
          ", ".join(k for k, v in sp.items() if v),
          "| without:", ", ".join(k for k, v in sp.items() if not v))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in dict(mainstime=mains_vs_time, agreement=agreement,
                         persubject=persubject).items():
        if what in (name, "all"):
            print(name, flush=True); fn()

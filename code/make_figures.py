"""Figures and tables for the manuscript, sized for one IEEE column (3.4 in).

  Fig. 1  fig1_electrodes.pdf : placement and accuracy versus number of contacts
  Fig. 2  fig2_mains.pdf      : mains pickup (spectrum; accuracy with and without notch)
  Fig. 3  fig3_sensor.pdf     : interference, contact faults, converter rate, window
  Table 1 table1_subsets.tex  : all 15 electrode subsets
  Table 2 table2_compute.tex  : compute budget

Run after features.py, confound_checks.py (notch) and sensor_experiments.py.
"""
import os, sys, glob, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.signal import welch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as F

R = os.path.join(HERE, "..", "results")
FG = os.path.join(HERE, "..", "figures")
RAW = os.path.join(HERE, "..", "data", "raw")
os.makedirs(FG, exist_ok=True)
COL = 3.4
C_GBM, C_MLP, C_BAD, C_GREY = "#2E5A87", "#CC8963", "#C44E52", "#8C8C8C"
CHANCE = 1 / 3
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.7, "lines.linewidth": 1.2,
    "lines.markersize": 3.5, "savefig.dpi": 400, "figure.facecolor": "white"})


def save(fig, name):
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FG, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def label(ax, s, x=-0.2):
    ax.text(x, 1.03, s, transform=ax.transAxes, fontweight="bold", fontsize=7.5, va="bottom")


def chance(ax, horizontal=True):
    (ax.axhline if horizontal else ax.axvline)(CHANCE, color=C_GREY, ls=":", lw=0.8)


# ---------------------------------------------------------------- Fig. 1
SITES = {"TP9": (-0.62, -0.46), "AF7": (-0.45, 0.58), "AF8": (0.45, 0.58), "TP10": (0.62, -0.46)}


def mini_head(ax, keep, gbm, mlp, good=True):
    ax.set_axis_off(); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.85, 1.3); ax.set_aspect("equal")
    ax.add_patch(Circle((0, 0), 1.0, fc="#F2F4F7", ec="#8899AA", lw=0.8))
    ax.plot([-0.14, 0, 0.14], [0.99, 1.19, 0.99], color="#8899AA", lw=0.8)
    col = C_GBM if good else C_BAD
    for name, (x, y) in SITES.items():
        on = name in keep
        ax.add_patch(Circle((x, y), 0.17, fc=col if on else "white", ec=col if on else "#99AABB", lw=0.8))
    ax.text(0, -1.12, "+".join(keep), ha="center", va="top", fontsize=5.5, color=col, fontweight="bold")
    ax.text(0, -1.52, f"{gbm:.3f} / {mlp:.3f}", ha="center", va="top", fontsize=5.5, color=col)


def fig1():
    d = pd.read_csv(os.path.join(R, "sensor_electrode_subsets.csv"))
    acc = d.set_index("electrodes")
    fig = plt.figure(figsize=(COL, 2.15))
    gs = fig.add_gridspec(3, 2, width_ratios=[0.75, 2.25], wspace=0.42, hspace=0.05)
    for i, (pair, good) in enumerate([("TP9+TP10", True), ("AF8+TP10", True), ("AF7+AF8", False)]):
        h = fig.add_subplot(gs[i, 0])
        mini_head(h, pair.split("+"), acc.loc[pair, "GBM_acc"], acc.loc[pair, "MLP_acc"], good)
        if i == 0:
            h.set_title("GBM / MLP", fontsize=5.5, color="#444", pad=1)
    ax = fig.add_subplot(gs[:, 1])
    for model, col, mk, ls, dx in [("GBM", C_GBM, "o", "-", -0.07), ("MLP", C_MLP, "^", "--", 0.07)]:
        ax.scatter(d.n_ch + dx, d[f"{model}_acc"], s=7, color=col, alpha=0.35, marker=mk, lw=0)
        best = d.groupby("n_ch")[f"{model}_acc"].max()
        ax.plot(best.index + dx, best.values, color=col, ls=ls, marker=mk, label=f"{model}, best subset")
        ax.scatter([2 + dx], [acc.loc["AF7+AF8", f"{model}_acc"]], s=26, marker=mk,
                   facecolor="none", edgecolor=C_BAD, lw=0.8)
    ax.annotate("AF7+AF8\n(frontal pair)", xy=(2.07, acc.loc["AF7+AF8", "MLP_acc"]), xytext=(2.45, 0.415),
                fontsize=5.5, color=C_BAD, arrowprops=dict(arrowstyle="-", color=C_BAD, lw=0.5))
    chance(ax); ax.text(0.65, CHANCE + 0.006, "chance", fontsize=5.5, color=C_GREY, ha="left")
    ax.set_xticks([1, 2, 3, 4]); ax.set_xlim(0.6, 4.4); ax.set_ylim(0.3, 0.7)
    ax.set_xlabel("contacts in array"); ax.set_ylabel("LOSO accuracy")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.07))
    save(fig, "fig1_electrodes")


# ---------------------------------------------------------------- Fig. 2
def mean_psd(filt):
    ps = []
    for fp in sorted(glob.glob(os.path.join(RAW, "subject*-*-*.csv"))):
        for seg in F._segments(pd.read_csv(fp), F.CHANNELS):
            if len(seg) < 512:
                continue
            for c in range(seg.shape[1]):
                x = filt(seg[:, c])
                f, p = welch(x - x.mean(), fs=256, nperseg=512)
                ps.append(p)
    return f, np.mean(ps, 0)


def fig2():
    f, raw = mean_psd(lambda x: x)
    _, nonotch = mean_psd(lambda x: F._bandpass(x, notch=()))
    _, notch = mean_psd(lambda x: F._bandpass(x))
    ref = raw[(f >= 1) & (f <= 45)].max()
    m = (f >= 1) & (f <= 100)
    db = lambda p: 10 * np.log10(np.maximum(p[m], 1e-30) / ref)
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL, 1.65), gridspec_kw=dict(width_ratios=[1.35, 1]))
    a.plot(f[m], db(raw), color=C_GREY, lw=0.7, label="raw")
    a.plot(f[m], db(nonotch), color=C_BAD, lw=0.9, label="no notch")
    a.plot(f[m], db(notch), color=C_GBM, lw=0.9, label="50 Hz notch")
    a.annotate("50 Hz", xy=(50, db(raw)[np.argmin(abs(f[m] - 50))]), xytext=(62, 1),
               fontsize=5.5, ha="left", arrowprops=dict(arrowstyle="-", lw=0.5, color="#444"))
    a.set_xlim(0, 100); a.set_ylim(-70, 8); a.set_xlabel("frequency (Hz)"); a.set_ylabel("power (dB)")
    a.legend(loc="lower left", handlelength=1.4); label(a, "(a)")
    d = pd.read_csv(os.path.join(R, "confound_notch.csv"))
    d = d[d.model == "GBM"].set_index("check")
    rows = [("all", "acc")] + [(s, f"subj_{s}") for s in "abcd"]
    y = np.arange(len(rows))[::-1]
    for yi, (name, col) in zip(y, rows):
        x0 = d.loc["same chain without the notch", col]; x1 = d.loc["corrected chain (50 Hz notch)", col]
        b.plot([x1, x0], [yi, yi], color="#BBBBBB", lw=1.0, zorder=1)
        b.scatter([x0], [yi], s=16, color=C_BAD, zorder=2, label="no notch" if name == "all" else None)
        b.scatter([x1], [yi], s=16, color=C_GBM, zorder=2, label="50 Hz notch" if name == "all" else None)
    chance(b, horizontal=False)
    b.set_yticks(y); b.set_yticklabels(["all"] + [f"subject {s}" for s in "abcd"])
    b.set_xlim(0.2, 1.0); b.set_xlabel("LOSO accuracy (GBM)"); b.grid(axis="y", alpha=0)
    b.set_ylim(-0.6, 5.1)
    b.legend(loc="upper center", bbox_to_anchor=(0.62, 1.03), ncol=2, handletextpad=0.1, columnspacing=0.8)
    label(b, "(b)", x=-0.42)
    save(fig, "fig2_mains")


# ---------------------------------------------------------------- Fig. 3
def fig3():
    n = pd.read_csv(os.path.join(R, "sensor_noise_faults.csv"))
    clean = n[n.degradation == "none"].acc.iloc[0]
    fig, ax = plt.subplots(2, 2, figsize=(COL, 2.75))
    a, b, c, w = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]
    for name, col, mk in [("white noise", C_GBM, "o"), ("50 Hz mains", C_MLP, "s"), ("baseline wander", "#55A868", "^")]:
        g = n[n.degradation == name].sort_values("snr_db", ascending=False)
        a.errorbar(g.snr_db, g.acc, yerr=g.acc_sd, color=col, marker=mk, capsize=1.5, label=name)
    a.axhline(clean, color="#444", ls="--", lw=0.7); chance(a)
    a.set_xlim(21, -1); a.set_xticks([20, 10, 5, 0]); a.set_ylim(0.3, 0.7)
    a.set_xlabel("interference SNR (dB)"); a.set_ylabel("LOSO accuracy"); a.legend(loc="lower left"); label(a, "(a)")
    faults = [("none", "clean", "#444"), ("contact degradation", "50% amp.", C_MLP),
              ("contact degradation", "20% amp.", C_MLP), ("contact dropout", "open", C_BAD)]
    levels = {"clean": "clean", "50% amp.": "50% amplitude", "20% amp.": "20% amplitude", "open": "random contact"}
    for i, (deg, lab, col) in enumerate(faults):
        r = n[(n.degradation == deg) & (n.level == levels[lab])].iloc[0]
        b.errorbar([i], [r.acc], yerr=[r.acc_sd], color=col, marker="o", capsize=2)
    b.axhline(clean, color="#444", ls="--", lw=0.7); chance(b)
    b.set_xticks(range(4)); b.set_xticklabels(["clean", "deg.\n50%", "deg.\n20%", "open"]); b.set_xlim(-0.5, 3.5)
    b.set_ylim(0.3, 0.7); b.set_xlabel("single-contact fault"); label(b, "(b)")
    r = pd.read_csv(os.path.join(R, "sensor_rate.csv"))
    c.plot(range(len(r)), r.acc, color=C_GBM, marker="o"); chance(c)
    c.set_xticks(range(len(r))); c.set_xticklabels([f"{v:.0f}" for v in r.fs_hz]); c.set_ylim(0.3, 0.7)
    c.set_xlabel("converter rate (Hz)"); c.set_ylabel("LOSO accuracy"); label(c, "(c)")
    wd = pd.read_csv(os.path.join(R, "sensor_window.csv"))
    for fs, col, mk, ls in [(256.0, C_GBM, "o", "-"), (112.0, C_MLP, "s", "-")]:   # dashed is reserved for the clean reference
        g = wd[wd.fs_hz == fs].sort_values("win_s")
        w.plot(g.win_s, g.acc, color=col, marker=mk, ls=ls, label=f"{fs:.0f} Hz")
    chance(w); w.set_xscale("log", base=2); w.set_xticks([0.5, 1, 2]); w.set_xticklabels(["0.5", "1", "2"])
    w.set_ylim(0.3, 0.7); w.set_xlabel("decision window (s)"); w.legend(loc="lower right"); label(w, "(d)")
    save(fig, "fig3_sensor")


# ---------------------------------------------------------------- tables
def table1():
    d = pd.read_csv(os.path.join(R, "sensor_electrode_subsets.csv"))
    sd = d.MLP_acc_sd.max()
    t = [r"\begin{table}[!t]", r"\caption{Cross-subject accuracy of every electrode subset}",
         r"\label{tab:subsets}", r"\centering\footnotesize", r"\setlength{\tabcolsep}{4.5pt}",
         r"\begin{tabular}{lcccc}", r"\toprule",
         r" & \multicolumn{3}{c}{GBM} & MLP \\", r"\cmidrule(lr){2-4}\cmidrule(lr){5-5}",
         r"Array & Acc. & Macro-F1 & ECE & Acc. \\", r"\midrule"]
    for k in [4, 3, 2, 1]:
        g = d[d.n_ch == k].sort_values("GBM_acc", ascending=False)
        for _, r in g.iterrows():
            cell = lambda v, best: (r"\textbf{%.3f}" % v) if (best and len(g) > 1) else "%.3f" % v
            t.append(f"{r.electrodes} & {cell(r.GBM_acc, r.GBM_acc == g.GBM_acc.max())} & {r.GBM_f1:.3f} & "
                     f"{r.GBM_ece:.3f} & {cell(r.MLP_acc, r.MLP_acc == g.MLP_acc.max())} \\\\")
        t.append(r"\midrule" if k > 1 else r"\bottomrule")
    t += [r"\end{tabular}", r"\par\smallskip",
          r"\parbox{\columnwidth}{\footnotesize Bold: best subset of each size; MLP s.d.\ over five seeds $\le$ %.3f.}" % sd,
          r"\end{table}"]
    open(os.path.join(FG, "table1_subsets.tex"), "w").write("\n".join(t) + "\n"); print("wrote table1_subsets")


def table2():
    j = json.load(open(os.path.join(R, "sensor_compute_budget.json")))
    fe = j["feature_extraction_ms"]; pair = j["pair"]
    t = [r"\begin{table}[!t]", r"\caption{Compute budget of the sensor node}", r"\label{tab:compute}",
         r"\centering\footnotesize", r"\begin{tabular}{lccc}", r"\toprule",
         r"Decoder & Model (kB) & Inference ($\mu$s) & Acc. \\", r"\midrule"]
    for cfg in ["full", "compact"]:
        for tag in ["4-electrode", "2-electrode"]:
            r = j["decoders"][f"{cfg} / {tag}"]
            t.append(f"{cfg.capitalize()}, {tag[0]} contacts & {r['model_kb']:.0f} & "
                     f"{1000 * r['inference_ms_per_window']:.1f} & {r['loso_acc']:.3f} \\\\")
    t += [r"\midrule",
          r"\multicolumn{4}{l}{Feature extraction: %.1f ms (4 contacts), %.1f ms (2 contacts)} \\"
          % (fe["4-electrode"], fe["2-electrode"]),
          r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
          r"\parbox{\columnwidth}{\footnotesize Per 1\,s window, single-threaded on the host CPU; "
          r"two contacts: %s. Timings vary between runs and are meaningful as ratios.}" % pair,
          r"\end{table}"]
    open(os.path.join(FG, "table2_compute.tex"), "w").write("\n".join(t) + "\n"); print("wrote table2_compute")



# ---------------------------------------------------------------- graphical abstract
def graphical_abstract():
    d = pd.read_csv(os.path.join(R, "sensor_electrode_subsets.csv")).set_index("electrodes")
    nt = pd.read_csv(os.path.join(R, "confound_notch.csv"))
    nt = nt[nt.model == "GBM"].set_index("check")
    full = d.loc["TP9+AF7+AF8+TP10"]
    best = {m: d[d.n_ch == 2][f"{m}_acc"].max() for m in ["GBM", "MLP"]}
    fig = plt.figure(figsize=(6.6, 2.95))             # IEEE graphical-abstract aspect, 660 x 295
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 1.35], wspace=0.45, left=0.03, right=0.95, bottom=0.2, top=0.86)
    h = fig.add_subplot(gs[0, 0])
    mini_head(h, ["TP9", "TP10"], d.loc["TP9+TP10", "GBM_acc"], d.loc["TP9+TP10", "MLP_acc"])
    h.texts[-1].remove(); h.texts[-1].remove()
    for name, (x, y) in SITES.items():
        if name in ("AF7", "AF8"):
            h.plot([x - 0.12, x + 0.12], [y - 0.12, y + 0.12], color=C_BAD, lw=1.0)
            h.plot([x - 0.12, x + 0.12], [y + 0.12, y - 0.12], color=C_BAD, lw=1.0)
    h.set_title("4 contacts $\\rightarrow$ 2", fontsize=10, color=C_GBM, fontweight="bold", pad=2)
    t = fig.add_subplot(gs[0, 1]); t.set_axis_off()
    t.text(0.5, 0.93, "Two contacts match four", ha="center", va="top", fontsize=10, fontweight="bold", color="#222")
    for i, (m, col) in enumerate([("GBM", C_GBM), ("MLP", C_MLP)]):
        t.text(0.5, 0.62 - 0.30 * i, f"{m}:  {best[m]:.2f}  vs  {full[f'{m}_acc']:.2f}", ha="center",
               va="center", fontsize=12, color=col, fontweight="bold")
    t.text(0.5, 0.0, "best pair vs full array, cross-subject accuracy", ha="center", va="bottom",
           fontsize=7, color="#555")
    b = fig.add_subplot(gs[0, 2])
    vals = [nt.loc["same chain without the notch", "acc"], nt.loc["corrected chain (50 Hz notch)", "acc"]]
    b.barh([1, 0], vals, color=[C_BAD, C_GBM], height=0.55)
    for yy, v in zip([1, 0], vals):
        b.text(v + 0.015, yy, f"{v:.2f}", va="center", fontsize=9, fontweight="bold")
    b.axvline(CHANCE, color=C_GREY, ls=":", lw=0.8)
    b.set_yticks([1, 0]); b.set_yticklabels(["no notch", "50 Hz notch"]); b.set_xlim(0, 1)
    b.set_xlabel("cross-subject accuracy (GBM)", fontsize=8); b.set_ylim(-0.5, 1.6); b.grid(axis="y", alpha=0)
    b.set_title("Mains pickup inflates accuracy", fontsize=9.5, fontweight="bold", color=C_BAD, pad=4)
    b.tick_params(labelsize=8)
    b.text(CHANCE + 0.01, 1.42, "chance", fontsize=7, color=C_GREY)
    for ext in ("pdf", "png"):                         # keep the exact aspect: no tight bbox
        fig.savefig(os.path.join(FG, f"graphical_abstract.{ext}"), dpi=100)
    plt.close(fig)
    print("wrote graphical_abstract (660 x 295)")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); table1(); table2(); graphical_abstract()

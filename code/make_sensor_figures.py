"""Figures for the sensor-level characterisation (S1-S4).

Sized for a single IEEE two-column text column (3.4 in wide), so no
down-scaling happens in the manuscript and the type stays legible.
"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    from pro_style import apply as _pa; _pa()
except Exception:
    pass
matplotlib.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6,
    "lines.linewidth": 1.2, "lines.markersize": 3.5, "axes.linewidth": 0.7,
})

R = os.path.join(HERE, "..", "results")
FG = os.path.join(HERE, "..", "figures")
os.makedirs(FG, exist_ok=True)
COL = 3.4                       # IEEE single-column width (in)
C_ACC, C_ECE, C_REF = "#2E5A87", "#CC8963", "#C44E52"


def save(fig, name):
    fig.tight_layout(pad=0.25)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FG, f"{name}.{ext}"), dpi=400,
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---- Fig. 4: electrode-count reduction -----------------------------------
d = pd.read_csv(os.path.join(R, "sensor_electrode_ablation.csv"))
full = d[d.n_ch == 4].iloc[0]
best = d.sort_values("acc", ascending=False).groupby("n_ch").first().reset_index()

fig, ax = plt.subplots(figsize=(COL, 2.05))
for k in sorted(d.n_ch.unique()):
    sub = d[d.n_ch == k]
    ax.scatter([k] * len(sub), sub.acc, s=9, alpha=0.5, color="0.6", zorder=2,
               label="all subsets" if k == 1 else None)
ax.plot(best.n_ch, best.acc, "o-", color=C_ACC, zorder=3, label="best subset")
ax.axhline(full.acc, ls="--", lw=0.9, color=C_REF, zorder=1,
           label="full 4-electrode array")
ax.axhline(1 / 3, ls=":", lw=0.9, color="0.35", zorder=1, label="chance")
for _, row in best.iterrows():
    if row.n_ch < 4:
        ax.annotate(row.electrodes, (row.n_ch, row.acc), fontsize=5.6,
                    textcoords="offset points", xytext=(4, -8))
ax.set_xticks([1, 2, 3, 4])
ax.set_xlabel("electrodes in array"); ax.set_ylabel("LOSO accuracy")
ax.set_ylim(0.28, 0.86)

axr = ax.twinx()
axr.plot(best.n_ch, best.ece, "s--", color=C_ECE, ms=3, lw=1.0,
         label="ECE (best subset)")
axr.set_ylabel("cross-subject ECE", color=C_ECE)
axr.tick_params(axis="y", colors=C_ECE, labelsize=6.5)
axr.grid(False); axr.spines["right"].set_visible(True)
axr.spines["right"].set_color(C_ECE); axr.spines["right"].set_linewidth(0.7)

h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axr.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="lower right", ncol=1, fontsize=5.6,
          handlelength=1.6, borderpad=0.2, labelspacing=0.25)
save(fig, "fig4_electrode_ablation")


# ---- Fig. 5: interference resilience -------------------------------------
n = pd.read_csv(os.path.join(R, "sensor_noise_resilience.csv"))
clean = n[n.degradation == "none"].iloc[0]
types = [("white noise", "#2E5A87", "o"),
         ("50 Hz mains", "#55A868", "^"),
         ("baseline wander", "#8172B3", "v")]

fig, ax = plt.subplots(1, 2, figsize=(COL, 1.75), sharex=True)
for t, c, m in types:
    sub = n[n.degradation == t].sort_values("snr_db", ascending=False)
    if not len(sub):
        continue
    ax[0].errorbar(sub.snr_db, sub.acc, yerr=sub.acc_sd, fmt=m + "-", color=c,
                   capsize=1.5, elinewidth=0.7, label=t)
    ax[1].errorbar(sub.snr_db, sub.ece, yerr=sub.ece_sd, fmt=m + "-", color=c,
                   capsize=1.5, elinewidth=0.7, label=t)
ax[0].axhline(clean.acc, ls="--", lw=0.9, color=C_REF, label="clean")
ax[1].axhline(clean.ece, ls="--", lw=0.9, color=C_REF)
ax[0].axhline(1 / 3, ls=":", lw=0.9, color="0.35", label="chance")
for a in ax:
    a.set_xlabel("interference SNR (dB)")
ax[0].invert_xaxis()          # sharex -> one call flips both; two would cancel
ax[0].set_ylabel("LOSO accuracy"); ax[1].set_ylabel("cross-subject ECE")
ax[0].legend(loc="lower left", fontsize=5.4, handlelength=1.5,
             borderpad=0.2, labelspacing=0.22)
save(fig, "fig5_noise_resilience")


# ---- Fig. 6: electrode faults + rate/window budget ------------------------
drop = n[n.degradation == "electrode dropout"]
cont = n[n.degradation == "contact degradation"]
rw = pd.read_csv(os.path.join(R, "sensor_rate_window.csv"))

fig, ax = plt.subplots(1, 2, figsize=(COL, 1.8))
labels = [f"open {l}" for l in drop.level] + \
         [f"degraded {l.split()[0]}" for l in cont.level]
vals = list(drop.acc) + list(cont.acc)
errs = list(drop.acc_sd) + list(cont.acc_sd)
ax[0].bar(range(len(vals)), vals, yerr=errs, capsize=1.5,
          error_kw=dict(elinewidth=0.7),
          color=["#2E5A87"] * len(drop) + ["#C44E52"] * len(cont), width=0.72)
ax[0].axhline(clean.acc, ls="--", lw=0.9, color="0.2", label="no fault")
ax[0].axhline(1 / 3, ls=":", lw=0.9, color="0.4", label="chance")
ax[0].set_xticks(range(len(vals)))
ax[0].set_xticklabels(labels, fontsize=5.2, rotation=38, ha="right")
ax[0].set_ylabel("LOSO accuracy"); ax[0].set_ylim(0.28, 0.86)
ax[0].set_title("single-electrode fault", fontsize=6.8, pad=2)
ax[0].legend(loc="lower left", fontsize=5.4, handlelength=1.5, borderpad=0.2)

for fs, c, m in zip(sorted(rw.fs_hz.unique(), reverse=True),
                    ["#2E5A87", "#CC8963", "#8C8C8C"], ["o", "s", "^"]):
    sub = rw[rw.fs_hz == fs].sort_values("win_s")
    ax[1].plot(sub.win_s, sub.acc, m + "-", color=c, label=f"{fs:.0f} Hz")
ax[1].axhline(1 / 3, ls=":", lw=0.9, color="0.4")
ax[1].set_xlabel("decision window (s)"); ax[1].set_ylabel("LOSO accuracy")
ax[1].set_xticks([0.5, 1.0, 2.0]); ax[1].set_ylim(0.28, 0.86)
ax[1].set_title("rate / latency budget", fontsize=6.8, pad=2)
ax[1].legend(loc="lower right", fontsize=5.4, handlelength=1.5, borderpad=0.2,
             title="ADC rate", title_fontsize=5.4)
save(fig, "fig6_fault_and_budget")
print("all sensor figures done")

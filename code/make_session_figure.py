"""Figure: the labels are confounded with the recording session.

(a) when each recording was made, per participant
(b) a decoder that sees no EEG against the full decoder
(c) per-participant accuracy against whether that participant has the day split
(d) the contrast that is free of the confound
"""
import os, json, datetime as dt
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
FG = os.path.join(HERE, "..", "manuscript")
try:
    import sys; sys.path.insert(0, "/home/pardus/Papers/_templates")
    from pro_style import apply as _pa; _pa()
except Exception:
    pass

NAVY, RED, GREY, ORANGE = "#2E5A87", "#C44E52", "#8C8C8C", "#CC8963"
STATE_C = {"relaxed": "#7F9BB8", "neutral": "#9FB3A8", "concentrating": RED}

tab = pd.read_csv(os.path.join(R, "session_recording_table.csv"))
split = pd.read_csv(os.path.join(R, "session_split.csv")).set_index("subject")
acc = pd.read_csv(os.path.join(R, "session_split_vs_accuracy.csv"))
within = pd.read_csv(os.path.join(R, "within_session_relaxed_vs_neutral.csv"))
tonly = json.load(open(os.path.join(R, "time_only_baseline.json")))
notch = pd.read_csv(os.path.join(R, "confound_notch.csv"))
gbm_notch = notch[(notch.model == "GBM") & notch.check.str.contains("corrected")].acc.iloc[0]

fig = plt.figure(figsize=(7.1, 5.2))

# ---- (a) recording timeline -------------------------------------------------
ax = fig.add_axes([0.085, 0.575, 0.40, 0.345])
subs = sorted(tab.subject.unique())
for i, s in enumerate(subs):
    g = tab[tab.subject == s].copy()
    t0 = g.t0.min()
    g["h"] = (g.t0 - t0) / 3600.0
    for _, r in g.iterrows():
        ax.scatter(r.h, i, s=46, color=STATE_C[r.state], zorder=3,
                   edgecolor="white", linewidth=0.7)
ax.set_yticks(range(len(subs)))
ax.set_yticklabels([f"participant {s}" for s in subs], fontsize=7.5)
ax.set_xlabel("hours from that participant's first recording", fontsize=8)
ax.axvspan(40, 50, color="0.92", zorder=0)
ax.text(45, len(subs) - 0.45, "second day", ha="center", fontsize=7, color="0.35")
ax.set_xlim(-3, 52); ax.tick_params(labelsize=7.5)
h = [plt.Line2D([], [], marker="o", ls="", color=c, label=k, markersize=5.5)
     for k, c in STATE_C.items()]
ax.legend(handles=h, fontsize=6.8, loc="center right", frameon=False)
ax.set_title("a   when each recording was made", fontsize=8.6, loc="left",
             weight="bold")

# ---- (b) EEG-free baseline --------------------------------------------------
ax = fig.add_axes([0.605, 0.575, 0.355, 0.345])
vals = [tonly["time_only_loso_acc"], gbm_notch]
labs = ["time only\n(no EEG)", "full decoder\n(notched EEG)"]
ax.bar([0, 1], vals, color=[ORANGE, NAVY], width=0.58)
for x, v in zip([0, 1], vals):
    ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=8.5, weight="bold")
ax.axhline(1/3, ls=":", color=GREY, lw=1.1)
ax.text(1.42, 1/3 + 0.008, "chance", fontsize=7, color=GREY, ha="right")
ax.set_xticks([0, 1]); ax.set_xticklabels(labs, fontsize=7.6)
ax.set_ylim(0, 0.70); ax.set_ylabel("LOSO accuracy", fontsize=8)
ax.tick_params(labelsize=7.5)
ax.set_title("b   a clock does almost all of it", fontsize=8.6, loc="left",
             weight="bold")

# ---- (c) split vs accuracy --------------------------------------------------
ax = fig.add_axes([0.085, 0.085, 0.40, 0.355])
w = 0.36
for k, (model, col) in enumerate([("GBM", NAVY), ("MLP", "#7F9BB8")]):
    g = acc[acc.model == model].set_index("subject").loc[subs]
    ax.bar(np.arange(len(subs)) + (k - 0.5) * w, g.loso_acc, width=w,
           color=col, label=model)
ax.axhline(1/3, ls=":", color=GREY, lw=1.1)
for i, s in enumerate(subs):
    if not split.loc[s, "separate_session_for_concentrating"]:
        ax.annotate("no day split", (i, 0.045), ha="center", fontsize=7,
                    color=RED, weight="bold")
ax.set_xticks(range(len(subs))); ax.set_xticklabels(subs, fontsize=8)
ax.set_xlabel("held-out participant", fontsize=8)
ax.set_ylabel("LOSO accuracy", fontsize=8)
ax.set_ylim(0, 0.82); ax.tick_params(labelsize=7.5)
ax.legend(fontsize=7, frameon=False, loc="upper right", ncol=2)
ax.set_title("c   the split predicts the accuracy", fontsize=8.6, loc="left",
             weight="bold")

# ---- (d) confound-free contrast --------------------------------------------
ax = fig.add_axes([0.605, 0.085, 0.355, 0.355])
g = within.set_index("model")
xs = np.arange(len(subs))
for k, (model, col) in enumerate([("GBM", NAVY), ("MLP", "#7F9BB8")]):
    v = [g.loc[model, f"subj_{s}"] for s in subs]
    ax.bar(xs + (k - 0.5) * w, v, width=w, color=col, label=model)
ax.axhline(0.5, ls=":", color=GREY, lw=1.1)
ax.text(len(subs) - 0.4, 0.508, "chance", fontsize=7, color=GREY, ha="right")
ax.set_xticks(xs); ax.set_xticklabels(subs, fontsize=8)
ax.set_xlabel("held-out participant", fontsize=8)
ax.set_ylabel("LOSO accuracy", fontsize=8)
ax.set_ylim(0.4, 0.78); ax.tick_params(labelsize=7.5)
ax.legend(fontsize=7, frameon=False, loc="upper right", ncol=2)
ax.set_title("d   relaxed vs neutral (same session)", fontsize=8.6, loc="left",
             weight="bold")

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FG, f"Figure1_session.{ext}"), dpi=400,
                bbox_inches="tight")
print("wrote Figure1_session")

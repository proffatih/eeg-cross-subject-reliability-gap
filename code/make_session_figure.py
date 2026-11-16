"""Figure: the labels are confounded with the recording session.

(a) when each recording was made, per participant
(b) a decoder that sees no EEG against the full decoder
(c) per-participant accuracy against whether that participant has the day split
(d) the contrast that is free of the day split, under the same three decoders
(e) 50 Hz power of every recording against when it was made, by state
"""
import os, json, datetime as dt
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
FG = os.path.join(HERE, "..", "manuscript")
os.makedirs(FG, exist_ok=True)
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

fig = plt.figure(figsize=(7.1, 7.5))

# ---- (a) recording timeline -------------------------------------------------
ax = fig.add_axes([0.085, 0.725, 0.40, 0.235])
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

# ---- (b) decoders that see no in-band EEG -----------------------------------
ax = fig.add_axes([0.605, 0.725, 0.355, 0.235])
art = pd.read_csv(os.path.join(R, "artifact_model_accuracy.csv")).set_index("model")
art_acc = float(art.loc["artifact only (50 Hz + DC + 55-128 Hz)", "acc"])
vals = [tonly["time_only_loso_acc"], gbm_notch, art_acc]
labs = ["clock only\n(no EEG)", "full decoder\n(notched EEG)",
        "artifacts only\n(no in-band EEG)"]
ax.bar([0, 1, 2], vals, color=[ORANGE, NAVY, RED], width=0.6)
for x, v in zip([0, 1, 2], vals):
    ax.text(x, v + 0.014, f"{v:.3f}", ha="center", fontsize=8.2, weight="bold")
ax.axhline(1/3, ls=":", color=GREY, lw=1.1)
ax.text(2.45, 1/3 + 0.01, "chance", fontsize=7, color=GREY, ha="right")
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(labs, fontsize=7.0)
ax.set_ylim(0, 0.88); ax.set_ylabel("LOSO accuracy", fontsize=8)
ax.tick_params(labelsize=7.5)
ax.set_title("b   what the in-band EEG adds", fontsize=8.6, loc="left",
             weight="bold")

# ---- (c) split vs accuracy --------------------------------------------------
ax = fig.add_axes([0.085, 0.395, 0.40, 0.245])
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

# ---- (d) the same three decoders on the confound-free contrast --------------
ax = fig.add_axes([0.605, 0.395, 0.355, 0.245])
wb = pd.read_csv(os.path.join(R, "within_session_baselines.csv")).set_index("model")
vals_d = [float(wb.loc["clock only (no EEG)", "acc"]),
          float(wb.loc["in-band EEG (full decoder)", "acc"]),
          float(wb.loc["artifact only (50 Hz + DC + 55-128 Hz)", "acc"])]
ax.bar([0, 1, 2], vals_d, color=[ORANGE, NAVY, RED], width=0.6)
for x, v in zip([0, 1, 2], vals_d):
    ax.text(x, v + 0.016, f"{v:.3f}", ha="center", fontsize=8.2, weight="bold")
ax.axhline(0.5, ls=":", color=GREY, lw=1.1)
ax.text(2.45, 0.515, "chance", fontsize=7, color=GREY, ha="right")
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["clock only\n(no EEG)", "full decoder\n(notched EEG)",
                    "artifacts only\n(no in-band EEG)"], fontsize=7.0)
ax.set_ylim(0, 1.0); ax.set_ylabel("LOSO accuracy", fontsize=8)
ax.tick_params(labelsize=7.5)
ax.set_title("d   relaxed vs neutral: same pattern", fontsize=8.6, loc="left",
             weight="bold")

# ---- (e) 50 Hz power against acquisition time, by state --------------------
# Split axis: the whole point is a slow rise inside day 1 and a large step at
# the day boundary. On one linear axis spanning 47 h the day-1 spread collapses
# into the left edge, so day 1 and day 2 are drawn on their own scales.
mt = pd.read_csv(os.path.join(R, "mains_vs_time.csv"))
MARK = {"a": "o", "b": "s", "c": "^", "d": "D"}
axL = fig.add_axes([0.085, 0.055, 0.655, 0.245])
axR = fig.add_axes([0.775, 0.055, 0.185, 0.245], sharey=axL)
d1, d2 = mt[~mt.second_day], mt[mt.second_day]
for ax, g in ((axL, d1), (axR, d2)):
    for _, r in g.iterrows():
        ax.scatter(r.hours_from_first, r.mains_log10_uV2, s=52,
                   color=STATE_C[r.state], marker=MARK[r.subject], zorder=3,
                   edgecolor="white", linewidth=0.7)
for sj in sorted(d1.subject.unique()):
    g = d1[d1.subject == sj].sort_values("hours_from_first")
    axL.plot(g.hours_from_first, g.mains_log10_uV2, color="0.78", lw=0.9, zorder=1)
axL.set_xlim(-0.25, 4.1); axR.set_xlim(45.0, 47.2)
axR.tick_params(labelleft=False, labelsize=7.5)
axL.tick_params(labelsize=7.5)
axL.spines["right"].set_visible(False); axR.spines["left"].set_visible(False)
for x0, ax in ((1.0, axL), (0.0, axR)):          # axis-break marks
    ax.plot([x0], [0], transform=ax.transAxes, clip_on=False, marker=(2, 0, 70),
            ms=7, color="0.45", mew=1.0)
    ax.plot([x0], [1], transform=ax.transAxes, clip_on=False, marker=(2, 0, 70),
            ms=7, color="0.45", mew=1.0)
axL.set_ylabel("50 Hz power  $\\log_{10}\\,\\mu V^2$", fontsize=8)
axL.set_xlabel("hours from that participant's first recording   (day 1)",
               fontsize=8)
axR.set_xlabel("day 2", fontsize=8)
hs = [plt.Line2D([], [], marker="o", ls="", color=c, label=k, markersize=5.5)
      for k, c in STATE_C.items()]
hs += [plt.Line2D([], [], marker=m, ls="", color="0.45", label=k, markersize=5.0)
       for k, m in MARK.items()]
axL.legend(handles=hs, fontsize=6.6, loc="upper left", frameon=False, ncol=4,
           columnspacing=0.9, handletextpad=0.3)
axL.set_title("e   the line level tracks the clock, not the task", fontsize=8.6,
              loc="left", weight="bold")

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FG, f"Figure1_session.{ext}"), dpi=400,
                bbox_inches="tight")
print("wrote Figure1_session")

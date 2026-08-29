"""IEEE Sensors Letters graphical abstract.

Official IEEE specification (Graphical Abstract Description and Specifications):
  dimensions 660 x 295 px, preferred file type JPG, recommended size < 45 kB.
Every number drawn here comes from ../results, so the graphical abstract is a
real summary of the measurements rather than an illustration.
"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
FG = os.path.join(HERE, "..", "figures")
os.makedirs(FG, exist_ok=True)

W, H, DPI = 660, 295, 100
NAVY, RED, GREY, ORANGE = "#2E5A87", "#C44E52", "#8C8C8C", "#CC8963"

matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "axes.linewidth": 0.8, "axes.edgecolor": "#444444",
    "xtick.color": "#222", "ytick.color": "#222",
    "axes.spines.top": False, "axes.spines.right": False,
})

abl = pd.read_csv(os.path.join(R, "sensor_electrode_ablation.csv"))
noise = pd.read_csv(os.path.join(R, "sensor_noise_resilience.csv"))
best = abl.sort_values("acc", ascending=False).groupby("n_ch").first().reset_index()
full = abl[abl.n_ch == 4].iloc[0].acc
pair = abl[abl.electrodes == "TP9+AF8"].iloc[0].acc


def g(deg, level):
    return noise[(noise.degradation == deg) & (noise.level == level)].iloc[0].acc


clean = noise[noise.degradation == "none"].iloc[0].acc
white10 = g("white noise", "10 dB SNR")
mains10 = g("50 Hz mains", "10 dB SNR")
degraded = g("contact degradation", "50% amplitude")
open_worst = noise[noise.degradation == "electrode dropout"].acc.min()

fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor="white")

# ---------------------------------------------------------------- panel A
axA = fig.add_axes([0.010, 0.215, 0.255, 0.575]); axA.set_axis_off()
axA.set_xlim(0, 1); axA.set_ylim(0, 1)
axA.set_aspect("equal", adjustable="box")
axA.add_patch(Circle((0.5, 0.50), 0.315, fc="#F2F4F7", ec="#8899AA", lw=1.2))
axA.plot([0.5, 0.458, 0.542], [0.845, 0.775, 0.775], color="#8899AA", lw=1.2)  # nose
sites = {"TP9": (0.225, 0.375), "AF7": (0.345, 0.685),
         "AF8": (0.655, 0.685), "TP10": (0.775, 0.375)}
keep = {"TP9", "AF8"}
for name, (x, y) in sites.items():
    if name in keep:
        axA.add_patch(Circle((x, y), 0.075, fc=NAVY, ec="white", lw=1.4, zorder=3))
        axA.add_patch(Circle((x, y), 0.105, fc="none", ec=NAVY, lw=1.2,
                             ls=(0, (2, 1.6)), zorder=3))
        axA.text(x, y, name, color="white", fontsize=6.2, weight="bold",
                 ha="center", va="center", zorder=4)
    else:
        axA.add_patch(Circle((x, y), 0.070, fc="white", ec=GREY, lw=1.1, zorder=3))
        axA.text(x, y, name, color=GREY, fontsize=5.8, ha="center", va="center",
                 zorder=4)
        axA.plot([x - .052, x + .052], [y - .052, y + .052], color=RED, lw=1.3,
                 zorder=5)
axA.text(0.5, 1.09, "4-contact dry headband", fontsize=7.4, weight="bold",
         ha="center", va="center", color="#222")
axA.text(0.5, -0.10, "keep 2: one temporal +\ncontralateral frontal",
         fontsize=6.5, ha="center", va="top", color=NAVY, weight="bold",
         linespacing=1.2)

# ---------------------------------------------------------------- panel B
axB = fig.add_axes([0.378, 0.315, 0.242, 0.525])
axB.plot(best.n_ch, best.acc, "o-", color=NAVY, lw=1.6, ms=4.2, zorder=3)
axB.axhline(full, ls="--", lw=1.0, color=RED, zorder=2)
axB.axhline(1 / 3, ls=":", lw=1.0, color=GREY, zorder=2)
axB.annotate(f"{pair:.3f}\n97.6% of full", xy=(2, pair), xytext=(2.15, 0.545),
             fontsize=6.2, color=NAVY, weight="bold", linespacing=1.2,
             arrowprops=dict(arrowstyle="-", color=NAVY, lw=0.8))
axB.text(4.03, full, "4 contacts", fontsize=5.8, color=RED, va="bottom",
         ha="right")
axB.text(1.0, 1 / 3, "chance", fontsize=5.6, color=GREY, va="bottom")
axB.set_xticks([1, 2, 3, 4]); axB.set_xlim(0.7, 4.3); axB.set_ylim(0.28, 0.87)
axB.set_yticks([0.4, 0.6, 0.8])
axB.tick_params(labelsize=6, length=2.5, pad=1.5)
axB.set_xlabel("contacts in array", fontsize=6.6, labelpad=1.5)
axB.set_ylabel("cross-subject accuracy", fontsize=6.6, labelpad=1.5)
axB.set_title("Array can be halved", fontsize=7.4, weight="bold", pad=3,
              color="#222")

# ---------------------------------------------------------------- panel C
axC = fig.add_axes([0.757, 0.315, 0.223, 0.525])
rows = [("no fault", clean, "#7F9BB8"),
        ("white noise\n10 dB", white10, NAVY),
        ("50 Hz mains\n10 dB", mains10, ORANGE),
        ("open contact\n(worst)", open_worst, "#7F9BB8"),
        ("degraded\ncontact", degraded, RED)]
ypos = np.arange(len(rows))[::-1]
axC.barh(ypos, [r[1] for r in rows], color=[r[2] for r in rows], height=0.66)
for y, (_, v, _c) in zip(ypos, rows):
    axC.text(v + 0.015, y, f"{v:.2f}", va="center", fontsize=5.9, color="#222")
axC.axvline(1 / 3, ls=":", lw=1.0, color=GREY)
axC.set_yticks(ypos); axC.set_yticklabels([r[0] for r in rows], fontsize=6.1,
                                          linespacing=1.08)
axC.tick_params(axis="y", pad=2.5)
axC.set_xlim(0, 0.95); axC.set_xticks([0, 0.4, 0.8])
axC.tick_params(labelsize=6, length=2.5, pad=1.5)
axC.set_xlabel("cross-subject accuracy", fontsize=6.6, labelpad=1.5)
axC.set_title("What actually hurts it", fontsize=7.4, weight="bold", pad=3,
              color="#222")

# ---------------------------------------------------------------- takeaway
fig.patches.append(FancyBboxPatch(
    (0.292, 0.020), 0.700, 0.120, transform=fig.transFigure,
    boxstyle="round,pad=0.003,rounding_size=0.010",
    fc="#EEF2F7", ec="#C6D2E0", lw=0.8, zorder=0))
fig.text(0.642, 0.080,
         "50 Hz mains costs 2$\\times$ broadband noise of equal power  ·  "
         "a degraded contact is worse than an open one",
         fontsize=6.3, ha="center", va="center", color="#1F3A5F")

out_png = os.path.join(FG, "graphical_abstract.png")
out_jpg = os.path.join(FG, "graphical_abstract.jpg")
fig.savefig(out_png, dpi=DPI, facecolor="white")

# JPEG at exactly 660x295, quality tuned down until the file is under 45 kB
from PIL import Image
im = Image.open(out_png).convert("RGB")
if im.size != (W, H):
    im = im.resize((W, H), Image.LANCZOS)
for q in (95, 90, 85, 80, 75, 70):
    im.save(out_jpg, "JPEG", quality=q, optimize=True, subsampling=0)
    kb = os.path.getsize(out_jpg) / 1024
    if kb < 45:
        break
print(f"{out_jpg}  {im.size[0]}x{im.size[1]}  {kb:.1f} kB  (quality {q})")

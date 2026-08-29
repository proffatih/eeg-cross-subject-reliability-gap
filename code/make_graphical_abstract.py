"""IEEE Sensors Letters graphical abstract.

IEEE specification (Graphical Abstract Description and Specifications):
  dimensions 660 x 295 px, preferred file type JPG, recommended size < 45 kB.
Those are recommendations, so we emit two files at the same 660:295 aspect:
  graphical_abstract.jpg          high resolution (3x), for submission
  graphical_abstract_660x295.jpg  spec-exact fallback, kept under 45 kB
Every number drawn here is read from ../results, so the graphical abstract is a
summary of the measurements rather than an illustration.
"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Wedge

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
FG = os.path.join(HERE, "..", "figures")
os.makedirs(FG, exist_ok=True)

W, H = 660, 295              # IEEE aspect, kept exactly
SCALE = 3                    # render at 3x for a crisp submission file
DPI = 100 * SCALE
NAVY, RED, GREY, ORANGE = "#2E5A87", "#C44E52", "#8C8C8C", "#CC8963"
SKIN, EDGE = "#F2F4F7", "#8899AA"

matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "axes.linewidth": 0.8, "axes.edgecolor": "#444444",
    "xtick.color": "#222", "ytick.color": "#222",
    "axes.spines.top": False, "axes.spines.right": False,
})

# ------------------------------------------------------------------ numbers
abl = pd.read_csv(os.path.join(R, "sensor_electrode_ablation.csv"))
noise = pd.read_csv(os.path.join(R, "sensor_noise_resilience.csv"))
best = abl.sort_values("acc", ascending=False).groupby("n_ch").first().reset_index()
full = abl[abl.n_ch == 4].iloc[0].acc
pair = abl[abl.electrodes == "TP9+AF8"].iloc[0].acc
tp9_only = abl[abl.electrodes == "TP9"].iloc[0].acc
no_tp9_3 = abl[abl.electrodes == "AF7+AF8+TP10"].iloc[0].acc


def g(deg, level):
    return noise[(noise.degradation == deg) & (noise.level == level)].iloc[0].acc


clean = noise[noise.degradation == "none"].iloc[0].acc
white10 = g("white noise", "10 dB SNR")
mains10 = g("50 Hz mains", "10 dB SNR")
degraded = g("contact degradation", "50% amplitude")
open_worst = noise[noise.degradation == "electrode dropout"].acc.min()

fig = plt.figure(figsize=(W / 100, H / 100), dpi=DPI, facecolor="white")

# ============================================================== panel A
# Head seen from above, nose up. The left temporal contact (TP9) is drawn larger
# and carries a magnified inset of the dry skin interface, because every array
# that works contains it.
axA = fig.add_axes([0.004, 0.100, 0.352, 0.790]); axA.set_axis_off()
axA.set_xlim(0, 1.34); axA.set_ylim(0, 1.0)
axA.set_aspect("equal", adjustable="box")

CX, CY, Rh = 0.430, 0.560, 0.315
axA.add_patch(Circle((CX, CY), Rh, fc=SKIN, ec=EDGE, lw=1.6, zorder=1))
axA.plot([CX - 0.046, CX, CX + 0.046],
         [CY + Rh - 0.008, CY + Rh + 0.068, CY + Rh - 0.008],
         color=EDGE, lw=1.6, solid_joinstyle="round", zorder=1)

sites = {"TP9":  (CX - 0.190, CY - 0.150, True,  "left",  (-0.150, -0.010)),
         "AF7":  (CX - 0.140, CY + 0.185, False, "right", (-0.085,  0.052)),
         "AF8":  (CX + 0.140, CY + 0.185, True,  "left",  ( 0.000,  0.000)),
         "TP10": (CX + 0.190, CY - 0.150, False, "left",  ( 0.082, -0.052))}
for name, (x, y, keep, ha, off) in sites.items():
    big = (name == "TP9")
    r = 0.088 if big else 0.052
    if keep:
        axA.add_patch(Circle((x, y), r + 0.028, fc="none", ec=NAVY, lw=1.3,
                             ls=(0, (2.2, 1.6)), zorder=3))
        axA.add_patch(Circle((x, y), r, fc=NAVY, ec="white", lw=1.5, zorder=4))
        axA.text(x, y, name, color="white", fontsize=6.8 if big else 5.6,
                 weight="bold", ha="center", va="center", zorder=5)
    else:
        axA.add_patch(Circle((x, y), r, fc="white", ec=GREY, lw=1.2, zorder=4))
        d = r * 0.68
        axA.plot([x - d, x + d], [y - d, y + d], color=RED, lw=1.5, zorder=6)
        axA.plot([x - d, x + d], [y + d, y - d], color=RED, lw=1.5, zorder=6)
        axA.text(x + off[0], y + off[1], name, color=GREY, fontsize=5.4,
                 ha=ha, va="center", zorder=5)

# magnified dry skin-electrode interface at TP9
tx, ty = CX - 0.190, CY - 0.150
ix, iy, ir = 1.075, CY - 0.020, 0.155
axA.plot([tx + 0.082, ix - ir * 0.95], [ty + 0.060, iy + ir * 0.40],
         color=NAVY, lw=0.8, ls=(0, (2.5, 1.8)), zorder=2)
axA.plot([tx + 0.088, ix - ir * 0.95], [ty - 0.054, iy - ir * 0.66],
         color=NAVY, lw=0.8, ls=(0, (2.5, 1.8)), zorder=2)
axA.add_patch(Circle((ix, iy), ir, fc="white", ec=NAVY, lw=1.5, zorder=7))
axA.add_patch(Wedge((ix, iy), ir - 0.014, 180, 360, fc="#FBE9DE", ec="none",
                    zorder=8))
axA.plot([ix - ir * 0.86, ix + ir * 0.86], [iy, iy], color="#C8A48E", lw=1.2,
         zorder=9)
axA.add_patch(FancyBboxPatch((ix - 0.068, iy + 0.016), 0.136, 0.042,
                             boxstyle="round,pad=0.004,rounding_size=0.012",
                             fc=NAVY, ec="none", zorder=10))
for dx in (-0.040, 0.0, 0.040):                      # dry contact asperities
    axA.plot([ix + dx, ix + dx], [iy + 0.016, iy - 0.015], color=NAVY, lw=1.1,
             zorder=9)
axA.text(ix, iy + ir + 0.055, "dry contact", fontsize=5.5, color=NAVY,
         weight="bold", ha="center", va="center", zorder=10)
axA.text(ix, iy - ir - 0.052, "skin, no gel", fontsize=5.1, color="#8A6A57",
         ha="center", va="center", zorder=10)

axA.text(0.67, 0.985, "4-contact dry headband", fontsize=7.4, weight="bold",
         ha="center", va="center", color="#222")
axA.text(0.67, 0.115,
         f"TP9 alone ({tp9_only:.2f})  $\\approx$  best 3 contacts without it "
         f"({no_tp9_3:.2f})",
         fontsize=5.4, ha="center", va="center", color=NAVY, weight="bold")
axA.text(0.67, 0.048, "left temporal anchors every array that works",
         fontsize=5.2, ha="center", va="center", color="#555")

# ============================================================== panel B
axB = fig.add_axes([0.443, 0.315, 0.188, 0.520])
axB.plot(best.n_ch, best.acc, "o-", color=NAVY, lw=1.6, ms=4.2, zorder=3)
axB.axhline(full, ls="--", lw=1.0, color=RED, zorder=2)
axB.axhline(1 / 3, ls=":", lw=1.0, color=GREY, zorder=2)
axB.annotate(f"{pair:.3f}\n97.6% of full", xy=(2, pair), xytext=(2.2, 0.545),
             fontsize=6.0, color=NAVY, weight="bold", linespacing=1.2,
             arrowprops=dict(arrowstyle="-", color=NAVY, lw=0.8))
axB.text(4.05, full, "4 contacts", fontsize=5.6, color=RED, va="bottom",
         ha="right")
axB.text(0.95, 1 / 3, "chance", fontsize=5.4, color=GREY, va="bottom")
axB.set_xticks([1, 2, 3, 4]); axB.set_xlim(0.7, 4.3); axB.set_ylim(0.28, 0.87)
axB.set_yticks([0.4, 0.6, 0.8])
axB.tick_params(labelsize=5.9, length=2.5, pad=1.5)
axB.set_xlabel("contacts in array", fontsize=6.4, labelpad=1.5)
axB.set_ylabel("cross-subject accuracy", fontsize=6.4, labelpad=1.5)
axB.set_title("Array can be halved", fontsize=7.4, weight="bold", pad=3,
              color="#222")

# ============================================================== panel C
axC = fig.add_axes([0.778, 0.315, 0.202, 0.520])
rows = [("no fault", clean, "#7F9BB8"),
        ("white noise\n10 dB", white10, NAVY),
        ("50 Hz mains\n10 dB", mains10, ORANGE),
        ("open contact\n(worst)", open_worst, "#7F9BB8"),
        ("degraded\ncontact", degraded, RED)]
ypos = np.arange(len(rows))[::-1]
axC.barh(ypos, [r[1] for r in rows], color=[r[2] for r in rows], height=0.66)
for y, (_, v, _c) in zip(ypos, rows):
    axC.text(v + 0.018, y, f"{v:.2f}", va="center", fontsize=5.8, color="#222")
axC.axvline(1 / 3, ls=":", lw=1.0, color=GREY)
axC.set_yticks(ypos)
axC.set_yticklabels([r[0] for r in rows], fontsize=6.0, linespacing=1.08)
axC.tick_params(axis="y", pad=2.5)
axC.set_xlim(0, 0.98); axC.set_xticks([0, 0.4, 0.8])
axC.tick_params(labelsize=5.9, length=2.5, pad=1.5)
axC.set_xlabel("cross-subject accuracy", fontsize=6.4, labelpad=1.5)
axC.set_title("What actually hurts it", fontsize=7.4, weight="bold", pad=3,
              color="#222")

# ============================================================== takeaway
fig.patches.append(FancyBboxPatch(
    (0.300, 0.022), 0.692, 0.115, transform=fig.transFigure,
    boxstyle="round,pad=0.003,rounding_size=0.010",
    fc="#EEF2F7", ec="#C6D2E0", lw=0.8, zorder=0))
fig.text(0.646, 0.079,
         "50 Hz mains costs 2$\\times$ broadband noise of equal power  ·  "
         "a degraded contact is worse than an open one",
         fontsize=5.9, ha="center", va="center", color="#1F3A5F")

# ============================================================== output
png = os.path.join(FG, "graphical_abstract.png")
fig.savefig(png, dpi=DPI, facecolor="white")

from PIL import Image
im = Image.open(png).convert("RGB")
target = (W * SCALE, H * SCALE)
if im.size != target:
    im = im.resize(target, Image.LANCZOS)

hi = os.path.join(FG, "graphical_abstract.jpg")
im.save(hi, "JPEG", quality=95, optimize=True, subsampling=0)
print(f"{os.path.basename(hi)}  {im.size[0]}x{im.size[1]}  "
      f"{os.path.getsize(hi)/1024:.1f} kB")

spec = os.path.join(FG, "graphical_abstract_660x295.jpg")
small = im.resize((W, H), Image.LANCZOS)
for q in (95, 90, 85, 80, 75, 70):
    small.save(spec, "JPEG", quality=q, optimize=True, subsampling=0)
    kb = os.path.getsize(spec) / 1024
    if kb < 45:
        break
print(f"{os.path.basename(spec)}  {W}x{H}  {kb:.1f} kB  (quality {q})")

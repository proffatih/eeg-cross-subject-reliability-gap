import matplotlib as mpl
from cycler import cycler
PALETTE=["#2E5A87","#C44E52","#55A868","#CC8963","#8172B3","#64B5CD","#8C8C8C"]
def apply():
    mpl.rcParams.update({
        "figure.dpi":150,"savefig.dpi":300,"savefig.bbox":"tight",
        "font.family":"serif","font.serif":["DejaVu Serif"],"mathtext.fontset":"dejavuserif",
        "font.size":10,"axes.titlesize":11,"axes.labelsize":10,"legend.fontsize":8.5,
        "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,
        "grid.alpha":0.25,"grid.linewidth":0.6,"axes.axisbelow":True,
        "axes.edgecolor":"#444444","axes.linewidth":0.9,"xtick.color":"#222","ytick.color":"#222",
        "legend.frameon":False,"figure.facecolor":"white",
        "axes.prop_cycle":cycler(color=PALETTE),
        "image.cmap":"viridis","lines.linewidth":1.8,"lines.markersize":5})

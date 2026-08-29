import numpy as np, csv, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from pro_style import apply as _pa; _pa()
except Exception as _e:
    pass
R="results"; F="figures"; os.makedirs(F,exist_ok=True)
z=np.load(f"{R}/fig_arrays.npz",allow_pickle=True)
M={r["model"]+"_"+r["protocol"]:r for r in csv.DictReader(open(f"{R}/metrics.csv"))}
CLS=["Relaxed","Neutral","Concentr."]

# Fig1: within vs LOSO accuracy + ECE (GBM, MLP)
fig,ax=plt.subplots(1,2,figsize=(8.4,3.4)); x=np.arange(2); w=0.35
for i,mdl in enumerate(["GBM","MLP"]):
    accs=[float(M[mdl+"_within"]["acc_mean"]),float(M[mdl+"_loso"]["acc_mean"])]
    errs=[float(M[mdl+"_within"]["acc_std"]),float(M[mdl+"_loso"]["acc_std"])]
    ax[0].bar(x+(i-0.5)*w,accs,w,yerr=errs,capsize=4,label=mdl)
    eces=[float(M[mdl+"_within"]["ece_mean"]),float(M[mdl+"_loso"]["ece_mean"])]
    ax[1].bar(x+(i-0.5)*w,eces,w,label=mdl)
for a,t,yl in zip(ax,["(a) Accuracy","(b) Calibration error (ECE)"],["Accuracy","ECE"]):
    a.set_xticks(x); a.set_xticklabels(["Within-subject","Cross-subject (LOSO)"]); a.set_ylabel(yl); a.set_title(t); a.legend(fontsize=9)
ax[0].set_ylim(0,1); plt.tight_layout()
plt.savefig(f"{F}/fig1_within_vs_loso.pdf"); plt.savefig(f"{F}/fig1_within_vs_loso.png",dpi=220); plt.close()

# Fig2: GBM confusion within vs LOSO
def cm(P,Y,k=3):
    p=P.argmax(1); C=np.zeros((k,k))
    for t,q in zip(Y,p): C[t,q]+=1
    return C/C.sum(1,keepdims=True)
fig,ax=plt.subplots(1,2,figsize=(8.2,3.7))
for a,(prot,ttl) in zip(ax,[("within","Within-subject"),("loso","Cross-subject (LOSO)")]):
    C=cm(z[f"GBM_{prot}_P"],z[f"GBM_{prot}_Y"])
    im=a.imshow(C,cmap="Blues",vmin=0,vmax=1)
    for ii in range(3):
        for jj in range(3): a.text(jj,ii,f"{C[ii,jj]:.2f}",ha="center",va="center",fontsize=9,color="white" if C[ii,jj]>.5 else "black")
    a.set_xticks(range(3)); a.set_xticklabels(CLS,fontsize=8); a.set_yticks(range(3)); a.set_yticklabels(CLS,fontsize=8)
    a.set_xlabel("Predicted"); a.set_ylabel("True"); a.set_title(f"GBM {ttl}")
plt.tight_layout(); plt.savefig(f"{F}/fig2_confusion_gbm.pdf"); plt.savefig(f"{F}/fig2_confusion_gbm.png",dpi=220); plt.close()

# Fig3: per-subject LOSO accuracy (GBM)
rows=[r for r in csv.DictReader(open(f"{R}/per_subject_loso.csv")) if r["model"]=="GBM"]
subs=[r["subject"] for r in rows]; accs=[float(r["loso_acc_mean"]) for r in rows]
plt.figure(figsize=(6.4,3.4))
plt.bar(subs,accs,color="#4472C4",edgecolor="k")
plt.axhline(float(M["GBM_loso"]["acc_mean"]),ls="--",c="r",label=f"LOSO mean={float(M['GBM_loso']['acc_mean']):.2f}")
plt.xlabel("Held-out subject"); plt.ylabel("LOSO accuracy"); plt.ylim(0,1); plt.title("Per-subject cross-subject accuracy (GBM)"); plt.legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{F}/fig3_per_subject.pdf"); plt.savefig(f"{F}/fig3_per_subject.png",dpi=220); plt.close()
print("3 figür yazıldı; LOSO subjects:",len(subs))

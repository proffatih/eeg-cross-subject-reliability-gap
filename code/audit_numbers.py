"""Check every decimal the compiled manuscript prints against the result files.

The audit runs on the PDF, not the .tex, because the failure this guards against
is a stale table or figure that the source no longer explains. Any 0.xxx in the
text that no committed result reproduces is reported; the only values expected
to be unmatched are the accuracies of the cited papers, which come from those
papers and not from our pipeline.

  python audit_numbers.py
"""
import os, re, json, glob, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
PDF = os.path.join(HERE, "..", "manuscript", "manuscript_R1.pdf")

# values that legitimately come from outside our pipeline
EXTERNAL = {"0.844", "0.624", "0.938", "0.970", "0.855", "0.843", "0.863",
            "0.819", "0.897", "0.336", "0.333", "0.500", "0.167"}


def derived():
    """Values the manuscript computes from the result files rather than reading
    off them. Each is recomputed here, so a stale one still fails the audit."""
    import pandas as pd
    out = set()
    d = pd.read_csv(os.path.join(R, "session_split_vs_accuracy.csv"))
    for m in ("GBM", "MLP"):                      # mean over the split participants
        g = d[(d.model == m) & (d.separate_session)]
        out.add(f"{g.loso_acc.mean():.3f}")
    return out


def harvested():
    """Every number any result file contains, to three decimals."""
    out = set()

    def add(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return
        out.add(f"{abs(f):.3f}")
        out.add(f"{abs(f):.2f}")

    for fp in glob.glob(os.path.join(R, "*.csv")):
        for line in open(fp, encoding="utf-8"):
            for tok in re.split(r"[,\s]+", line):
                add(tok)
    for fp in glob.glob(os.path.join(R, "*.json")):
        def walk(o):
            if isinstance(o, dict):
                [walk(v) for v in o.values()]
            elif isinstance(o, list):
                [walk(v) for v in o]
            else:
                add(o)
        walk(json.load(open(fp, encoding="utf-8")))
    return out


def main():
    txt = subprocess.run(["pdftotext", PDF, "-"], capture_output=True,
                         text=True, check=True).stdout
    have = harvested() | derived()
    printed = re.findall(r"\b0\.\d{3}\b", txt)
    missing = [v for v in sorted(set(printed))
               if v not in have and v not in EXTERNAL]
    print(f"  decimals printed : {len(printed)} ({len(set(printed))} distinct)")
    print(f"  unmatched        : {len(missing)}")
    for v in missing:
        ctx = [l.strip() for l in txt.splitlines() if v in l][:1]
        print(f"    {v}   {ctx[0][:90] if ctx else ''}")
    return len(missing)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)

"""Replace the hand-maintained table bodies in the manuscript with the
generated ones, and refuse to proceed if they were already out of step.

The first revision shipped a table that still printed a withdrawn number
because the body was pasted by hand and never refreshed. This closes that path:
make_tables.py writes the bodies, this splices them in, and --check reports a
non-zero status when the manuscript disagrees with the results.

  python splice_tables.py [--check]
"""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(HERE, "..", "manuscript")
TEX = os.path.join(M, "manuscript_R1.tex")
# label -> generated body; each body ends at \bottomrule
BODIES = {"tab:array": "tab_array.tex", "tab:budget": "tab_budget.tex"}


def _bounds(s, label):
    """The body of the tabular inside the table environment carrying `label`,
    from the row after the header \\midrule to \\bottomrule inclusive."""
    i = s.index("\\label{%s}" % label)
    j = s.index("\\end{table}", i)
    block = s[i:j]
    mid = block.index("\\midrule") + len("\\midrule")
    bot = block.index("\\bottomrule") + len("\\bottomrule")
    return i + mid, i + bot


def main(check=False):
    s = open(TEX, encoding="utf-8").read()
    bad = 0
    for label, fn in BODIES.items():
        gen = open(os.path.join(M, fn), encoding="utf-8").read().strip()
        a, b = _bounds(s, label)
        cur = s[a:b].strip()
        if _norm(cur) == _norm(gen):
            print(f"  {label:12s} up to date")
            continue
        bad += 1
        if check:
            print(f"  {label:12s} DIFFERS from {fn}")
        else:
            s = s[:a] + "\n" + gen + s[b:]
            print(f"  {label:12s} spliced from {fn}")
    if not check and bad:
        open(TEX, "w", encoding="utf-8").write(s)
    return bad


def _norm(t):
    return re.sub(r"\s+", " ", t).strip()


if __name__ == "__main__":
    n = main("--check" in sys.argv)
    sys.exit(1 if ("--check" in sys.argv and n) else 0)

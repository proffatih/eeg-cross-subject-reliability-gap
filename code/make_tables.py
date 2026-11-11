"""Emit the two sensor tables of the manuscript from the committed result files.

The rows of Table 3 (electrode subsets) and Table 4 (compute budget) used to be
maintained by hand, which let them drift from the results they report. They are
generated here instead, so that regenerating the results regenerates the tables.

  python make_tables.py          -> manuscript/tab_array.tex, manuscript/tab_budget.tex
"""
import os, json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
M = os.path.join(HERE, "..", "manuscript")
os.makedirs(M, exist_ok=True)


def tab_array():
    d = pd.read_csv(os.path.join(R, "sensor_electrode_subsets.csv"))
    rows = []
    for k in [4, 3, 2, 1]:
        g = d[d.n_ch == k].sort_values("GBM_acc", ascending=False)
        for _, r in g.iterrows():
            b = lambda v, best: (r"\textbf{%.3f}" % v) if (best and len(g) > 1) else "%.3f" % v
            rows.append(f"{r.electrodes} & {b(r.GBM_acc, r.GBM_acc == g.GBM_acc.max())} & "
                        f"{r.GBM_f1:.3f} & {r.GBM_ece:.3f} & "
                        f"{b(r.MLP_acc, r.MLP_acc == g.MLP_acc.max())} \\\\")
        rows.append(r"\midrule" if k > 1 else r"\bottomrule")
    open(os.path.join(M, "tab_array.tex"), "w").write("\n".join(rows) + "\n")
    print(f"tab_array.tex: {len(d)} subsets, MLP s.d. <= {d.MLP_acc_sd.max():.3f}")


def tab_budget():
    j = json.load(open(os.path.join(R, "sensor_compute_budget.json")))
    fe, rows = j["feature_extraction_ms"], []
    for cfg in ["full", "compact"]:
        for tag in ["4-electrode", "2-electrode"]:
            r = j["decoders"][f"{cfg} / {tag}"]
            rows.append(f"{cfg.capitalize()}, {tag[0]} contacts & {r['model_kb']:.0f} & "
                        f"{1000 * r['inference_ms_per_window_batch']:.0f} & "
                        f"{1000 * r['inference_ms_per_window_single']:.0f} & "
                        f"{r['loso_acc']:.3f} \\\\")
    rows += [r"\midrule",
             r"\multicolumn{5}{l}{Feature extraction: %.2f ms (4 contacts), %.2f ms (2 contacts)} \\"
             % (fe["4-electrode"], fe["2-electrode"]), r"\bottomrule"]
    open(os.path.join(M, "tab_budget.tex"), "w").write("\n".join(rows) + "\n")
    b = j["decoders"]["full / 4-electrode"]
    print(f"tab_budget.tex: batch {1000*b['inference_ms_per_window_batch']:.0f} us, "
          f"single {1000*b['inference_ms_per_window_single']:.0f} us, "
          f"feature {fe['4-electrode']:.2f} ms, batch size {b['batch_size']}")


if __name__ == "__main__":
    tab_array(); tab_budget()

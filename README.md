# The Cross-Subject Reliability Gap in EEG Affective-State Sensing

Reproducible code and results for:

> F. Gül, "The Cross-Subject Reliability Gap in EEG Affective-State Sensing:
> Accuracy and Calibration Under Leave-One-Subject-Out Evaluation,"
> submitted to *IEEE Sensors Letters*, 2026.

Artificial Intelligence and Internet of Things Research Laboratory,
Department of Electrical and Electronics Engineering / Department of Biomedical Engineering,
Recep Tayyip Erdoğan University, Rize, Türkiye.

## What this is

Within-subject cross-validation is the default reporting protocol in wearable
affective-EEG studies, and it overstates how a model behaves on a **new** user.
This repository reproduces, end to end, the accuracy **and calibration** cost of
moving from within-subject evaluation to leave-one-subject-out (LOSO) evaluation
on an open consumer-grade (4-channel Muse headband) mental-state EEG dataset.

Headline numbers reproduced by `code/experiment.py`:

| Model | Protocol | Accuracy | Macro-F1 | ECE |
|---|---|---|---|---|
| GBM | within-subject | 0.903 | 0.906 | 0.082 |
| GBM | LOSO | 0.787 | 0.785 | 0.190 |

An 11.5-point accuracy gap, and a calibration error that worsens 2.3×.
Per-subject LOSO accuracy ranges from 0.39 to 0.97.

## Data

The dataset is public and is **not** redistributed here:

> J. J. Bird, L. J. Manso, E. P. Ribeiro, A. Ekárt, D. R. Faria,
> "A study on mental state classification using EEG-based brain–machine interface,"
> *Proc. 9th Int. Conf. on Intelligent Systems (IS)*, 2018, pp. 795–800.
> doi:[10.1109/IS.2018.8710576](https://doi.org/10.1109/IS.2018.8710576)

Four subjects × three states (relaxed / neutral / concentrating) × two sessions.
Place the raw per-recording CSVs (`subject<a-d>-<state>-<session>.csv`) in `data/raw/`.

## Reproduce

```bash
pip install numpy pandas scipy scikit-learn matplotlib
python code/features.py      # data/raw/*.csv  ->  data/features.npz  (48-D feature vector)
python code/experiment.py    # -> results/*.csv, results/*.json
python code/make_figures.py  # -> figures/*.png
```

`results/` and `figures/` in this repository are the committed outputs of exactly
these three commands, so a fresh run can be diffed against them.

## Layout

- `code/features.py` — band-power / statistical feature extraction (48 features, 1 s windows)
- `code/experiment.py` — within-subject grouped CV and LOSO for GBM and MLP; accuracy, macro-F1, ECE (15 bins), averaged over seeds
- `code/make_figures.py` — manuscript figures
- `results/metrics.csv` — per model × protocol metrics
- `results/generalization_gap.csv` — within-subject → LOSO deltas
- `results/per_subject_loso.csv` — per-held-out-subject LOSO accuracy
- `results/confusion_gbm.json` — pooled LOSO confusion matrix

## Declaration on generative AI

Generative AI tools were used for language editing and code-scaffolding assistance.
All experiments, numbers, figures and conclusions were produced by the code in this
repository and verified by the author.

## License

MIT (code). The underlying dataset remains under its original terms.

# Two Dry Electrodes Suffice — sensor characterisation of a wearable affective-EEG headband

Reproducible code and results for:

> F. Gül, "Two Dry Electrodes Suffice: Array Reduction, Interference Resilience
> and Cross-Subject Calibration of a Wearable Affective-EEG Sensor,"
> submitted to *IEEE Sensors Letters*, 2026.

Artificial Intelligence and Internet of Things Research Laboratory,
Department of Electrical and Electronics Engineering / Department of Biomedical Engineering,
Recep Tayyip Erdoğan University, Rize, Türkiye.

## What this is

A four-channel dry-electrode consumer EEG headband, characterised against the
performance that actually matters for a wearable: performance on a user the
device has never seen (leave-one-subject-out), scored with accuracy **and**
probability calibration. The decoder is held fixed; every result is a statement
about the sensor.

| Question | Answer measured here |
|---|---|
| How many electrodes? | The pair TP9+AF8 gives 0.768 vs 0.787 for all four — 97.6%. The symmetric frontal pair AF7+AF8 gives only 0.542: placement, not count, sets the limit. |
| Where should the analogue budget go? | 50 Hz mains pickup costs ~2× the accuracy of broadband noise at equal power (0.513 vs 0.637 at 10 dB SNR). Baseline wander is already rejected by the 1 Hz high-pass. |
| What about a failing electrode? | A silently degraded contact (0.515) is worse than an open one (0.654–0.743). Detect and exclude beats tolerate. |
| Can the ADC rate be lowered? | No: 256→128 Hz costs 19 points. It is the waveform-shape features that collapse (0.780→0.565), not band power (0.631→0.623). |
| Can the decision window be shortened? | Yes, 2 s → 0.5 s is free (0.797 → 0.801). |
| What dominates the compute budget? | Feature extraction (3.63 ms for 4 contacts, 1.76 ms for 2) over inference (0.015 ms) by >200×. |

Baseline, for reference: within-subject cross-validation would report 0.903
instead of 0.787, with the calibration error doubling (ECE 0.082 → 0.190) on an
unseen wearer.

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
python code/features.py             # data/raw/*.csv -> data/features.npz (48-D)
python code/experiment.py           # within-subject vs LOSO baseline
python code/sensor_experiments.py   # S1-S4 sensor characterisation (~25 min)
python code/make_figures.py
python code/make_sensor_figures.py
```

`results/` and `figures/` in this repository are the committed outputs of exactly
these three commands, so a fresh run can be diffed against them.

## Layout

- `code/features.py` — band-power and waveform-shape feature extraction (12 features per contact, 1 s windows)
- `code/experiment.py` — within-subject grouped CV and LOSO baseline; accuracy, macro-F1, ECE (15 bins)
- `code/sensor_experiments.py` — the sensor characterisation: electrode-array ablation over all 15 subsets (S1), injected noise / mains / wander / contact-degradation / dropout (S2), sampling rate and window length (S3), feature-group diagnosis of the rate sensitivity (S3b), compute and footprint budget (S4)
- `code/make_figures.py`, `code/make_sensor_figures.py` — figures
- `results/sensor_electrode_ablation.csv` — every array subset
- `results/sensor_noise_resilience.csv` — degradations, 3 noise realisations each
- `results/sensor_rate_window.csv`, `results/sensor_feature_group_vs_rate.csv` — rate/latency budget and its cause
- `results/sensor_edge_cost.json` — footprint and per-window latency
- `results/metrics.csv`, `results/per_subject_loso.csv` — within-subject vs LOSO baseline

## Declaration on generative AI

Generative AI tools were used for language editing and code-scaffolding assistance.
All experiments, numbers, figures and conclusions were produced by the code in this
repository and verified by the author.

## License

MIT (code). The underlying dataset remains under its original terms.

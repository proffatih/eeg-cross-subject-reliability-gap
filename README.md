# Two dry electrodes suffice: analysis code

Cross-subject (leave-one-subject-out) characterisation of a four-channel
dry-electrode Muse headband (TP9, AF7, AF8, TP10) on the open recordings of
Bird et al. (2018): 4 subjects x 3 mental states (relaxed, neutral,
concentrating) x 2 sessions = 24 recordings at 256 Hz.

## Data

Copy the 24 files `subject[a-d]-[state]-[1-2].csv` from
`dataset/original_data/` of https://github.com/jordan-bird/eeg-feature-generation
into `data/raw/`.

## Setup

Python 3.12, then `pip install -r code/requirements.txt`.

## Run order

All scripts are run from `code/`; results are written to `results/`.

| Script | What it does | Outputs | Paper | Time* |
|---|---|---|---|---|
| `verify_sampling_rate.py` | Sampling rate from timestamps; lists recordings with timestamp gaps | screen | Sec. II | s |
| `features.py` | Splits recordings at gaps, 50 Hz notch + 1-45 Hz band-pass, 1 s windows (50% overlap), 10 features per contact | `data/features.npz` | Sec. II | 15 s |
| `experiment.py` | GBM and MLP, within-subject (leave-one-session-out) and LOSO | `metrics.csv`, `per_group.csv`, `confusion_gbm.json` | Sec. II | 1 min |
| `confound_checks.py` | Notch vs no notch; 50 Hz-only, DC-only and broadband controls; line prominence; extra notches; causal filters; 1-20 Hz | `confound_*.csv` | Mains section | 4 min |
| `sensor_experiments.py s1` | All 15 electrode subsets, GBM and MLP | `sensor_electrode_subsets.csv`, `..._per_subject.csv` | Sec. III | 4 min |
| `sensor_experiments.py s2` | Noise and contact faults (absolute noise level; trained clean, tested degraded) | `sensor_noise_faults.csv`, `noise_reference.json` | Sec. IV | 11 min |
| `sensor_experiments.py s3` | Converter rate 256-112 Hz | `sensor_rate.csv` | Sec. V | 2 min |
| `sensor_experiments.py s3w` | Decision window 2, 1, 0.5 s at 256 and 112 Hz | `sensor_window.csv` | Sec. V | 3 min |
| `sensor_experiments.py s4` | Feature-extraction and inference time, model size | `sensor_compute_budget.json` | Sec. V | 1 min |
| `make_figures.py` | Figures 1-3, Tables 1-2 and the graphical abstract | `figures/` | all | 30 s |
| `sensor_experiments.py s3c` | Optional: band-controlled rate test (identical content at every rate) | `sensor_rate_band_controlled.csv` | reply to reviewers | 2 min |

`features.py` must be run first. `python sensor_experiments.py` without an
argument runs s1, s2, s3, s3w and s4. *Single CPU core.

## Model settings

Both decoders are fitted on features z-scored with the statistics of the
training fold (`StandardScaler`).

| Decoder | Settings (scikit-learn 1.8.0) |
|---|---|
| GBM (reference) | `HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08)`; all other parameters at their defaults. At this data size it has no random element and is run once. |
| MLP | `MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-3, batch_size=128, learning_rate_init=1e-3, max_iter=400, early_stopping=True, n_iter_no_change=20)`; defaults otherwise (Adam, stratified 10% validation split, best weights restored). Averaged over seeds 0-4. |
| Compact GBM (Table II) | `HistGradientBoostingClassifier(max_iter=60, learning_rate=0.2, max_depth=3, max_leaf_nodes=8)` |

## Reproducibility

Every random element uses a fixed seed. The GBM has no random element at this
data size (no early stopping, no feature or bin subsampling) and is run once;
the MLP is averaged over seeds 0-4; injected degradations use noise seeds 0-2.
With the versions in `requirements.txt` every number is reproduced exactly,
except timings, which vary between runs and machines and are meaningful only
as ratios between array sizes.

## Changes from the original pipeline

- 50 Hz notch before the band-pass. The recordings contain state-dependent
  mains pickup; without the notch it reaches the features through the gentle
  order-4 band edge and inflates cross-subject accuracy from 0.58 to 0.78.
- Recordings are split at timestamp gaps longer than 0.1 s
  (`subjectb-relaxed-2` consists of ten 3-4.5 s pieces); 2442 windows.
- The window mean (always zero after centring) and RMS (equal to the standard
  deviation) were removed: 10 features per contact, 40 in total.
- Within-subject evaluation is leave-one-session-out (8 fixed folds) instead of
  a random grouped K-fold, whose result depended on the library version.
- Noise is injected at one absolute level for every recording and the decoder is
  trained on clean data; contact dropout is applied to a random contact, like
  contact degradation, so the two faults are compared on the same footing.
- The converter-rate test covers 256-112 Hz, all of which carry the full
  analysis band, using rational resampling.
- `np.trapz` replaced by `np.trapezoid`; library versions pinned.

# A recording-session confound accounts for cross-subject accuracy on a four-contact dry-electrode EEG benchmark

Analysis code for the manuscript of the same name.

Cross-subject (leave-one-subject-out) characterisation of a four-channel
dry-electrode Muse headband (TP9, AF7, AF8, TP10) on the open recordings of
Bird et al. (2018): 4 subjects x 3 mental states (relaxed, neutral,
concentrating) x 2 trials = 24 recordings at 256 Hz.

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
| `verify_sampling_rate.py` | Sampling rate from timestamps; lists recordings with timestamp gaps | screen | Methods | s |
| `features.py` | Splits recordings at gaps, 50 Hz notch + 1-45 Hz band-pass, 1 s windows (50% overlap), 10 features per contact | `data/features.npz` | Methods | 15 s |
| `experiment.py` | GBM and MLP, within-subject (leave-one-session-out) and LOSO | `metrics.csv`, `per_group.csv`, `confusion_gbm.json` | Methods | 1 min |
| `confound_checks.py` | Notch vs no notch; 50 Hz-only, DC-only and broadband controls; line prominence; extra notches; causal filters; 1-20 Hz | `confound_*.csv` | Results | 4 min |
| `sensor_experiments.py s1` | All 15 electrode subsets, GBM and MLP | `sensor_electrode_subsets.csv`, `..._per_subject.csv` | Results | 4 min |
| `sensor_experiments.py s2` | Noise and contact faults (absolute noise level; trained clean, tested degraded) | `sensor_noise_faults.csv`, `noise_reference.json` | Results | 11 min |
| `sensor_experiments.py s3` | Converter rate 256-112 Hz | `sensor_rate.csv` | Results | 2 min |
| `sensor_experiments.py s3w` | Decision window 2, 1, 0.5 s at 256 and 112 Hz | `sensor_window.csv` | Results | 3 min |
| `sensor_experiments.py s4` | Feature-extraction and inference time, model size | `sensor_compute_budget.json` | Results | 1 min |
| `session_confound.py` | Recording times; day split; split vs accuracy; relaxed-vs-neutral; clock-only baseline | `session_*.csv`, `time_only_baseline.json`, `within_session_*.csv` | Results, Fig. 1 | 2 min |
| `reviewer_followups.py` | 50 Hz power vs time; artifact-only decoder; prediction agreement; per-subject artifact baselines | `mains_vs_time*.{csv,json}`, `artifact_model_*.{csv,json}`, `artifact_baselines_per_subject.csv` | Results | 3 min |
| `make_session_figure.py` | Figure 1 (session confound) | `manuscript/Figure1_session.pdf` | Fig. 1 | 10 s |
| `make_tables.py` | Tables 3 and 4 straight from the result files | `manuscript/tab_*.tex` | Tables 3-4 | s |
| `make_figures.py` | Figures 2-4, Tables 1-2 and the graphical abstract | `figures/` | all | 30 s |
| `followups_round2.py` | Time-definition sensitivity; recording-level permutation tests; clock-only and artifact-only baselines for relaxed vs neutral; mains with time as a covariate | `mains_time_definitions.json`, `within_session_permutation.csv`, `within_session_baselines.csv`, `mains_covariate_model.json` | Results | 6 min |
| `splice_tables.py --check` | Fails if the manuscript's table bodies disagree with the result files | screen | guard | s |
| `audit_numbers.py` | Compares every decimal in the compiled PDF against the committed results | screen | guard | s |
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

- 50 Hz notch before the band-pass. The recordings contain session-correlated
  mains pickup; without the notch it reaches the features through the gentle
  order-4 band edge and inflates cross-subject accuracy from 0.57 to 0.78.
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

## Revision, November 2026

Two reviewers at *Scientific Reports* prompted a correction that changed the
paper's conclusion, and this repository now reflects it.

**The labels in this dataset are confounded with the recording session.** The
`timestamps` column shows that in three of the four participants the
concentrating recordings were made on a different day from the relaxed and
neutral recordings, about 46 h later. A classifier that sees no EEG at all, only
where a window sits in time, reaches 0.569 cross-subject accuracy against 0.333
chance; the full decoder on notch-filtered data reaches 0.574. A 50 Hz notch
removes one symptom, not the confound.

**No contrast in this dataset survives the controls.** Relaxed versus neutral,
recorded on the same day for every participant, is the one contrast free of the
day split and is decoded at 0.606 and 0.659 against 0.500. A second round of
checks showed that this does not make it neural either: a decoder given only the
recording clock reaches 0.592 on that contrast and an artifact-only decoder
0.850. Permuting labels over each participant's four recordings — the
exchangeable unit, since windows overlap and are serially correlated — leaves
every participant short of significance (p = 0.167 for a and b, 0.500 for c and
d; with four recordings the smallest attainable p is 0.167). The earlier claim
that these recordings carry demonstrable cross-subject neural information has
been withdrawn.

Run `python code/session_confound.py all` to reproduce this.

**Three further analyses were run at the reviewers' request**
(`python code/reviewer_followups.py all`).

- *Artifact-only decoder.* The 50 Hz power, the DC offset and the 55-128 Hz
  power of each contact — twelve numbers, none of which survives the 1-45 Hz
  band-pass — reach **0.761** under the same leave-one-subject-out folds, i.e.
  nineteen points above the full in-band decoder. Per participant: 0.839, 0.993,
  0.372, 0.838; the three with the day split are decoded almost perfectly and
  the one without it at chance.
- *Prediction comparison.* The in-band decoder and the artifact-only decoder
  give the same class on 0.600 of the windows (kappa 0.36) against the 0.336
  expected of two independent decoders at these class frequencies.
- *50 Hz power against time.* Within the first day the line level rises with
  the time at which a recording was made (Spearman rho 0.56, p 0.015, n 18);
  at the day boundary it steps by +25.1 dB; in the participant without a split
  the concentrating recordings sit only 5.8 dB above the rest.

**Two defects in the earlier code were found by a reviewer and are fixed here.**

- Band power was computed by trapezoidal integration over the bins of a band.
  For a band holding a single bin this returns exactly zero, which silently
  deleted the delta feature whenever the grid spacing reached 2 Hz (any 0.5 s
  window). Band power is now the sum of the in-band bins times the bin width,
  and the Welch segment is one second of data wherever the window allows, so the
  frequency grid no longer changes with the sampling rate.
- Inference latency was timed in batch and divided by the batch size, then
  compared against per-window feature extraction. Both are now measured the same
  way: inference costs 0.032 ms per window in batch (n = 688) but 4.40 ms one
  window at a time, against 3.04 ms for feature extraction. The earlier
  conclusion that feature extraction sets the compute budget does not hold for a
  node that classifies as the data arrive.

All results in `results/` were regenerated with the corrected pipeline. The
headline three-class figure moved from 0.577 to 0.574.

**The electrode, interference, converter-rate and window comparisons are kept
for completeness but are scored against the confounded label.** They describe
this dataset; they are not hardware guidance.

### Second round of checks, 16 November 2026

An internal audit by a co-author who had not written the package asked for three
things the first revision had not done. All three are in `followups_round2.py`.

**1. The 50 Hz-versus-time correlation depends on how "time" is defined.** The
first revision reported one definition. Under three:

| definition of time | rho | p |
|---|---|---|
| rank within each participant | **+0.584** | 0.011 |
| hours since that participant's own first recording | +0.562 | 0.015 |
| absolute wall-clock time | **-0.296** | 0.233 |

The sign is not stable, because the participants were recorded in different
parts of the afternoon. The manuscript now leads with the within-participant
form and states all three.

**2. Time as a covariate.** Mains level regressed on state with acquisition time
and participant as covariates. Over all 24 recordings the coefficient on time is
+0.042 per hour (p = 0.003), while concentrating and relaxed differ from neutral
by +0.46 (p = 0.37) and -0.26 (p = 0.39). State does not survive the model.

**3. Recording-level permutation tests**, reported above, which withdrew the
last positive claim.

Two guards were added at the same time, after the first revision was found to
contain a table body that still printed a withdrawn number:
`splice_tables.py --check` fails when the manuscript's tables disagree with the
result files, and `audit_numbers.py` compares every decimal in the compiled PDF
against the committed results (currently 217 decimals, 0 unmatched).

One data note that matters for anyone reusing these files:
`subjectb-relaxed-2.csv` spans 958 s but contains **41.5 s** of data, in ten
fragments separated by nine gaps totalling 916 s. Its windows do not sit at a
single point in time, and a duration taken from first and last timestamp is
wrong for this recording.

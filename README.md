# Cumulative Divergence (CD): reproducibility package

Repository: https://github.com/erbiber/CD  
Pre-registration: https://doi.org/10.17605/OSF.IO/HEAXQ

Code, figures, and result tables for the cumulative-divergence (CD) estimator and its
validation. CD measures cumulative within-window distributional change in a single-trial
signal: a window is split into k+1 consecutive sub-windows, the per-step Gaussian
Kullback–Leibler divergence between consecutive sub-windows is summed across steps, and a
closed-form small-sample bias floor b(m) = (2m−1)/(m(m−3)) is subtracted at each step.
Each per-step term decomposes exactly into an entropy/variance-contraction term (A), a
variance-trajectory term (B), and a mean-shift term (C).

## Folder layout

- `scripts/`  — analysis code (plain Python, no comments or docstrings)
- `results/` — CSV/text outputs consumed by the manuscript tables and figures


## Scripts

- `simulation.py`            — synthetic validation: bias floor, AR(1) sweep, kurtosis
                               stress, false-positive control, eight-family benchmark
                               (Tables 2–7).
- `table_7bc_components.py`  — per-component A/B/C standalone and incremental AUC on the
                               benchmark families (Tables 8 and 9).
- `appendix_tables.py`       — Appendix A.5 (bias-floor Monte-Carlo variance), C.2 (Bartlett
                               effective-n accuracy), C.3 (pre-whitening residual ACF), and
                               E (k-split exploratory AUC sweep). Reuses the canonical CD
                               core from `simulation.py` verbatim.
- `supplementary_analyses.py`— SE-augmented Table 2, Appendix C.4 (lag-1 autocorrelation
                               finite-sample bias), and Appendix F.1 (joint AR(1)×kurtosis
                               false-positive stress). Reproduces one value each of Tables 2 and 3 as a
                               self-consistency check before writing new output.
- `effect_size_sweep.py`     — Appendix F.2 effect-size sensitivity sweep (mean-ramp and
                               V-shaped-variance families).
- `k_dependence.py`          — dependence of debiased CD on the number of transitions k for a
                               unit mean ramp over 2,142 samples (§6.5, limitation xi).
- `figures.py`               — Figures 1–5 (pipeline schematic, bias floor, AR(1) sweep,
                               kurtosis, benchmark).
- `rho_envelope_erpcore.py`  — real-data pipeline for ERP CORE P3 (active visual oddball).
- `rho_envelope_ds006018.py` — real-data pipeline for OpenNeuro ds006018.
- `realdata_analysis.py`     — autocorrelation boundary (Table 10), component magnitudes
                               (Table 13), and Target-vs-Standard discrimination (Table 14,
                               ERP CORE only) from the per-trial CSVs.
- `make_figure6.py`          — Figure 6 (real-data autocorrelation: native vs decimated).
- `realdata_cd.py`           — real-data CD, self-contained. For each recording: 0.1–30 Hz FIR
                               band-pass; ICA (20 components, FastICA, seed 42, fitted on a 1 Hz
                               high-passed copy; components matching the EOG channels removed);
                               event codes read from the trailing digits of each annotation;
                               one trial set per participant, rejected on the largest
                               peak-to-peak (default 100 µV) over −200 to 800 ms across all
                               scalp channels under each of the as-recorded, Cz, linked-mastoid
                               and average references; baseline −200 to 0 ms.
                               Fz under the as-recorded reference, Cz (ERP CORE only), linked
                               mastoids, the scalp average and the surface Laplacian (CSD), at
                               the native rate and at 128 Hz. Writes per trial the realised m,
                               n_eff and k, the bias subtracted and the closed-form k·b(n_eff),
                               and flags windows with m < 8. As a positive control it also
                               writes the conventional mean amplitude of the baseline-corrected,
                               undifferenced epoch at Fz and Pz in each window (`amp_fz`,
                               `amp_pz`; µV, or µV/m² for the surface Laplacian).

### Running `realdata_cd.py`

Place `realdata_cd.py` in `<root>/scripts/<any folder>/`. Without arguments it reads
`<root>/data/erp_core_P3` (ERP CORE, BIDS layout) and `<root>/data_ds006018/ds006018`
(ds006018, BIDS layout, task `visualoddball`) and writes to `<root>/results/P300`:

```
python realdata_cd.py --dataset erpcore --jobs 8
python realdata_cd.py --dataset ds006018 --jobs 8
```

Each job holds one recording with its ICA and surface Laplacian in memory (about 1.5 GB for
ERP CORE). On a machine with limited memory, run the participants in parts, one job at a
time; each part is a separate process, so memory is released between parts:

```
python realdata_cd.py --dataset erpcore --part 1/8 --jobs 1
...
python realdata_cd.py --dataset erpcore --part 8/8 --jobs 1
python realdata_cd.py --dataset erpcore --jobs 1
```

With `--no-ica` every output name carries `_noica` (for example `cd_realdata_erpcore_noica_rho.csv`),
so the runs with and without ICA can share one output folder and cache.
`--part K/N` processes the K-th of N consecutive groups of participants and stores them in the
cache without writing output files; the final run without `--part` reads the cache, computes
any participant still missing and writes the output files. `scripts/run_realdata_cd.bat` runs
both datasets this way on Windows, first with ICA and then without. If any participant fails (for example, memory exhausted),
no output files are written and the failed participants are listed; running the same command
again recomputes only them.

`--erpcore`, `--ds006018`, `--ds006018-subdir` and `--out` override the folders; `--refs`,
`--reject`, `--no-ica`, `--ica-n` and `--subjects` change the analysis; `--summary-only`
rebuilds the summary files from the per-trial files. Every output name starts with
`cd_realdata_<dataset>`: per-trial files `_pertrial_<derivation>.csv.gz` (one per derivation,
written at full precision), `_subjects.csv` (inclusion and trial counts), `_rho.csv`
(autocorrelation per window), `_bias.csv` (realised m, n_eff, k and the bias check),
`_finite.csv` (finite-value rate), `_auc.csv` (target versus standard Mann–Whitney AUC pooled
over trials, all windows and windows with m ≥ 8) and `_auc_subject.csv` (AUC within each
participant with at least 5 trials per class, Wilcoxon signed-rank test of the participant AUCs
against 0.5, Holm correction over all rows of a dataset). Per-participant results are cached in `<out>/cache` as
`cd_realdata_*.pkl`; delete them to recompute. A run restricted with `--subjects` adds the
participant list to the output names.

Run any script from inside the folder holding its inputs. The two real-data pipelines read
raw data from the local paths set at the top of each file (`DATA_ROOT` / `CACHE_DIR`) and
write `*_pertrial.csv`, `*_summary.csv`, and `*_p3.txt`. `realdata_analysis.py` and
`make_figure6.py` read the per-trial CSVs in `results/`.

## Estimator configurations (matches manuscript Table 1a)

Every configuration used in the paper, in one place:

- Simulations (§3–4): fixed high-resolution setting, k = 20 transitions (21
  non-overlapping sub-windows), m = 102 samples per sub-window. Chosen to isolate
  estimator behaviour from sub-window-length artefacts. `DEFAULT_K = 20`,
  `DEFAULT_M = 102` in `simulation.py`.
- Real data (§5): adaptive rule of Equation (2). k = 15 default (`N_STEPS_FIXED = 15`),
  reduced per window to k_eff so that each sub-window keeps m = floor(n/(k_eff+1)) ≥ 8
  samples (`MIN_WIN_SAMPLES = 8`). Realised per-window k_eff: ERP CORE 15,15,11,15;
  ds006018 11,8,5,15 (baseline → late; see Table 11).
- Sampling rate: ERP CORE 1024 Hz, ds006018 500 Hz (native). Optional decimation to
  128 Hz (`DECIM_SFREQ = 128.0`), applied to each analysis window after epoching.
- AR(1) pre-whitening: adaptive, triggered when the median per-sub-window lag-1
  autocorrelation exceeds 0.20 (`RHO_THRESHOLD = 0.20`).
- Two-split special case (k = 1): available; a larger m is recommended so the per-step
  bias floor, which then equals the trial-level floor, stays small relative to signal.
- Shared across all configurations: `RNG_SEED = 20260519`, the per-step closed-form bias
  floor b(m) = (2m−1)/(m(m−3)), and per-step debiasing.

## Scale-invariance and sampling rate (important)

`compute_cd` normalises each trial's differenced signal to unit standard deviation before
estimation. CD's components are scale-invariant by construction; the normalisation lifts
small-amplitude inputs (e.g. EEG in volts, ~1e-6) off the absolute 1e-12 numerical floors
that would otherwise clamp the variance and bias the autocorrelation. Both real-data scripts
embed `_self_test_scale_invariance()` (run at the start of `main()`, or via
`python <script>.py --self-test`), which asserts that a signal and its 1e-6-scaled copy
yield identical rho, A, B, C, and CD.

Sampling rate is controlled by `DECIM_SFREQ` at the top of each real-data script:
- `DECIM_SFREQ = None`  → native rate (1024 Hz ERP CORE / 500 Hz ds006018); outputs untagged.
- `DECIM_SFREQ = 128.0` → analysis windows decimated to 128 Hz; outputs tagged `_decim128`.
Decimation is applied to each extracted analysis window after epoching at the native rate,
never to the continuous recording (raw-level resampling would jitter event timing).

## Real-data result summary

On single-trial first-differenced Fz from two independent acquisitions:
- At native rate the differenced-signal lag-1 autocorrelation is high (median rho ≈ 0.86–0.97),
  above the estimator's reliable AR(1) range; first-differencing does not whiten oversampled
  band-limited EEG.
- The pre-registered Target-vs-Standard discrimination is at chance (AUC ≈ 0.50–0.55) for CD,
  for each component, and for the endpoint summaries, across all windows.
- Decimation to 128 Hz returns the long windows (baseline, late) to the reliable range and
  collapses the components to O(1), but the short windows become under-sampled and the
  discrimination null persists — so the null is not an artifact of the autocorrelation regime.

## Datasets

- ERP CORE P3 (active visual oddball; 1024 Hz). Read locally via MNE.
- OpenNeuro ds006018 (Isbell et al. 2025; 500 Hz; target and standard trials), via OpenNeuro.

## Result files (`results/`)

Simulation tables (from `simulation.py`):
- `table_2_bias_floor.csv`         — Table 2, closed-form vs empirical bias floor.
- `table_2_bias_floor_with_se.csv` — SE-augmented Table 2 (from `supplementary_analyses.py`).
- `table_3_ar1_sweep.csv`          — Table 3, AR(1) correction-strategy sweep.
- `table_4_kurtosis.csv`           — Table 4, Student-t kurtosis stress.
- `table_5_perm_power.csv`         — Table 5, empirical-null detection power (α = 0.05;
                                     empirical-null threshold ≈ 0.196).
- `table_6_fpr_control.csv`        — Table 6, false-positive control by branch.
- `table_7_benchmark.csv`          — Table 7, eight-family CD-vs-endpoint AUC benchmark.

Component tables (from `table_7bc_components.py`):
- `table_8_components_standalone.csv`   — Table 8, standalone A/B/C AUC.
- `table_9_components_incremental.csv`  — Table 9, incremental (endpoints+ABC) AUC.

Real-data tables (from `rho_envelope_*.py` and `realdata_analysis.py`):
- `rho_erpcore_native_pertrial.csv`, `rho_erpcore_decim128_pertrial.csv`,
  `rho_erpcore_decim128_summary.csv`, `rho_erpcore_decim128_p3.txt`  — ERP CORE per-trial/summary.
- `rho_ds006018_native_pertrial.csv`, `rho_ds006018_decim128_pertrial.csv`,
  `rho_ds006018_decim128_summary.csv`, `rho_ds006018_decim128_p3.txt` — ds006018 per-trial/summary.
- `table_10_realdata_rho_boundary.csv`  — Table 10, per-window autocorrelation boundary (both datasets).
- `table_11_realdata_biasfloor.csv`     — Table 11, per-step bias-floor precision (ERP CORE).
- `table_12_realdata_finitevalue.csv`   — Table 12, finite-value rate (ERP CORE).
- `table_13_realdata_components.csv`     — Table 13, per-window A/B/C magnitudes (both datasets).
- `table_14_realdata_discrimination.csv` — Table 14, Target-vs-Standard discrimination (ERP CORE only).

Appendix tables (from `appendix_tables.py`, `supplementary_analyses.py`, `effect_size_sweep.py`):
- `appendix_a5_variance.csv`      — Appendix A.5, bias-floor Monte-Carlo variance.
- `appendix_c2_bartlett.csv`      — Appendix C.2, Bartlett effective-n accuracy.
- `appendix_c3_prewhiten_acf.csv` — Appendix C.3, pre-whitening residual autocorrelation.
- `appendix_c4_rho_bias.csv`      — Appendix C.4, lag-1 autocorrelation finite-sample bias.
- `appendix_e_ksplit.csv`         — Appendix E, k-split exploratory AUC.
- `appendix_f1_joint_stress.csv`  — Appendix F.1, joint AR(1)×kurtosis false-positive rates.
- `appendix_f2_effect_size.csv`   — Appendix F.2, effect-size sensitivity sweep.
- `appendix_e2_k_dependence.csv`  — step-count dependence of debiased CD (from `k_dependence.py`).

All simulation, appendix, and component CSVs regenerate byte for byte from the fixed seed
(`RNG_SEED = 20260519`) in `simulation.py`, independent of `PYTHONHASHSEED` (each Table 5
family is seeded with a CRC-32 of its name). Verified with numpy 2.5.3, pandas 3.0.5,
scipy 1.18.1 and scikit-learn 1.9.1.

# Cumulative Divergence (CD): reproducibility package

Repository: https://github.com/erbiber/CD  
Pre-registration: https://doi.org/10.17605/OSF.IO/HEAXQ (`preregistration.pdf`)

Code, results and figures for the cumulative-divergence (CD) estimator, its validation by
simulation, its benchmark against classical within-window statistics, and its application to
two public visual-oddball EEG datasets. CD splits a window into k+1 consecutive sub-windows,
sums the per-step Gaussian Kullback–Leibler divergence between neighbouring sub-windows, and
subtracts the closed-form null floor b(m) = (2m−1)/(m(m−3)) at each step. Each per-step term
splits exactly into an entropy-reduction term (A), a variance-shrinkage term (B) and a
mean-shift term (C). An exactly unbiased per-step estimator (Equation 6 of the paper) is
computed from the same component sums.

## Folder layout

- `scripts/`  — analysis code (plain Python, no comments or docstrings)
- `results/`  — every CSV behind the tables and figures of the paper
- `figures/`  — Figures 1–6 as used in the paper
- `preregistration.pdf`, `requirements.txt`, `environment.yml`

## Where each table and figure comes from

| Paper | File(s) in `results/` | Script |
|---|---|---|
| Table 1 (notation), Table 2 (configurations) | — | — |
| Table 3, Figure 2 | `table_2_bias_floor.csv` | `simulation.py` |
| Table 4, Figure 3 | `table_3_ar1_sweep.csv` | `simulation.py` |
| Table 5, Figure 4 | `table_4_kurtosis.csv` | `simulation.py` |
| Table 6 | `table_5_perm_power.csv` | `simulation.py` |
| Table 7 | `table_6_fpr_control.csv` | `simulation.py` |
| Table 8 | `table_u1_step_bias.csv` | `unbiased_cd.py` |
| Table 9 | `table_u2_path_window_length.csv` | `unbiased_cd.py` |
| Table 10, Figure 5 | `table_7_benchmark.csv` | `simulation.py` |
| Table 11 | `table_8_components_standalone.csv` | `table_7bc_components.py` |
| Table 12 | `table_9_components_incremental.csv` | `table_7bc_components.py` |
| Table 13 | `table_7x_classical_comparators.csv` | `comparators.py` |
| Table 14 | `table_7y_order_sensitivity.csv` | `comparators.py` |
| Table 15 | `table_7z_weak_effects.csv` | `comparators.py` |
| Table 16 | `cd_realdata_erpcore_noica_prereg.csv` | `realdata_cd.py` |
| Table 17, Figure 6 | `cd_realdata_<dataset>_rho.csv`, `cd_realdata_<dataset>_bias.csv` | `realdata_cd.py`; figure by `make_figure6.py` |
| Table 18 | `cd_realdata_<dataset>_auc_subject.csv`, `_auc_subject_explore.csv`, `_unbiased_auc_subject.csv` | `realdata_cd.py`, `realdata_post.py` |
| Table 19 | `incremental_<dataset>_mastoid.csv` | `realdata_incremental.py` |
| Table 20 | `cd_realdata_<dataset>_unbiased_summary.csv` | `realdata_post.py` |
| Table 21 (guide) | — | — |
| Table A1 | `appendix_a5_variance.csv` | `appendix_tables.py` |
| Table C1 | `appendix_c2_bartlett.csv` | `appendix_tables.py` |
| Table C2 | `appendix_c3_prewhiten_acf.csv` | `appendix_tables.py` |
| Table C3 | `appendix_c4_rho_bias.csv` | `supplementary_analyses.py` |
| Table E1 | `appendix_e_ksplit.csv` | `appendix_tables.py` |
| Table F1 | `appendix_f1_joint_stress.csv` | `supplementary_analyses.py` |
| Table F2 | `appendix_f2_effect_size.csv` | `effect_size_sweep.py` |
| Table G1 | `table_u3_null.csv` | `unbiased_cd.py` |
| Table G2 | `table_u4_ar_prewhitened.csv` | `unbiased_cd.py` |
| Table H1 | `appendix_g_sector_floors.csv` | `comparators.py` |
| Table H2 | `appendix_g_parity_orders.csv` | `comparators.py` |
| Table H3 | `appendix_g_parity_by_family.csv` | `comparators.py` |
| Figure 1 | — (schematic) | `figures.py` |

The CSV names keep the numbering of the simulation scripts, so `table_2_bias_floor.csv` is
Table 3 of the paper; the table above is the reference.

Further outputs cited in the text:
- `table_2_bias_floor_with_se.csv` — an independent replication of Table 3 on a separate
  random stream, with standard errors (`supplementary_analyses.py`).
- `table_9x_incremental_over_classical.csv` — CD and its components added to the classical
  statistics on the benchmark families (`comparators.py`).
- `appendix_e2_k_dependence.csv` — dependence of the floor-debiased CD on the number of
  transitions k for a unit mean ramp over 2,142 samples (§6.5; `k_dependence.py`).
- `appendix_u5_fisher_action.csv` — written by `unbiased_cd.py`; not reported in the paper.

Real-data dataset names: `erpcore` (ERP CORE, with ICA), `erpcore_noica` (ERP CORE, without
ICA) and `ds006018` (OpenNeuro ds006018, with ICA).

## Scripts

- `simulation.py` — canonical simulation pipeline: bias floor, AR(1) sweep, kurtosis stress,
  empirical-null power, false-positive control and the eight-family benchmark (Tables 3–7
  and 10). Holds the CD core shared by all simulation scripts.
- `unbiased_cd.py` — the exactly unbiased estimator of §2.10: per-step bias, path bias, null
  and pre-whitened AR(1) checks (Tables 8, 9, G1, G2).
- `table_7bc_components.py` — standalone and incremental AUC of the components A, B and C
  on the benchmark families (Tables 11 and 12).
- `comparators.py` — classical within-window statistics (ANOVA F, Bartlett, likelihood
  ratio, von Neumann ratios, slopes) on the benchmark trials, order sensitivity and weak
  effects (Tables 13–15), and the sector floors and parity parts of Appendix H.
- `appendix_tables.py` — Appendices A.5, C.2, C.3 and E (Tables A1, C1, C2, E1).
- `supplementary_analyses.py` — the replication of Table 3 with standard errors, Appendix C.4
  and Appendix F.1 (Tables C3, F1). It first reproduces two values of Tables 3 and 4 as a
  self-consistency check.
- `effect_size_sweep.py` — Appendix F.2 (Table F2).
- `k_dependence.py` — step-count dependence (§6.5).
- `figures.py` — Figures 1–5.
- `realdata_cd.py` — real-data pipeline for both datasets (see below).
- `realdata_post.py` — unbiased CD from the per-trial component sums written by
  `realdata_cd.py` (Tables 18 and 20).
- `realdata_incremental.py` — incremental test of CD over the classical statistics on real
  data (Table 19).
- `make_figure6.py` — Figure 6 from `cd_realdata_<dataset>_rho.csv`.
- `run_realdata_cd.bat` — runs `realdata_cd.py` for both datasets in parts on Windows.

The simulation and appendix scripts write to the folder they are run from, and `figures.py`
reads its CSVs from there; run each from inside the folder holding its inputs.

## Running the real-data pipeline

`realdata_cd.py` reads raw BIDS data. For each recording it applies a 0.1–30 Hz FIR band-pass;
ICA (20 components, FastICA, seed 42, fitted on a 1 Hz high-passed copy; components matching
the EOG channels removed); event codes read from the trailing digits of each annotation; one
trial set per participant, rejected when the peak-to-peak amplitude from −200 to 800 ms
exceeds 100 µV in any scalp channel under any of the four voltage derivations; and a −200 to
0 ms baseline. Fz is analysed under the reference as recorded, Cz (ERP CORE only), the mean of
the two mastoid electrodes, the scalp average and the surface Laplacian (CSD), at the native
rate and after decimation of each extracted window to 128 Hz. As a positive control it also
writes the mean amplitude of the baseline-corrected, undifferenced epoch at Fz and Pz
(`amp_fz`, `amp_pz`; µV, or µV/m² for the surface Laplacian).

Without arguments the script reads `<root>/data/erp_core_P3` (ERP CORE) and
`<root>/data_ds006018/ds006018` (task `visualoddball`), where `<root>` is two folders above the
script, and writes to `<root>/results/P300`. `--erpcore`, `--ds006018`, `--ds006018-subdir`
and `--out` override these folders. To write into this repository's `results/` folder:

```
python scripts/realdata_cd.py --dataset erpcore --erpcore <ERP CORE folder> --out results --jobs 1
python scripts/realdata_cd.py --dataset ds006018 --ds006018 <ds006018 folder> --out results --jobs 1
python scripts/realdata_cd.py --dataset erpcore --erpcore <ERP CORE folder> --out results --jobs 1 --no-ica
```

Each job holds one recording with its ICA and surface Laplacian in memory (about 1.5 GB for
ERP CORE). With limited memory, run the participants in parts (`--part K/N`); each part is
stored in the cache without writing output files, and a final run without `--part` writes
them. `run_realdata_cd.bat` does this for both datasets, with and without ICA. If a
participant fails, no output files are written and the failed participants are listed;
running the same command again recomputes only them. `--summary-only` rebuilds the summary
files from the per-trial files. Per-participant results are cached in `<out>/cache`.

Outputs, each named `cd_realdata_<dataset>`: per-trial files `_pertrial_<derivation>.csv.gz`
(full precision), `_subjects.csv`, `_rho.csv`, `_bias.csv`, `_finite.csv`, `_auc.csv`,
`_auc_subject.csv`, `_auc_subject_explore.csv` and `_prereg.csv`.

The two post-processing scripts read the per-trial files:

```
python scripts/realdata_post.py --dataset erpcore --out results
python scripts/realdata_post.py --dataset ds006018 --out results
python scripts/realdata_post.py --dataset erpcore --no-ica --out results
python scripts/realdata_incremental.py results/cd_realdata_erpcore_pertrial_mastoid.csv.gz mastoid results/incremental_erpcore_mastoid.csv
python scripts/realdata_incremental.py results/cd_realdata_erpcore_noica_pertrial_mastoid.csv.gz mastoid results/incremental_erpcore_noica_mastoid.csv
python scripts/realdata_incremental.py results/cd_realdata_ds006018_pertrial_mastoid.csv.gz mastoid results/incremental_ds006018_mastoid.csv
python scripts/make_figure6.py results
```

`make_figure6.py` writes `figure_6_rho_distributions.png` to the `figures/` folder next to the
results folder it reads.

## Estimator configurations (paper Table 2)

- Simulations (§3–4): k = 20 transitions (21 non-overlapping sub-windows), m = 102 samples
  per sub-window (`DEFAULT_K = 20`, `DEFAULT_M = 102` in `simulation.py`).
- Real data (§5): adaptive rule of Equation (2). k = 15 by default, reduced per window to
  k_eff so that each sub-window keeps m = floor(n/(k_eff+1)) ≥ 8 samples
  (`MIN_WIN_SAMPLES = 8`). Realised k_eff at the native rate: ERP CORE 15, 15, 11, 15;
  ds006018 11, 8, 5, 15 (baseline to late; paper Table 17).
- Sampling rate: ERP CORE 1024 Hz, ds006018 500 Hz; optional decimation of each extracted
  window to 128 Hz after epoching, never of the continuous recording.
- AR(1) pre-whitening: applied when the median per-sub-window lag-1 autocorrelation exceeds
  0.20 (`RHO_THRESHOLD = 0.20`).
- Shared: `RNG_SEED = 20260519`, the per-step closed-form floor b(m) and per-step debiasing.

The forward real-data estimator normalises each trial's differenced segment to unit standard
deviation; the components are scale-invariant, and `realdata_cd.py` checks this at the start
of every run.

## Main results

- In simulation the closed-form floor matched the plug-in mean within 1.4 % for sub-windows
  of 13–1,000 samples; the unbiased estimator showed no detectable bias, whereas floor
  subtraction overstated a smooth path divergence; AR(1) pre-whitening kept null means within
  ±0.5 up to ρ = 0.95; heavy tails inflated false positives.
- Classical statistics on the same sub-windows detected every simulated change at least as
  well as CD, and CD added essentially no information to them.
- On ERP CORE and ds006018 the pre-registered checks of finite values and floor arithmetic
  held, but single-trial autocorrelation exceeded the characterised range. Neither CD nor the
  classical statistics separated targets from standards (|AUC − 0.5| ≤ 0.084), whereas a Pz
  amplitude control did (AUC 0.63–0.66).

## Datasets

- ERP CORE P3 (Kappenman et al., 2021; active visual oddball; 1024 Hz).
- OpenNeuro ds006018 (Isbell et al., 2025; active visual oddball; 500 Hz).

## Reproducibility

All simulation, appendix and component CSVs regenerate byte for byte from the fixed seed
(`RNG_SEED = 20260519`), independent of `PYTHONHASHSEED` (each Table 6 family is seeded with a
CRC-32 of its name). `realdata_post.py`, `realdata_incremental.py` and `make_figure6.py`
reproduce their outputs byte for byte from the per-trial files in `results/`. Verified with
numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, scikit-learn 1.9.1 and matplotlib 3.11.2.

from __future__ import annotations
import math
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
RNG_SEED = 20260519
OUT_DIR = Path('.')
DEFAULT_M = 102
DEFAULT_K = 20
DEFAULT_N_TRIALS = 600
DEFAULT_N_NULL_CALIB = 5000

def bias_floor(n: int) -> float:
    if n <= 3:
        return float('nan')
    return (2 * n - 1) / (n * (n - 3))

def per_step_sym_kl(a: np.ndarray, b: np.ndarray) -> float:
    s2_a, s2_b = (a.var(ddof=1), b.var(ddof=1))
    if s2_a <= 0 or s2_b <= 0:
        return float('nan')
    mu_a, mu_b = (a.mean(), b.mean())
    df = 0.5 * math.log(s2_a / s2_b) + (s2_b + (mu_b - mu_a) ** 2) / (2 * s2_a) - 0.5
    dr = 0.5 * math.log(s2_b / s2_a) + (s2_a + (mu_a - mu_b) ** 2) / (2 * s2_b) - 0.5
    return 0.5 * (df + dr)

def lag1_rho(w: np.ndarray) -> float:
    if w.size < 2:
        return 0.0
    x0 = w[:-1] - w[:-1].mean()
    x1 = w[1:] - w[1:].mean()
    denom = math.sqrt((x0 ** 2).sum() * (x1 ** 2).sum())
    if denom == 0:
        return 0.0
    return float((x0 * x1).sum() / denom)

def ar1_prewhiten(x: np.ndarray, rho: float | None=None) -> np.ndarray:
    if rho is None:
        rho = lag1_rho(x)
    rho = float(np.clip(rho, -0.99, 0.99))
    return x[1:] - rho * x[:-1]

def cumulative_divergence(trial: np.ndarray, m: int, k: int, branch: str='iid') -> float:
    windows = [trial[j * m:(j + 1) * m] for j in range(k + 1)]
    if branch == 'prewhiten':
        pw_windows = [ar1_prewhiten(w) for w in windows]
        raw = sum((per_step_sym_kl(pw_windows[j], pw_windows[j + 1]) for j in range(k)))
        return raw - k * bias_floor(m - 1)
    raw = sum((per_step_sym_kl(windows[j], windows[j + 1]) for j in range(k)))
    if branch == 'iid':
        return raw - k * bias_floor(m)
    if branch == 'bartlett':
        rho_med = float(np.median([lag1_rho(w) for w in windows]))
        rho_med = max(0.0, rho_med)
        n_eff = max(int(round(m * (1 - rho_med) / max(1 + rho_med, 1e-06))), 4)
        return raw - k * bias_floor(n_eff)
    raise ValueError(f'unknown branch: {branch!r}')

def gen_null_iid(rng: np.random.Generator, m: int, k: int) -> np.ndarray:
    return rng.standard_normal(m * (k + 1))

def gen_null_ar(rng: np.random.Generator, m: int, k: int, rho: float) -> np.ndarray:
    n = m * (k + 1)
    eps = rng.standard_normal(n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + math.sqrt(max(1 - rho ** 2, 1e-09)) * eps[i]
    return x

def gen_null_t(rng: np.random.Generator, m: int, k: int, df: float) -> np.ndarray:
    n = m * (k + 1)
    raw = student_t.rvs(df, size=n, random_state=rng)
    scale = math.sqrt((df - 2) / df) if df > 2 else 1.0
    return raw * scale

def gen_null_ar085(rng, m, k):
    return gen_null_ar(rng, m, k, 0.85)

def gen_null_t4(rng, m, k):
    return gen_null_t(rng, m, k, 4.0)

def gen_mean_ramp(rng, m, k):
    means = np.linspace(0, 1.0, k + 1)
    return np.concatenate([rng.standard_normal(m) + means[j] for j in range(k + 1)])

def gen_var_ramp(rng, m, k):
    sigmas = np.linspace(1.0, 1.8, k + 1)
    return np.concatenate([rng.standard_normal(m) * sigmas[j] for j in range(k + 1)])

def gen_v_variance(rng, m, k):
    half = (k + 1) // 2 + 1
    rest = k + 1 - half + 1
    up = np.linspace(1.0, 1.8, half)
    down = np.linspace(1.8, 1.0, rest)[1:]
    sigmas = np.concatenate([up, down])[:k + 1]
    return np.concatenate([rng.standard_normal(m) * sigmas[j] for j in range(k + 1)])

def gen_step_mean(rng, m, k):
    mid = (k + 1) // 2
    means = np.where(np.arange(k + 1) < mid, 0.0, 0.7)
    return np.concatenate([rng.standard_normal(m) + means[j] for j in range(k + 1)])

def gen_oscillatory(rng, m, k):
    j = np.arange(k + 1)
    sigmas = 1.0 + 0.4 * np.sin(2 * math.pi * j / (k + 1))
    return np.concatenate([rng.standard_normal(m) * sigmas[ji] for ji in j])

def gen_mixed(rng, m, k):
    means = np.linspace(0, 0.6, k + 1)
    sigmas = np.linspace(1.0, 1.5, k + 1)
    return np.concatenate([rng.standard_normal(m) * sigmas[j] + means[j] for j in range(k + 1)])
BENCHMARK_FAMILIES: list[tuple[str, str, Callable]] = [('null_ar085', 'AR(1) rho=0.85, no change', gen_null_ar085), ('null_t4', 'Student-t df=4, no change', gen_null_t4), ('mean_ramp', 'Mean ramp 0 -> +1.0', gen_mean_ramp), ('var_ramp', 'Variance ramp sigma 1.0 -> 1.8', gen_var_ramp), ('v_variance', 'V-shaped variance', gen_v_variance), ('step_mean', 'Mid-trial step +0.7', gen_step_mean), ('oscillatory', 'Oscillatory variance', gen_oscillatory), ('mixed', 'Mean ramp + variance ramp', gen_mixed)]

def _sample_entropy(x: np.ndarray, m_emb: int=2, r: float | None=None) -> float:
    if x.size < m_emb + 2:
        return float('nan')
    if r is None:
        r = 0.2 * x.std(ddof=1)
    if r <= 0:
        return float('nan')
    n = x.size

    def _phi(mm: int) -> float:
        templates = np.array([x[i:i + mm] for i in range(n - mm + 1)])
        if templates.size == 0:
            return 0.0
        cnt, total = (0, 0)
        for i in range(templates.shape[0]):
            diffs = np.max(np.abs(templates - templates[i]), axis=1)
            cnt += int(np.sum((diffs < r) & (np.arange(templates.shape[0]) != i)))
            total += templates.shape[0] - 1
        return cnt / max(total, 1)
    A = _phi(m_emb + 1)
    B = _phi(m_emb)
    if A <= 0 or B <= 0:
        return float('nan')
    return -math.log(A / B)

def endpoint_summaries(trial: np.ndarray, m: int, k: int) -> dict:
    first = trial[:m]
    last = trial[k * m:(k + 1) * m]
    return {'delta_mean': abs(last.mean() - first.mean()), 'delta_sd': abs(last.std(ddof=1) - first.std(ddof=1)), 'delta_rms': abs(math.sqrt((last ** 2).mean()) - math.sqrt((first ** 2).mean())), 'delta_samp_en': abs(_sample_entropy(last) - _sample_entropy(first))}

def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float('nan')
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))).astype(float) + 1
    u_pos = ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2
    return float(u_pos / (pos.size * neg.size))

def table_2_bias_floor(out_path: Path, n_trials_default: int=8000) -> pd.DataFrame:
    n_list = [(13, n_trials_default), (21, n_trials_default), (30, n_trials_default), (51, n_trials_default), (102, n_trials_default), (256, 50000), (1000, 50000)]
    rows = []
    for n, n_trials in n_list:
        rng = np.random.default_rng(RNG_SEED + 1 + n)
        vals = np.empty(n_trials)
        for t in range(n_trials):
            a = rng.standard_normal(n)
            b = rng.standard_normal(n)
            vals[t] = per_step_sym_kl(a, b)
        vals = vals[np.isfinite(vals)]
        b_closed = bias_floor(n)
        rel_err_pct = abs(b_closed - vals.mean()) / abs(b_closed) * 100
        rows.append({'n': n, 'closed_form_b': b_closed, 'empirical_mean': float(vals.mean()), 'empirical_sd': float(vals.std(ddof=1)), 'rel_error_pct': float(rel_err_pct), 'n_trials': int(vals.size)})
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df

def table_3_ar1_sweep(out_path: Path, m: int=DEFAULT_M, k: int=DEFAULT_K, n_trials: int=800) -> pd.DataFrame:
    rho_list = [0.0, 0.1, 0.2, 0.4, 0.55, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    rows = []
    for rho in rho_list:
        rng = np.random.default_rng(RNG_SEED + 11 + int(rho * 100))
        iid_vals = np.empty(n_trials)
        bart_vals = np.empty(n_trials)
        pw_vals = np.empty(n_trials)
        for t in range(n_trials):
            trial = gen_null_ar(rng, m, k, rho)
            iid_vals[t] = cumulative_divergence(trial, m, k, 'iid')
            bart_vals[t] = cumulative_divergence(trial, m, k, 'bartlett')
            pw_vals[t] = cumulative_divergence(trial, m, k, 'prewhiten')
        rows.append({'rho': rho, 'iid_mean': float(np.nanmean(iid_vals)), 'iid_sd': float(np.nanstd(iid_vals, ddof=1)), 'bartlett_mean': float(np.nanmean(bart_vals)), 'bartlett_sd': float(np.nanstd(bart_vals, ddof=1)), 'prewhiten_mean': float(np.nanmean(pw_vals)), 'prewhiten_sd': float(np.nanstd(pw_vals, ddof=1)), 'n_trials': n_trials})
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df

def table_4_kurtosis(out_path: Path, m: int=DEFAULT_M, k: int=DEFAULT_K, n_trials: int=2000) -> pd.DataFrame:
    df_list = [3.5, 4.0, 5.0, 7.0, 10.0, 30.0, 1000000.0]
    rows = []
    for nu in df_list:
        rng = np.random.default_rng(RNG_SEED + 21 + (int(nu) if nu < 100000.0 else 999))
        cd_vals = np.empty(n_trials)
        emp_kurts = np.empty(n_trials)
        for t in range(n_trials):
            trial = gen_null_t(rng, m, k, nu)
            cd_vals[t] = cumulative_divergence(trial, m, k, 'iid')
            x = trial - trial.mean()
            v = x.var(ddof=1)
            emp_kurts[t] = (x ** 4).mean() / max(v ** 2, 1e-12) - 3.0
        theory_excess = 6.0 / (nu - 4) if nu > 4 else float('inf')
        rows.append({'df': nu if nu < 100000.0 else float('inf'), 'excess_kurtosis_theory': theory_excess if math.isfinite(theory_excess) else 1000000.0, 'empirical_kurtosis_excess': float(np.nanmean(emp_kurts)), 'debiased_cd_mean': float(np.nanmean(cd_vals)), 'debiased_cd_sd': float(np.nanstd(cd_vals, ddof=1)), 'n_trials': n_trials})
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df

def table_5_perm_power(out_path: Path, m: int=DEFAULT_M, k: int=DEFAULT_K, n_trials: int=DEFAULT_N_TRIALS, n_calib: int=DEFAULT_N_NULL_CALIB) -> pd.DataFrame:
    rng_null = np.random.default_rng(RNG_SEED + 31)
    null_cd = np.empty(n_calib)
    for t in range(n_calib):
        null_cd[t] = cumulative_divergence(gen_null_iid(rng_null, m, k), m, k, 'iid')
    threshold = float(np.nanpercentile(null_cd, 95))
    real_change = [('mean_ramp', gen_mean_ramp), ('var_ramp', gen_var_ramp), ('v_variance', gen_v_variance), ('step_mean', gen_step_mean), ('oscillatory', gen_oscillatory), ('mixed', gen_mixed)]
    rows = []
    for name, gen in real_change:
        rng = np.random.default_rng(RNG_SEED + 32 + zlib.crc32(name.encode()) % 1000)
        vals = np.empty(n_trials)
        for t in range(n_trials):
            vals[t] = cumulative_divergence(gen(rng, m, k), m, k, 'iid')
        vals_f = vals[np.isfinite(vals)]
        rows.append({'family': name, 'mean_CD': float(vals_f.mean()), 'sd_CD': float(vals_f.std(ddof=1)), 'n_trials': int(vals_f.size), 'empirical_null_threshold': threshold, 'n_null_calibration_trials': n_calib, 'power_alpha_0_05': float((vals_f > threshold).mean())})
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df

def table_6_fpr_control(out_path: Path, m: int=DEFAULT_M, k: int=DEFAULT_K, n_calib: int=DEFAULT_N_NULL_CALIB, n_trials: int=2000) -> pd.DataFrame:
    rng_iid = np.random.default_rng(RNG_SEED + 41)
    iid_null = np.array([cumulative_divergence(gen_null_iid(rng_iid, m, k), m, k, 'iid') for _ in range(n_calib)])
    iid_thr = float(np.nanpercentile(iid_null, 95))
    rng_pw = np.random.default_rng(RNG_SEED + 42)
    pw_null = np.array([cumulative_divergence(gen_null_iid(rng_pw, m, k), m, k, 'prewhiten') for _ in range(2000)])
    pw_thr = float(np.nanpercentile(pw_null, 95))
    rows = []
    for rho in [0.0, 0.2, 0.4, 0.55, 0.7, 0.85]:
        rng_a = np.random.default_rng(RNG_SEED + 51 + int(rho * 100))
        rng_b = np.random.default_rng(RNG_SEED + 52 + int(rho * 100))
        iid_vals = np.array([cumulative_divergence(gen_null_ar(rng_a, m, k, rho), m, k, 'iid') for _ in range(n_trials)])
        pw_vals = np.array([cumulative_divergence(gen_null_ar(rng_b, m, k, rho), m, k, 'prewhiten') for _ in range(n_trials)])
        rows.append({'null_regime': f'AR1_rho_{rho}', 'parameter': rho, 'fpr_iid_branch': float(np.nanmean(iid_vals > iid_thr)), 'fpr_prewhiten_branch': float(np.nanmean(pw_vals > pw_thr)), 'iid_threshold': iid_thr, 'pw_threshold': pw_thr, 'n_trials': n_trials})
    for nu in [3.5, 4.0, 5.0, 7.0, 10.0, 30.0, 1000000.0]:
        rng_t = np.random.default_rng(RNG_SEED + 61 + (int(nu) if nu < 100000.0 else 999))
        vals = np.array([cumulative_divergence(gen_null_t(rng_t, m, k, nu), m, k, 'iid') for _ in range(n_trials)])
        excess_k = 6.0 / max(nu - 4.0, 1e-06) if nu > 4.0 else float('inf')
        rows.append({'null_regime': f'student_t_df_{nu}' if nu < 100000.0 else 'student_t_gaussian_limit', 'parameter': nu, 'fpr_iid_branch': float(np.nanmean(vals > iid_thr)), 'fpr_prewhiten_branch': float('nan'), 'iid_threshold': iid_thr, 'pw_threshold': pw_thr, 'n_trials': n_trials, 'approx_excess_kurtosis': excess_k if math.isfinite(excess_k) else 1000000.0})
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df

def table_7_benchmark(out_path: Path, m: int=DEFAULT_M, k: int=DEFAULT_K, n_trials: int=DEFAULT_N_TRIALS) -> pd.DataFrame:
    rows = []
    for fam_name, fam_desc, gen_fn in BENCHMARK_FAMILIES:
        rng_pos = np.random.default_rng(RNG_SEED + 71)
        rng_neg = np.random.default_rng(RNG_SEED + 72)
        cd_pos = np.empty(n_trials)
        cd_neg = np.empty(n_trials)
        dm_pos = np.empty(n_trials)
        dm_neg = np.empty(n_trials)
        dsd_pos = np.empty(n_trials)
        dsd_neg = np.empty(n_trials)
        drms_pos = np.empty(n_trials)
        drms_neg = np.empty(n_trials)
        dse_pos = np.empty(n_trials)
        dse_neg = np.empty(n_trials)
        for t in range(n_trials):
            tp = gen_fn(rng_pos, m, k)
            tn = gen_null_iid(rng_neg, m, k)
            cd_pos[t] = cumulative_divergence(tp, m, k, 'iid')
            cd_neg[t] = cumulative_divergence(tn, m, k, 'iid')
            ep_p, ep_n = (endpoint_summaries(tp, m, k), endpoint_summaries(tn, m, k))
            dm_pos[t], dm_neg[t] = (ep_p['delta_mean'], ep_n['delta_mean'])
            dsd_pos[t], dsd_neg[t] = (ep_p['delta_sd'], ep_n['delta_sd'])
            drms_pos[t], drms_neg[t] = (ep_p['delta_rms'], ep_n['delta_rms'])
            dse_pos[t], dse_neg[t] = (ep_p['delta_samp_en'], ep_n['delta_samp_en'])
        rows.append({'family': fam_name, 'description': fam_desc, 'auc_cd': auc(cd_pos, cd_neg), 'auc_delta_mean': auc(dm_pos, dm_neg), 'auc_delta_sd': auc(dsd_pos, dsd_neg), 'auc_delta_rms': auc(drms_pos, drms_neg), 'auc_delta_samp_en': auc(dse_pos, dse_neg), 'n_trials_per_class': n_trials})
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df

def main():
    print('CD canonical simulation pipeline')
    print(f'  RNG_SEED = {RNG_SEED}')
    print(f'  Path-integral default: m = {DEFAULT_M}, k = {DEFAULT_K}')
    print(f'  Outputs in: {OUT_DIR.resolve()}')
    print()
    print('[1/6] Table 2 — closed-form bias-floor validation (§3.1)')
    df2 = table_2_bias_floor(OUT_DIR / 'table_2_bias_floor.csv')
    print(df2.to_string(index=False))
    print()
    print('[2/6] Table 3 — AR(1) sweep, three correction branches (§3.2)')
    df3 = table_3_ar1_sweep(OUT_DIR / 'table_3_ar1_sweep.csv')
    print(df3.to_string(index=False))
    print()
    print('[3/6] Table 4 — kurtosis stress, Student-t (§3.3)')
    df4 = table_4_kurtosis(OUT_DIR / 'table_4_kurtosis.csv')
    print(df4.to_string(index=False))
    print()
    print('[4/6] Table 5 — empirical-null detection power (§3.4)')
    df5 = table_5_perm_power(OUT_DIR / 'table_5_perm_power.csv')
    print(df5.to_string(index=False))
    print()
    print('[5/6] Table 6 — FPR control across operating envelope (§3.5)')
    df6 = table_6_fpr_control(OUT_DIR / 'table_6_fpr_control.csv')
    print(df6.to_string(index=False))
    print()
    print('[6/6] Table 7 — eight-family benchmark vs endpoint summaries (§4)')
    df7 = table_7_benchmark(OUT_DIR / 'table_7_benchmark.csv')
    print(df7.to_string(index=False))
    print()
    print('Done. Six CSVs written to', OUT_DIR.resolve())
if __name__ == '__main__':
    main()

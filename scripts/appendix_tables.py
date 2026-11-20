from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd

from simulation import (
    bias_floor, per_step_sym_kl, lag1_rho, ar1_prewhiten,
    cumulative_divergence, gen_null_ar, auc, RNG_SEED,
)

OUT_DIR = Path('.')


def table_a5_variance(out_path: Path, n_values=(21, 30, 51, 102, 256), n_trials: int = 50000) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for n in n_values:
        vals = np.empty(n_trials)
        for t in range(n_trials):
            a = rng.standard_normal(n)
            b = rng.standard_normal(n)
            vals[t] = per_step_sym_kl(a, b)
        bn = bias_floor(n)
        emp_mean = float(vals.mean())
        var = float(vals.var(ddof=1))
        sd_over_mean = float(vals.std(ddof=1) / emp_mean) if emp_mean != 0 else float('nan')
        rows.append(dict(n=n, bn=bn, empirical_mean=round(emp_mean, 5),
                          var=round(var, 5), sd_over_mean=round(sd_over_mean, 2)))
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def table_c2_bartlett(out_path: Path, nh: int = 100, n_trials: int = 4000,
                       rhos=(0.0, 0.2, 0.4, 0.55, 0.7)) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for rho in rhos:
        vals = np.empty(n_trials)
        for t in range(n_trials):
            x = gen_null_ar(rng, nh, 1, rho)
            a, b = x[:nh], x[nh:2 * nh]
            vals[t] = per_step_sym_kl(a, b)
        n_eff = nh * (1 - rho) / (1 + rho)
        b_neff = bias_floor(n_eff)
        mc_mean = float(vals.mean())
        rel_err = (mc_mean - b_neff) / b_neff
        rows.append(dict(rho=rho, n_eff=round(n_eff, 1), b_n_eff=round(b_neff, 4),
                          mc_mean_raw_kl=round(mc_mean, 4), relative_error_pct=round(rel_err * 100, 1)))
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def table_c3_prewhiten_acf(out_path: Path, length: int = 2000, n_trials: int = 2000,
                            rhos=(0.20, 0.55, 0.85)) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for rho in rhos:
        orig_acf = np.empty(n_trials)
        pw_acf = np.empty(n_trials)
        for t in range(n_trials):
            x = gen_null_ar(rng, length, 0, rho)
            orig_acf[t] = lag1_rho(x)
            y = ar1_prewhiten(x)
            pw_acf[t] = lag1_rho(y)
        rows.append(dict(true_rho=rho, mean_lag1_acf_original=round(float(orig_acf.mean()), 4),
                          mean_lag1_acf_prewhitened=round(float(pw_acf.mean()), 4)))
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def gen_e_v_variance(rng: np.random.Generator, n: int = 100) -> np.ndarray:
    third = n // 3
    sigmas = np.empty(n)
    sigmas[:third] = 2.0
    sigmas[third:2 * third] = 0.5
    sigmas[2 * third:] = 2.0
    return rng.standard_normal(n) * sigmas


def gen_e_step_mean(rng: np.random.Generator, n: int = 100) -> np.ndarray:
    half = n // 2
    means = np.where(np.arange(n) < half, 0.0, 1.0)
    return rng.standard_normal(n) + means


def gen_e_oscillatory(rng: np.random.Generator, n: int = 100) -> np.ndarray:
    t = np.arange(n) / n
    sigmas = 1.0 + 0.8 * np.sin(6 * math.pi * t)
    return rng.standard_normal(n) * sigmas


E_FAMILIES = [
    ('V-shaped variance', gen_e_v_variance),
    ('Step-mean change', gen_e_step_mean),
    ('Oscillatory variance', gen_e_oscillatory),
]


def table_e_ksplit(out_path: Path, n: int = 100, n_trials: int = 1000,
                    k_values=(1, 2, 3, 4)) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for fam_name, gen_fn in E_FAMILIES:
        pos_trials = [gen_fn(rng, n) for _ in range(n_trials)]
        null_trials = [rng.standard_normal(n) for _ in range(n_trials)]
        row = {'family': fam_name}
        for k in k_values:
            m = n // (k + 1)
            cd_pos = np.array([cumulative_divergence(tr[:m * (k + 1)], m, k, branch='iid') for tr in pos_trials])
            cd_neg = np.array([cumulative_divergence(tr[:m * (k + 1)], m, k, branch='iid') for tr in null_trials])
            row[f'k_{k}_auc'] = round(auc(cd_pos, cd_neg), 3)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def main():
    print('[1/4] Appendix A.5 -- bias-floor Monte Carlo variance')
    df_a5 = table_a5_variance(OUT_DIR / 'appendix_a5_variance.csv')
    print(df_a5.to_string(index=False))

    print('\n[2/4] Appendix C.2 -- Bartlett approximation accuracy')
    df_c2 = table_c2_bartlett(OUT_DIR / 'appendix_c2_bartlett.csv')
    print(df_c2.to_string(index=False))

    print('\n[3/4] Appendix C.3 -- pre-whitening residual ACF')
    df_c3 = table_c3_prewhiten_acf(OUT_DIR / 'appendix_c3_prewhiten_acf.csv')
    print(df_c3.to_string(index=False))

    print('\n[4/4] Appendix E -- k-split exploratory simulation')
    df_e = table_e_ksplit(OUT_DIR / 'appendix_e_ksplit.csv')
    print(df_e.to_string(index=False))


if __name__ == '__main__':
    main()

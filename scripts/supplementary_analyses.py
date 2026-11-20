from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from simulation import (
    bias_floor, per_step_sym_kl, lag1_rho, ar1_prewhiten,
    cumulative_divergence, RNG_SEED, DEFAULT_M, DEFAULT_K,
)

OUT_DIR = Path('.')

IID_THRESHOLD = 0.19880386046614967
PW_THRESHOLD = 0.21272109878307105


def _self_consistency_check():
    rng = np.random.default_rng(RNG_SEED)
    n_trials = 8000
    vals = np.array([per_step_sym_kl(rng.standard_normal(102), rng.standard_normal(102))
                      for _ in range(n_trials)])
    m102_mean = float(vals.mean())
    print(f"[self-check] m=102 bias-floor empirical mean (8000 trials): {m102_mean:.5f} "
          f"(results/table_2_bias_floor.csv: 0.02007)")

    rng2 = np.random.default_rng(RNG_SEED)
    n_trials2 = 800
    m, k = DEFAULT_M, DEFAULT_K
    rho = 0.85
    cds = np.empty(n_trials2)
    for t in range(n_trials2):
        n = m * (k + 1)
        eps = rng2.standard_normal(n)
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = rho * x[i - 1] + math.sqrt(max(1 - rho ** 2, 1e-9)) * eps[i]
        cds[t] = cumulative_divergence(x, m, k, branch='iid')
    rho085_mean = float(cds.mean())
    print(f"[self-check] rho=0.85 AR(1) sweep i.i.d.-branch mean (800 trials): {rho085_mean:.3f} "
          f"(results/table_3_ar1_sweep.csv: 3.488)")
    print()


def table_2_with_se(out_path: Path) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for n, n_trials in [(13, 8000), (21, 8000), (30, 8000), (51, 8000), (102, 8000),
                         (256, 50000), (1000, 50000)]:
        vals = np.array([per_step_sym_kl(rng.standard_normal(n), rng.standard_normal(n))
                          for _ in range(n_trials)])
        bn = bias_floor(n)
        emp_mean = float(vals.mean())
        sd = float(vals.std(ddof=1))
        se = sd / math.sqrt(n_trials)
        rel_err = abs(bn - emp_mean) / bn
        rows.append(dict(n=n, bn=round(bn, 5), empirical_mean=round(emp_mean, 5),
                          se=round(se, 5), ci_lo=round(emp_mean - 1.96 * se, 5),
                          ci_hi=round(emp_mean + 1.96 * se, 5),
                          relative_error_pct=round(rel_err * 100, 2), n_trials=n_trials))
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def appendix_c4_rho_bias(out_path: Path, m: int = 102, n_trials: int = 20000,
                          rhos=(0.0, 0.2, 0.4, 0.55, 0.7, 0.85, 0.95)) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for rho in rhos:
        ests = np.empty(n_trials)
        for t in range(n_trials):
            eps = rng.standard_normal(m)
            x = np.zeros(m)
            for i in range(1, m):
                x[i] = rho * x[i - 1] + math.sqrt(max(1 - rho ** 2, 1e-9)) * eps[i]
            ests[t] = lag1_rho(x)
        mean_est = float(ests.mean())
        sd_est = float(ests.std(ddof=1))
        rows.append(dict(true_rho=rho, mean_rho_hat=round(mean_est, 4),
                          sd=round(sd_est, 4), bias=round(mean_est - rho, 4)))
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def gen_ar_t_innovations(rng: np.random.Generator, n: int, rho: float, df: float) -> np.ndarray:
    if df == float('inf'):
        eps = rng.standard_normal(n)
    else:
        eps = student_t.rvs(df, size=n, random_state=rng)
        eps = eps * math.sqrt((df - 2) / df) if df > 2 else eps
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + math.sqrt(max(1 - rho ** 2, 1e-9)) * eps[i]
    return x


def appendix_f1_joint_stress(out_path: Path, m: int = DEFAULT_M, k: int = DEFAULT_K,
                              n_trials: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    n = m * (k + 1)
    combos = [(0.0, float('inf')), (0.0, 7), (0.0, 4),
              (0.2, float('inf')), (0.2, 7), (0.2, 4),
              (0.55, float('inf')), (0.55, 7), (0.55, 4)]
    rows = []
    for rho, df in combos:
        fpr_iid = 0
        fpr_pw = 0
        for t in range(n_trials):
            x = gen_ar_t_innovations(rng, n, rho, df)
            cd_iid = cumulative_divergence(x, m, k, branch='iid')
            cd_pw = cumulative_divergence(x, m, k, branch='prewhiten')
            if abs(cd_iid) > IID_THRESHOLD:
                fpr_iid += 1
            if abs(cd_pw) > PW_THRESHOLD:
                fpr_pw += 1
        df_label = 'inf (Gaussian)' if df == float('inf') else df
        rows.append(dict(rho=rho, student_t_df=df_label,
                          iid_branch_fpr_pct=round(100 * fpr_iid / n_trials, 1),
                          prewhiten_branch_fpr_pct=round(100 * fpr_pw / n_trials, 1)))
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, index=False)
    return df_out


def main():
    _self_consistency_check()

    print('[1/3] SE-augmented Table 2')
    print(table_2_with_se(OUT_DIR / 'table_2_bias_floor_with_se.csv').to_string(index=False))

    print('\n[2/3] Appendix C.4 -- lag-1 autocorrelation finite-sample bias')
    print(appendix_c4_rho_bias(OUT_DIR / 'appendix_c4_rho_bias.csv').to_string(index=False))

    print('\n[3/3] Appendix F.1 -- joint AR(1) x Student-t stress')
    print(appendix_f1_joint_stress(OUT_DIR / 'appendix_f1_joint_stress.csv').to_string(index=False))


if __name__ == '__main__':
    main()

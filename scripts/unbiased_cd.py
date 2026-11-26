import math
from pathlib import Path
import numpy as np
import pandas as pd
from simulation import RNG_SEED, bias_floor, ar1_prewhiten
OUT_DIR = Path('.')
STEP_N = (6, 8, 13, 30, 102)
STEP_CASES = ((0.0, 1.0), (0.5, 1.0), (0.0, 2.0), (0.5, 0.5), (1.0, 3.0))
STEP_TRIALS = 400000
PATH_K = 15
PATH_M = (6, 8, 13, 19, 30, 102)
PATH_TRIALS = 40000
NULL_N = (13, 21, 30, 51, 102, 256, 1000)
NULL_TRIALS = 50000
AR_RHOS = (0.0, 0.2, 0.4, 0.55, 0.7, 0.85)
AR_M, AR_K, AR_TRIALS = 102, 20, 4000
ACTION_T = 2142
ACTION_K = (2, 5, 10, 15, 20, 40)
ACTION_TRIALS = 4000
CHUNK = 20000


def q_of(n):
    return (n - 3.0) / (n - 1.0)


def step_terms(mo, vo, mn, vn, n):
    plug = 0.5 * np.log(vo / vn) + (vn - vo) / (2.0 * vo) + (mn - mo) ** 2 / (2.0 * vo)
    q = q_of(n)
    r = q * vn / vo
    unb = 0.5 * np.log(vo / vn) + (r - 1.0) / 2.0 + 0.5 * (q * (mn - mo) ** 2 / vo - (1.0 + r) / n)
    return plug, unb


def true_forward(mo, vo, mn, vn):
    return 0.5 * math.log(vo / vn) + (vn - vo) / (2.0 * vo) + (mn - mo) ** 2 / (2.0 * vo)


def cd_unbiased_from_components(sum_a, sum_b, sum_c, n, k):
    q = q_of(n)
    return sum_a + q * (1.0 - 1.0 / n) * sum_b + q * sum_c + k * ((q - 1.0) / 2.0 - (1.0 + q) / (2.0 * n))


def cd_unbiased_symmetric_from_sectors(mean_sector, var_sector, n, k):
    q = q_of(n)
    return q * (1.0 - 1.0 / n) * var_sector + q * mean_sector + k * ((q - 1.0) / 2.0 - (1.0 + q) / (2.0 * n))


def table_step_bias():
    rows = []
    for i, n in enumerate(STEP_N):
        for j, (dmu, r) in enumerate(STEP_CASES):
            rng = np.random.default_rng(RNG_SEED + 1000 + 10 * i + j)
            P, U = [], []
            for _ in range(STEP_TRIALS // CHUNK):
                a = rng.standard_normal((CHUNK, n))
                c = rng.standard_normal((CHUNK, n)) * math.sqrt(r) + dmu
                p, u = step_terms(a.mean(1), a.var(1, ddof=1), c.mean(1), c.var(1, ddof=1), n)
                P.append(p)
                U.append(u)
            P, U = np.concatenate(P), np.concatenate(U)
            t = true_forward(0.0, 1.0, dmu, r)
            fl = P - bias_floor(n)
            rows.append(dict(n=n, delta_mu=dmu, variance_ratio=r, true_kl=t, n_trials=STEP_TRIALS,
                             mean_floor_debiased=float(fl.mean()), bias_floor_debiased=float(fl.mean() - t), sd_floor_debiased=float(fl.std(ddof=1)),
                             mean_unbiased=float(U.mean()), bias_unbiased=float(U.mean() - t), se_unbiased=float(U.std(ddof=1) / math.sqrt(len(U))),
                             z_unbiased=float((U.mean() - t) / (U.std(ddof=1) / math.sqrt(len(U)))), sd_unbiased=float(U.std(ddof=1))))
    return pd.DataFrame(rows)


def path_parameters(k):
    j = np.arange(k + 1)
    return 0.6 * np.sin(np.pi * j / k), 1.0 + 0.5 * j / k


def table_path_window_length():
    mu, sd = path_parameters(PATH_K)
    true = sum(true_forward(mu[i], sd[i] ** 2, mu[i + 1], sd[i + 1] ** 2) for i in range(PATH_K))
    rows = []
    for i, m in enumerate(PATH_M):
        rng = np.random.default_rng(RNG_SEED + 2000 + i)
        F, U = [], []
        for _ in range(PATH_TRIALS // 2000):
            x = rng.standard_normal((2000, PATH_K + 1, m)) * sd[None, :, None] + mu[None, :, None]
            M, V = x.mean(2), x.var(2, ddof=1)
            p, u = step_terms(M[:, :-1], V[:, :-1], M[:, 1:], V[:, 1:], m)
            F.append(p.sum(1) - PATH_K * bias_floor(m))
            U.append(u.sum(1))
        F, U = np.concatenate(F), np.concatenate(U)
        se = U.std(ddof=1) / math.sqrt(len(U))
        rows.append(dict(m=m, k=PATH_K, true_path_kl=true, n_trials=PATH_TRIALS, mean_floor_debiased=float(F.mean()), bias_pct_floor_debiased=float(100 * (F.mean() - true) / true),
                         sd_floor_debiased=float(F.std(ddof=1)), mean_unbiased=float(U.mean()), bias_pct_unbiased=float(100 * (U.mean() - true) / true), se_unbiased=float(se),
                         z_unbiased=float((U.mean() - true) / se), sd_unbiased=float(U.std(ddof=1))))
    return pd.DataFrame(rows)


def table_null():
    rows = []
    for i, n in enumerate(NULL_N):
        rng = np.random.default_rng(RNG_SEED + 3000 + i)
        U = []
        for _ in range(NULL_TRIALS // 10000):
            a = rng.standard_normal((10000, n))
            c = rng.standard_normal((10000, n))
            U.append(step_terms(a.mean(1), a.var(1, ddof=1), c.mean(1), c.var(1, ddof=1), n)[1])
        U = np.concatenate(U)
        se = U.std(ddof=1) / math.sqrt(len(U))
        rows.append(dict(n=n, n_trials=NULL_TRIALS, mean_unbiased=float(U.mean()), se=float(se), z=float(U.mean() / se), sd=float(U.std(ddof=1))))
    return pd.DataFrame(rows)


def ar1_series(rng, n, rho):
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0] / math.sqrt(1.0 - rho ** 2) if rho < 1 else e[0]
    for t in range(1, n):
        x[t] = rho * x[t - 1] + e[t]
    return x


def table_ar_prewhitened():
    rows = []
    for i, rho in enumerate(AR_RHOS):
        rng = np.random.default_rng(RNG_SEED + 4000 + i)
        F, U = [], []
        for _ in range(AR_TRIALS):
            x = ar1_series(rng, (AR_K + 1) * AR_M, rho)
            ws = [ar1_prewhiten(x[j * AR_M:(j + 1) * AR_M]) for j in range(AR_K + 1)]
            n = len(ws[0])
            M = np.array([w.mean() for w in ws])
            V = np.array([w.var(ddof=1) for w in ws])
            p, u = step_terms(M[:-1], V[:-1], M[1:], V[1:], n)
            F.append(p.sum() - AR_K * bias_floor(n))
            U.append(u.sum())
        F, U = np.array(F), np.array(U)
        rows.append(dict(rho=rho, m=AR_M, k=AR_K, n_eff=AR_M - 1, n_trials=AR_TRIALS, mean_floor_debiased=float(F.mean()), se_floor=float(F.std(ddof=1) / math.sqrt(len(F))),
                         mean_unbiased=float(U.mean()), se_unbiased=float(U.std(ddof=1) / math.sqrt(len(U)))))
    return pd.DataFrame(rows)


def table_action():
    tt = np.linspace(0.0, 1.0, 200001)
    mu_f = lambda t: 0.6 * np.sin(np.pi * t)
    sd_f = lambda t: 1.0 + 0.5 * t
    dmu, dsd = np.gradient(mu_f(tt), tt), np.gradient(sd_f(tt), tt)
    action = 0.5 * float(np.trapezoid(dmu ** 2 / sd_f(tt) ** 2 + 2.0 * dsd ** 2 / sd_f(tt) ** 2, tt))
    rows = []
    for i, k in enumerate(ACTION_K):
        rng = np.random.default_rng(RNG_SEED + 5000 + i)
        m = ACTION_T // (k + 1)
        t = (np.arange((k + 1) * m) + 0.5) / ((k + 1) * m)
        F, U = [], []
        for _ in range(ACTION_TRIALS // 1000):
            x = (rng.standard_normal((1000, (k + 1) * m)) * sd_f(t)[None, :] + mu_f(t)[None, :]).reshape(1000, k + 1, m)
            M, V = x.mean(2), x.var(2, ddof=1)
            p, u = step_terms(M[:, :-1], V[:, :-1], M[:, 1:], V[:, 1:], m)
            F.append(k * (p.sum(1) - k * bias_floor(m)))
            U.append(k * u.sum(1))
        F, U = np.concatenate(F), np.concatenate(U)
        rows.append(dict(k=k, m=m, half_fisher_action=action, n_trials=ACTION_TRIALS, mean_k_cd_floor=float(F.mean()), mean_k_cd_unbiased=float(U.mean()),
                         se_k_cd_unbiased=float(U.std(ddof=1) / math.sqrt(len(U))), sd_k_cd_unbiased=float(U.std(ddof=1))))
    return pd.DataFrame(rows)


def main():
    outs = {'table_u1_step_bias.csv': table_step_bias(), 'table_u2_path_window_length.csv': table_path_window_length(), 'table_u3_null.csv': table_null(),
            'table_u4_ar_prewhitened.csv': table_ar_prewhitened(), 'appendix_u5_fisher_action.csv': table_action()}
    for f, df in outs.items():
        df.to_csv(OUT_DIR / f, index=False)
        print(f'=== {f} ===')
        print(df.to_string(index=False))
        print()


if __name__ == '__main__':
    main()

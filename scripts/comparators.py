import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from simulation import RNG_SEED, DEFAULT_M, DEFAULT_K, DEFAULT_N_TRIALS, BENCHMARK_FAMILIES, gen_null_iid, endpoint_summaries, auc, per_step_sym_kl, ar1_prewhiten, bias_floor
from table_7bc_components import paired_cv_auc
OUT_DIR = Path('.')
CV_SEED_BASE = 9100
SECTOR_N = (8, 13, 21, 51, 102)
SECTOR_K = 15
SECTOR_TRIALS = 100000
SECTOR_CHUNK = 10000
WEAK_SCALES = (0.1, 0.2, 0.3, 0.5)
PARITY_D = (0.1, 0.05, 0.025, 0.0125)


def unbiased_sym(ws):
    n = len(ws[0])
    q = (n - 3.0) / (n - 1.0)
    tot = 0.0
    for a, b in zip(ws[:-1], ws[1:]):
        for o, w in ((a, b), (b, a)):
            vo, vn = o.var(ddof=1), w.var(ddof=1)
            r = q * vn / vo
            tot += 0.5 * (0.5 * math.log(vo / vn) + (r - 1.0) / 2.0 + 0.5 * (q * (w.mean() - o.mean()) ** 2 / vo - (1.0 + r) / n))
    return float(tot)


def windows_of(trial, m, k):
    return [trial[j * m:(j + 1) * m] for j in range(k + 1)]


def forward_kl(a, b):
    va, vb = a.var(ddof=1), b.var(ddof=1)
    return 0.5 * math.log(va / vb) + (vb - va) / (2.0 * va) + (b.mean() - a.mean()) ** 2 / (2.0 * va)


def lrt_pooled(ws):
    allx = np.concatenate(ws)
    v0 = allx.var()
    return 0.5 * sum(len(w) * math.log(v0 / w.var()) for w in ws)


def von_neumann(s):
    s = np.asarray(s, dtype=float)
    return float(np.sum(np.diff(s) ** 2) / np.sum((s - s.mean()) ** 2))


def fisher_mssd(ws):
    mu = np.array([w.mean() for w in ws])
    v = np.array([w.var(ddof=1) for w in ws])
    vbar = 0.5 * (v[1:] + v[:-1])
    return 0.5 * float(np.sum(np.diff(mu) ** 2 / vbar + 2.0 * np.diff(0.5 * np.log(v)) ** 2))


def features(trial, m, k):
    ws = windows_of(trial, m, k)
    pw = [ar1_prewhiten(w) for w in ws]
    sym = [per_step_sym_kl(ws[j], ws[j + 1]) for j in range(k)]
    fwd = [forward_kl(ws[j], ws[j + 1]) for j in range(k)]
    rev = [forward_kl(ws[j + 1], ws[j]) for j in range(k)]
    sym_pw = sum(per_step_sym_kl(pw[j], pw[j + 1]) for j in range(k))
    raw = float(sum(sym))
    lrt = lrt_pooled(ws)
    A = sum(0.5 * math.log(ws[j].var(ddof=1) / ws[j + 1].var(ddof=1)) for j in range(k))
    B = sum((ws[j + 1].var(ddof=1) - ws[j].var(ddof=1)) / (2.0 * ws[j].var(ddof=1)) for j in range(k))
    C = sum((ws[j + 1].mean() - ws[j].mean()) ** 2 / (2.0 * ws[j].var(ddof=1)) for j in range(k))
    return dict(cd_iid=raw - k * bias_floor(m), cd_pw=sym_pw - k * bias_floor(m - 1), cd_unbiased=unbiased_sym(ws),
                anova_f=float(stats.f_oneway(*ws).statistic), bartlett=float(stats.bartlett(*ws).statistic),
                anova_f_pw=float(stats.f_oneway(*pw).statistic), bartlett_pw=float(stats.bartlett(*pw).statistic),
                lrt_pooled=lrt, lrt_pooled_pw=lrt_pooled(pw), fisher_mssd=fisher_mssd(ws),
                roughness_ratio=raw / (lrt / m), cd_even=raw, cd_odd=0.5 * float(sum(f - r for f, r in zip(fwd, rev))),
                comp_A=A, comp_B=B, comp_C=C,
                vn_mean_low=-von_neumann([w.mean() for w in ws]), vn_logvar_low=-von_neumann([math.log(w.var(ddof=1)) for w in ws]),
                slope_mean=float(np.polyfit(np.arange(k + 1), [w.mean() for w in ws], 1)[0]), slope_logvar=float(np.polyfit(np.arange(k + 1), [math.log(w.var(ddof=1)) for w in ws], 1)[0]))


def family_draws(gen_fn, m, k, n):
    rp = np.random.default_rng(RNG_SEED + 71)
    rn = np.random.default_rng(RNG_SEED + 72)
    P, Q, EP, EN = [], [], [], []
    for _ in range(n):
        tp = gen_fn(rp, m, k)
        tn = gen_null_iid(rn, m, k)
        P.append(features(tp, m, k))
        Q.append(features(tn, m, k))
        EP.append(endpoint_summaries(tp, m, k))
        EN.append(endpoint_summaries(tn, m, k))
    return pd.DataFrame(P), pd.DataFrame(Q), pd.DataFrame(EP), pd.DataFrame(EN)


def table_comparators(m, k, n):
    cols = ['cd_iid', 'cd_unbiased', 'cd_pw', 'anova_f', 'bartlett', 'anova_f_pw', 'bartlett_pw', 'lrt_pooled', 'lrt_pooled_pw', 'vn_mean_low', 'vn_logvar_low', 'fisher_mssd', 'roughness_ratio']
    rows_a, rows_i, rows_p = [], [], []
    for idx, (name, desc, gen_fn) in enumerate(BENCHMARK_FAMILIES):
        P, Q, EP, EN = family_draws(gen_fn, m, k, n)
        r = dict(family=name, description=desc)
        for c in cols:
            r['auc_' + c] = auc(P[c].to_numpy(), Q[c].to_numpy())
        both = pd.concat([P, Q])
        r['spearman_cd_fisher_mssd'] = float(stats.spearmanr(both.cd_iid, both.fisher_mssd).statistic)
        r['n_trials_per_class'] = n
        rows_a.append(r)
        y = np.concatenate([np.ones(n), np.zeros(n)])
        X0 = np.column_stack([pd.concat([EP, EN])[c].to_numpy() for c in ('delta_mean', 'delta_sd', 'delta_rms', 'delta_samp_en')] + [both[c].to_numpy() for c in ('anova_f', 'bartlett', 'vn_mean_low', 'vn_logvar_low')])
        X1 = np.column_stack([X0] + [both[c].to_numpy() for c in ('comp_A', 'comp_B', 'comp_C', 'cd_iid', 'roughness_ratio')])
        ok = np.all(np.isfinite(X1), axis=1)
        b0, b1 = paired_cv_auc(X0[ok], X1[ok], y[ok], seed=CV_SEED_BASE + idx)
        d = b1 - b0
        rows_i.append(dict(family=name, description=desc, auc_endpoints_anova_bartlett_vn=float(b0.mean()), auc_plus_ABC_CD=float(b1.mean()),
                           delta_mean=float(d.mean()), delta_ci_lo=float(np.percentile(d, 2.5)), delta_ci_hi=float(np.percentile(d, 97.5)), n_trials_per_class=n))
        rows_p.append(dict(family=name, description=desc, mean_even=float(P.cd_even.mean()), mean_odd=float(P.cd_odd.mean()), sd_odd=float(P.cd_odd.std(ddof=1)),
                           null_mean_odd=float(Q.cd_odd.mean()), null_sd_odd=float(Q.cd_odd.std(ddof=1)),
                           auc_odd_signed=auc(P.cd_odd.to_numpy(), Q.cd_odd.to_numpy()), auc_odd_abs=auc(P.cd_odd.abs().to_numpy(), Q.cd_odd.abs().to_numpy()), n_trials_per_class=n))
    return pd.DataFrame(rows_a), pd.DataFrame(rows_i), pd.DataFrame(rows_p)


def table_order(m, k, n):
    half = (k + 1) // 2 + 1
    rest = k + 1 - half + 1
    v_sig = np.concatenate([np.linspace(1.0, 1.8, half), np.linspace(1.8, 1.0, rest)[1:]])[:k + 1]
    contrasts = [('v_variance', v_sig, np.zeros(k + 1)),
                 ('step_mean', np.ones(k + 1), np.where(np.arange(k + 1) < (k + 1) // 2, 0.0, 0.7)),
                 ('oscillatory', 1.0 + 0.4 * np.sin(2 * math.pi * np.arange(k + 1) / (k + 1)), np.zeros(k + 1)),
                 ('mean_ramp', np.ones(k + 1), np.linspace(0.0, 1.0, k + 1))]
    cols = ['cd_iid', 'anova_f', 'bartlett', 'lrt_pooled', 'vn_mean_low', 'vn_logvar_low', 'roughness_ratio']
    rows = []
    for name, sig, mu in contrasts:
        rp = np.random.default_rng(RNG_SEED + 81)
        rn = np.random.default_rng(RNG_SEED + 82)
        P, Q = [], []
        for _ in range(n):
            perm = rp.permutation(k + 1)
            P.append(features(np.concatenate([rp.standard_normal(m) * sig[perm][j] + mu[perm][j] for j in range(k + 1)]), m, k))
            Q.append(features(np.concatenate([rn.standard_normal(m) * sig[j] + mu[j] for j in range(k + 1)]), m, k))
        P, Q = pd.DataFrame(P), pd.DataFrame(Q)
        r = dict(contrast=f'{name}: shuffled vs ordered sub-windows')
        for c in cols:
            r['auc_' + c] = auc(P[c].to_numpy(), Q[c].to_numpy())
        r['median_roughness_ordered'] = float(Q.roughness_ratio.median())
        r['median_roughness_shuffled'] = float(P.roughness_ratio.median())
        r['n_trials_per_class'] = n
        rows.append(r)
    return pd.DataFrame(rows)


def table_weak_effects(m, k, n):
    half = (k + 1) // 2 + 1
    rest = k + 1 - half + 1
    j = np.arange(k + 1)
    shapes = {'mean_ramp': lambda s: (np.ones(k + 1), np.linspace(0.0, 1.0 * s, k + 1)),
              'var_ramp': lambda s: (np.linspace(1.0, 1.0 + 0.8 * s, k + 1), np.zeros(k + 1)),
              'v_variance': lambda s: (np.concatenate([np.linspace(1.0, 1.0 + 0.8 * s, half), np.linspace(1.0 + 0.8 * s, 1.0, rest)[1:]])[:k + 1], np.zeros(k + 1)),
              'step_mean': lambda s: (np.ones(k + 1), np.where(j < (k + 1) // 2, 0.0, 0.7 * s)),
              'oscillatory': lambda s: (1.0 + 0.4 * s * np.sin(2 * math.pi * j / (k + 1)), np.zeros(k + 1))}
    cols = ['cd_iid', 'cd_unbiased', 'anova_f', 'bartlett', 'lrt_pooled', 'vn_mean_low', 'vn_logvar_low', 'fisher_mssd']
    rows = []
    for name, shape in shapes.items():
        for s in WEAK_SCALES:
            sig, mu = shape(s)
            rp = np.random.default_rng(RNG_SEED + 101)
            rn = np.random.default_rng(RNG_SEED + 102)
            P = pd.DataFrame([features(np.concatenate([rp.standard_normal(m) * sig[i] + mu[i] for i in range(k + 1)]), m, k) for _ in range(n)])
            Q = pd.DataFrame([features(gen_null_iid(rn, m, k), m, k) for _ in range(n)])
            r = dict(family=name, scale=s)
            for c in cols:
                r['auc_' + c] = auc(P[c].to_numpy(), Q[c].to_numpy())
            r['auc_roughness_ratio_low'] = auc(-P.roughness_ratio.to_numpy(), -Q.roughness_ratio.to_numpy())
            r['auc_abs_slope_mean'] = auc(P.slope_mean.abs().to_numpy(), Q.slope_mean.abs().to_numpy())
            r['auc_abs_slope_logvar'] = auc(P.slope_logvar.abs().to_numpy(), Q.slope_logvar.abs().to_numpy())
            both = pd.concat([P, Q])
            y = np.concatenate([np.ones(n), np.zeros(n)])
            ext = ['comp_A', 'comp_B', 'comp_C', 'cd_iid', 'roughness_ratio']
            for tag, base in (('dispersion', ['anova_f', 'bartlett', 'lrt_pooled']), ('dispersion_vn', ['anova_f', 'bartlett', 'lrt_pooled', 'vn_mean_low', 'vn_logvar_low']),
                              ('full_classical', ['anova_f', 'bartlett', 'lrt_pooled', 'vn_mean_low', 'vn_logvar_low', 'slope_mean', 'slope_logvar'])):
                b0, b1 = paired_cv_auc(both[base].to_numpy(), both[base + ext].to_numpy(), y, seed=CV_SEED_BASE + 50 + len(rows))
                d = b1 - b0
                r.update({f'auc_cv_{tag}': float(b0.mean()), f'auc_cv_{tag}_plus_cd': float(b1.mean()), f'delta_{tag}': float(d.mean()), f'delta_{tag}_ci_lo': float(np.percentile(d, 2.5)), f'delta_{tag}_ci_hi': float(np.percentile(d, 97.5))})
            r['n_trials_per_class'] = n
            rows.append(r)
    return pd.DataFrame(rows)


def table_sector_floors():
    rows = []
    for i, n in enumerate(SECTOR_N):
        rng = np.random.default_rng(RNG_SEED + 91 + i)
        Cs, ABs = [], []
        for _ in range(SECTOR_TRIALS // SECTOR_CHUNK):
            x = rng.standard_normal((SECTOR_CHUNK, SECTOR_K + 1, n))
            mu = x.mean(axis=2)
            v = x.var(axis=2, ddof=1)
            Cs.append(((mu[:, 1:] - mu[:, :-1]) ** 2 / (2.0 * v[:, :-1])).sum(axis=1))
            ABs.append((0.5 * np.log(v[:, :-1] / v[:, 1:]) + (v[:, 1:] - v[:, :-1]) / (2.0 * v[:, :-1])).sum(axis=1))
        C, AB = np.concatenate(Cs), np.concatenate(ABs)
        fc, fv = SECTOR_K * (n - 1) / (n * (n - 3)), SECTOR_K / (n - 3)
        se = lambda a: float(a.std(ddof=1) / math.sqrt(len(a)))
        rows.append(dict(n=n, k=SECTOR_K, n_trials=SECTOR_TRIALS, floor_mean_sector=fc, mean_sum_C=float(C.mean()), se_sum_C=se(C), debiased_C=float(C.mean() - fc),
                         floor_variance_sector=fv, mean_sum_AB=float(AB.mean()), se_sum_AB=se(AB), debiased_AB=float(AB.mean() - fv),
                         floor_total=SECTOR_K * bias_floor(n), mean_total=float((C + AB).mean())))
    return pd.DataFrame(rows)


def table_parity_orders():
    rows = []
    for d in PARITY_D:
        r = 1.0 + d
        kf = 0.5 * (r - 1.0 - math.log(r))
        kr = 0.5 * (1.0 / r - 1.0 + math.log(r))
        rows.append(dict(d=d, forward=kf, reverse=kr, even=0.5 * (kf + kr), odd=0.5 * (kf - kr), even_over_d2=0.5 * (kf + kr) / d ** 2, odd_over_d3=0.5 * (kf - kr) / d ** 3))
    return pd.DataFrame(rows)


def main():
    m, k, n = DEFAULT_M, DEFAULT_K, DEFAULT_N_TRIALS
    a, i, p = table_comparators(m, k, n)
    outs = {'table_7x_classical_comparators.csv': a, 'table_9x_incremental_over_classical.csv': i, 'appendix_g_parity_by_family.csv': p,
            'table_7y_order_sensitivity.csv': table_order(m, k, n), 'table_7z_weak_effects.csv': table_weak_effects(m, k, n), 'appendix_g_sector_floors.csv': table_sector_floors(), 'appendix_g_parity_orders.csv': table_parity_orders()}
    for f, df in outs.items():
        df.to_csv(OUT_DIR / f, index=False)
        print(f'=== {f} ===')
        print(df.to_string(index=False))
        print()


if __name__ == '__main__':
    main()

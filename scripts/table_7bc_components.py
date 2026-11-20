import math
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from simulation import RNG_SEED, DEFAULT_M, DEFAULT_K, DEFAULT_N_TRIALS, BENCHMARK_FAMILIES, gen_null_iid, endpoint_summaries, auc
OUT_DIR = Path('.')
N_SPLITS = 5
N_REPEATS = 20
CV_SEED_BASE = 9000

def forward_kl_components(a: np.ndarray, b: np.ndarray):
    s2_a, s2_b = (a.var(ddof=1), b.var(ddof=1))
    if s2_a <= 0 or s2_b <= 0:
        return (float('nan'), float('nan'), float('nan'))
    mu_a, mu_b = (a.mean(), b.mean())
    A = 0.5 * math.log(s2_a / s2_b)
    B = (s2_b - s2_a) / (2.0 * s2_a)
    C = (mu_b - mu_a) ** 2 / (2.0 * s2_a)
    return (A, B, C)

def cumulative_components(trial: np.ndarray, m: int, k: int):
    windows = [trial[j * m:(j + 1) * m] for j in range(k + 1)]
    A_sum = B_sum = C_sum = 0.0
    for j in range(k):
        A, B, C = forward_kl_components(windows[j], windows[j + 1])
        A_sum += A
        B_sum += B
        C_sum += C
    return (A_sum, B_sum, C_sum)

def fit_auc(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
    p = clf.predict_proba(sc.transform(Xte))[:, 1]
    return roc_auc_score(yte, p)

def paired_cv_auc(X_base, X_ext, y, n_splits=N_SPLITS, n_repeats=N_REPEATS, seed=0):
    rng = np.random.default_rng(seed)
    base_out, ext_out = ([], [])
    for rep in range(n_repeats):
        fold_seed = int(rng.integers(0, 2 ** 31 - 1))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=fold_seed)
        base_folds, ext_folds = ([], [])
        for tr_idx, te_idx in skf.split(X_base, y):
            base_folds.append(fit_auc(X_base[tr_idx], y[tr_idx], X_base[te_idx], y[te_idx]))
            ext_folds.append(fit_auc(X_ext[tr_idx], y[tr_idx], X_ext[te_idx], y[te_idx]))
        base_out.append(float(np.mean(base_folds)))
        ext_out.append(float(np.mean(ext_folds)))
    return (np.array(base_out), np.array(ext_out))

def main():
    m, k, n_trials = (DEFAULT_M, DEFAULT_K, DEFAULT_N_TRIALS)
    rows_b, rows_c = ([], [])
    for fam_idx, (fam_name, fam_desc, gen_fn) in enumerate(BENCHMARK_FAMILIES):
        rng_pos = np.random.default_rng(RNG_SEED + 71)
        rng_neg = np.random.default_rng(RNG_SEED + 72)
        A_pos = np.empty(n_trials)
        A_neg = np.empty(n_trials)
        B_pos = np.empty(n_trials)
        B_neg = np.empty(n_trials)
        C_pos = np.empty(n_trials)
        C_neg = np.empty(n_trials)
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
            A_pos[t], B_pos[t], C_pos[t] = cumulative_components(tp, m, k)
            A_neg[t], B_neg[t], C_neg[t] = cumulative_components(tn, m, k)
            ep_p, ep_n = (endpoint_summaries(tp, m, k), endpoint_summaries(tn, m, k))
            dm_pos[t], dm_neg[t] = (ep_p['delta_mean'], ep_n['delta_mean'])
            dsd_pos[t], dsd_neg[t] = (ep_p['delta_sd'], ep_n['delta_sd'])
            drms_pos[t], drms_neg[t] = (ep_p['delta_rms'], ep_n['delta_rms'])
            dse_pos[t], dse_neg[t] = (ep_p['delta_samp_en'], ep_n['delta_samp_en'])
        rows_b.append({'family': fam_name, 'description': fam_desc, 'auc_A_trial': auc(np.abs(A_pos), np.abs(A_neg)), 'auc_B_trial': auc(np.abs(B_pos), np.abs(B_neg)), 'auc_C_trial': auc(C_pos, C_neg), 'n_trials_per_class': n_trials})
        X_base = np.column_stack([np.concatenate([dm_pos, dm_neg]), np.concatenate([dsd_pos, dsd_neg]), np.concatenate([drms_pos, drms_neg]), np.concatenate([dse_pos, dse_neg])])
        X_ext = np.column_stack([X_base, np.concatenate([A_pos, A_neg]), np.concatenate([B_pos, B_neg]), np.concatenate([C_pos, C_neg])])
        y = np.concatenate([np.ones(n_trials), np.zeros(n_trials)])
        finite = np.all(np.isfinite(X_ext), axis=1)
        X_base, X_ext, y_f = (X_base[finite], X_ext[finite], y[finite])
        base_dist, ext_dist = paired_cv_auc(X_base, X_ext, y_f, seed=CV_SEED_BASE + fam_idx)
        delta = ext_dist - base_dist
        rows_c.append({'family': fam_name, 'description': fam_desc, 'auc_endpoints_only': float(np.mean(base_dist)), 'auc_endpoints_plus_ABC': float(np.mean(ext_dist)), 'delta_mean': float(np.mean(delta)), 'delta_ci_lo': float(np.percentile(delta, 2.5)), 'delta_ci_hi': float(np.percentile(delta, 97.5)), 'n_trials_per_class': n_trials})
    df_b = pd.DataFrame(rows_b)
    df_c = pd.DataFrame(rows_c)
    df_b.to_csv(OUT_DIR / 'table_7b_components.csv', index=False)
    df_c.to_csv(OUT_DIR / 'table_7c_incremental_auc.csv', index=False)
    print('=== Table 7B: standalone component AUC ===')
    print(df_b.to_string(index=False))
    print()
    print('=== Table 7C: incremental AUC (endpoints alone vs +A,B,C) ===')
    print(df_c.to_string(index=False))
if __name__ == '__main__':
    main()

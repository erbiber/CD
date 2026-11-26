import argparse
import glob
import os
import numpy as np
import pandas as pd
import realdata_cd as R
KEYS = ['dataset', 'subject', 'reference', 'rate', 'window', 'trial', 'code', 'condition']


def q_of(n):
    return (n - 3.0) / (n - 1.0)


def cd_unbiased_forward(sum_a, sum_b, sum_c, n, k):
    q = q_of(n)
    return sum_a + q * (1.0 - 1.0 / n) * sum_b + q * sum_c + k * ((q - 1.0) / 2.0 - (1.0 + q) / (2.0 * n))


def cd_unbiased_symmetric(mean_sector, var_sector, n, k):
    q = q_of(n)
    return q * (1.0 - 1.0 / n) * var_sector + q * mean_sector + k * ((q - 1.0) / 2.0 - (1.0 + q) / (2.0 * n))


def add_unbiased(per):
    per = per.copy()
    n = per.n_eff.astype(float)
    ok = (per.finite == 1) & (n > 3) & (per.n_steps_used > 0)
    per['cd_unbiased'] = np.where(ok, cd_unbiased_forward(per.comp_A, per.comp_B, per.comp_C, n, per.n_steps_used), np.nan)
    per['unbiased_finite'] = np.isfinite(per.cd_unbiased).astype(int)
    if 'pr_cd_hat' in per.columns:
        npr = per.pr_n_eff.astype(float)
        okp = (per.pr_finite == 1) & (npr > 3) & (per.pr_n_steps_used > 0)
        per['pr_cd_unbiased'] = np.where(okp, cd_unbiased_symmetric(per.pr_mean_sector, per.pr_var_sector, npr, per.pr_n_steps_used), np.nan)
    return per


def describe(per):
    cols = [c for c in ('cd_hat', 'cd_unbiased', 'pr_cd_hat', 'pr_cd_unbiased') if c in per.columns]
    out = []
    for key, w in per.groupby(['dataset', 'reference', 'rate', 'window'], sort=False):
        r = dict(zip(['dataset', 'reference', 'rate', 'window'], key), n_rows=len(w), median_n_eff=w.n_eff.median(), median_k=w.n_steps_used.median())
        for c in cols:
            x = w[c].dropna()
            r.update({f'{c}_n': len(x), f'{c}_median': x.median(), f'{c}_q25': x.quantile(0.25), f'{c}_q75': x.quantile(0.75), f'{c}_frac_negative': float((x < 0).mean()) if len(x) else np.nan})
        out.append(r)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description='Exactly unbiased (minimum-variance unbiased) cumulative divergence computed from the per-trial component sums written by realdata_cd.py.')
    ap.add_argument('--dataset', required=True, choices=sorted(R.REFS))
    ap.add_argument('--out', default=R.RESULTS_DIR)
    ap.add_argument('--no-ica', action='store_true')
    a = ap.parse_args()
    stem = f'cd_realdata_{a.dataset}' + ('_noica' if a.no_ica else '')
    files = [os.path.join(a.out, f'{stem}_pertrial_{r}.csv.gz') for r in R.REFS[a.dataset]]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        raise SystemExit(f'no per-trial files for {stem} in {a.out}')
    per = add_unbiased(pd.concat([pd.read_csv(f, float_precision='round_trip') for f in files], ignore_index=True))
    keep = KEYS + ['n_eff', 'n_steps_used', 'cd_hat', 'cd_unbiased'] + (['pr_n_eff', 'pr_n_steps_used', 'pr_cd_hat', 'pr_cd_unbiased'] if 'pr_cd_unbiased' in per.columns else [])
    f = os.path.join(a.out, f'{stem}_unbiased_pertrial.csv.gz')
    per[keep].to_csv(f, index=False, float_format='%.17g', compression={'method': 'gzip', 'mtime': 0})
    print(f'wrote {os.path.basename(f)} ({len(per)} rows)')
    d = describe(per)
    f = os.path.join(a.out, f'{stem}_unbiased_summary.csv')
    d.to_csv(f, index=False)
    print(f'wrote {os.path.basename(f)} ({len(d)} rows)')
    cols = ['cd_unbiased'] + (['pr_cd_unbiased'] if 'pr_cd_unbiased' in per.columns else [])
    s = R.subject_auc(per, cols)
    f = os.path.join(a.out, f'{stem}_unbiased_auc_subject.csv')
    s.to_csv(f, index=False)
    print(f'wrote {os.path.basename(f)} ({len(s)} rows)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

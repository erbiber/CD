import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
WINDOWS = ['baseline', 'early_post', 'mid_post', 'late_post']
MEAS = ['cd_hat', 'comp_A', 'comp_B', 'comp_C', 'delta_mean', 'delta_sd', 'delta_rms']
FILES = {('ERP CORE', 'native'): 'rho_erpcore_native_pertrial.csv', ('ERP CORE', 'decim128'): 'rho_erpcore_decim128_pertrial.csv', ('ds006018', 'native'): 'rho_ds006018_native_pertrial.csv', ('ds006018', 'decim128'): 'rho_ds006018_decim128_pertrial.csv'}

def folded_auc(pos, neg):
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if len(pos) < 3 or len(neg) < 3:
        return np.nan
    U, _ = mannwhitneyu(pos, neg, alternative='two-sided')
    a = U / (len(pos) * len(neg))
    return round(max(a, 1 - a), 3)

def boundary():
    rows = []
    for (ds, rate), f in FILES.items():
        d = pd.read_csv(f)
        d = d[d.finite == 1]
        for w in WINDOWS:
            s = d[d.window == w]
            rows.append(dict(dataset=ds, rate=rate, window=w, n=len(s), rho_median=round(s.rho_med.median(), 3), rho_iqr_lo=round(s.rho_med.quantile(0.25), 3), rho_iqr_hi=round(s.rho_med.quantile(0.75), 3), rho_p95=round(s.rho_med.quantile(0.95), 3), frac_le_0p5=round((s.rho_med <= 0.5).mean(), 3), prewhiten_rate=round((s.rho_med > 0.2).mean(), 3)))
    return pd.DataFrame(rows)

def components():
    rows = []
    for (ds, rate), f in FILES.items():
        d = pd.read_csv(f)
        d = d[d.finite == 1]
        for w in WINDOWS:
            s = d[d.window == w]
            rows.append(dict(dataset=ds, rate=rate, window=w, median_abs_A=round(s.comp_A.abs().median(), 3), median_abs_B=round(s.comp_B.abs().median(), 3), median_C=round(s.comp_C.median(), 3), median_cd_hat=round(s.cd_hat.median(), 3)))
    return pd.DataFrame(rows)

def discrimination():
    rows = []
    for (ds, rate), f in FILES.items():
        if ds != 'ERP CORE':
            continue
        d = pd.read_csv(f)
        for w in WINDOWS:
            s = d[d.window == w]
            t = s[s.condition == 'Target']
            st = s[s.condition == 'Standard']
            r = dict(rate=rate, window=w, n_target=len(t), n_standard=len(st))
            for m in MEAS:
                r[f'auc_{m}'] = folded_auc(t[m].values, st[m].values)
            rows.append(r)
    return pd.DataFrame(rows)

def main():
    boundary().to_csv('table_realdata_rho_boundary.csv', index=False)
    components().to_csv('table_realdata_components.csv', index=False)
    discrimination().to_csv('table_realdata_discrimination.csv', index=False)
    print('Wrote table_realdata_rho_boundary.csv, table_realdata_components.csv, table_realdata_discrimination.csv')
if __name__ == '__main__':
    main()

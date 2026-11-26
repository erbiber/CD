import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
CLASSICAL = ['anova_f', 'bartlett', 'lrt_pooled', 'vn_mean', 'vn_logvar', 'delta_mean', 'delta_sd']
CDSET = ['comp_A', 'comp_B', 'comp_C', 'cd_hat']
SEED = 20261124


def cv_auc(X, y, seed):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, p)


def main(path, ref, out):
    d = pd.read_csv(path, float_precision='round_trip')
    d = d[(d.reference == ref) & (d.finite == 1)]
    rows, cors = [], []
    for (rate, win), w in d.groupby(['rate', 'window'], sort=False):
        w = w.dropna(subset=CLASSICAL + CDSET)
        rho = spearmanr(w.cd_hat, w.lrt_pooled).statistic
        res = []
        for s, h in w.groupby('subject', sort=True):
            y = (h.condition == 'Target').astype(int).to_numpy()
            if y.sum() < 10 or (1 - y).sum() < 10:
                continue
            a0 = cv_auc(h[CLASSICAL].to_numpy(float), y, SEED)
            a1 = cv_auc(h[CLASSICAL + CDSET].to_numpy(float), y, SEED)
            res.append((a0, a1))
        r = np.array(res)
        dlt = r[:, 1] - r[:, 0]
        rows.append(dict(reference=ref, rate=rate, window=win, n_subjects=len(r), spearman_cd_vs_lrt=rho,
                         mean_auc_classical=r[:, 0].mean(), mean_auc_classical_plus_cd=r[:, 1].mean(), mean_delta=dlt.mean(),
                         p_wilcoxon_delta=float(wilcoxon(dlt).pvalue) if np.any(dlt != 0) else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(df.round(4).to_string(index=False))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])

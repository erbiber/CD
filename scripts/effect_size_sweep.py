from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd

from simulation import (
    cumulative_divergence, endpoint_summaries, auc, RNG_SEED,
    DEFAULT_M, DEFAULT_K, DEFAULT_N_TRIALS,
)

OUT_DIR = Path('.')


def gen_mean_ramp_shift(rng: np.random.Generator, m: int, k: int, shift: float) -> np.ndarray:
    means = np.linspace(0.0, shift, k + 1)
    return np.concatenate([rng.standard_normal(m) + means[j] for j in range(k + 1)])


def gen_v_variance_peak(rng: np.random.Generator, m: int, k: int, peak: float) -> np.ndarray:
    half = (k + 1) // 2 + 1
    rest = k + 1 - half + 1
    up = np.linspace(1.0, peak, half)
    down = np.linspace(peak, 1.0, rest)[1:]
    sigmas = np.concatenate([up, down])[:k + 1]
    return np.concatenate([rng.standard_normal(m) * sigmas[j] for j in range(k + 1)])


def _best_endpoint_auc(pos_summaries: list[dict], neg_summaries: list[dict]) -> float:
    best = 0.5
    for key in ('delta_mean', 'delta_sd', 'delta_rms', 'delta_samp_en'):
        pos = np.array([s[key] for s in pos_summaries])
        neg = np.array([s[key] for s in neg_summaries])
        a = auc(pos, neg)
        if np.isfinite(a) and abs(a - 0.5) > abs(best - 0.5):
            best = a
    return best


def effect_size_sweep(out_path: Path, m: int = DEFAULT_M, k: int = DEFAULT_K,
                       n_trials: int = DEFAULT_N_TRIALS
                       ) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows = []

    mean_shifts = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    for shift in mean_shifts:
        cd_pos, cd_neg, ep_pos, ep_neg = [], [], [], []
        for _ in range(n_trials):
            tr_pos = gen_mean_ramp_shift(rng, m, k, shift)
            tr_neg = gen_mean_ramp_shift(rng, m, k, 0.0)
            cd_pos.append(cumulative_divergence(tr_pos, m, k, branch='iid'))
            cd_neg.append(cumulative_divergence(tr_neg, m, k, branch='iid'))
            ep_pos.append(endpoint_summaries(tr_pos, m, k))
            ep_neg.append(endpoint_summaries(tr_neg, m, k))
        cd_auc = auc(np.array(cd_pos), np.array(cd_neg))
        best_ep = _best_endpoint_auc(ep_pos, ep_neg)
        rows.append(dict(family='mean_ramp', magnitude=shift,
                          cd_auc=round(cd_auc, 3), best_endpoint_auc=round(best_ep, 3)))

    peaks = [1.0, 1.25, 1.5, 1.8, 2.0, 2.25, 2.5, 2.75, 3.0]
    for peak in peaks:
        cd_pos, cd_neg, ep_pos, ep_neg = [], [], [], []
        for _ in range(n_trials):
            tr_pos = gen_v_variance_peak(rng, m, k, peak)
            tr_neg = gen_v_variance_peak(rng, m, k, 1.0)
            cd_pos.append(cumulative_divergence(tr_pos, m, k, branch='iid'))
            cd_neg.append(cumulative_divergence(tr_neg, m, k, branch='iid'))
            ep_pos.append(endpoint_summaries(tr_pos, m, k))
            ep_neg.append(endpoint_summaries(tr_neg, m, k))
        cd_auc = auc(np.array(cd_pos), np.array(cd_neg))
        best_ep = _best_endpoint_auc(ep_pos, ep_neg)
        rows.append(dict(family='v_variance', magnitude=peak,
                          cd_auc=round(cd_auc, 3), best_endpoint_auc=round(best_ep, 3)))

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def main():
    df = effect_size_sweep(OUT_DIR / 'appendix_f2_effect_size.csv')
    print(df.to_string(index=False))
    print()
    mr = df[df.family == 'mean_ramp']
    vv = df[df.family == 'v_variance']
    print(f"mean_ramp: CD AUC at shift=0 -> {mr.iloc[0].cd_auc}, at shift=2.0 -> {mr.iloc[-1].cd_auc} "
          f"(manuscript: 0.53 -> 0.73)")
    print(f"           best endpoint reaches >=0.99 by shift={mr[mr.best_endpoint_auc >= 0.99].magnitude.min() if (mr.best_endpoint_auc >= 0.99).any() else 'not reached'} "
          f"(manuscript: saturates near 1.0 by shift 0.75)")
    print(f"v_variance: CD AUC at peak=1.0 -> {vv.iloc[0].cd_auc}, at peak=3.0 -> {vv.iloc[-1].cd_auc} "
          f"(manuscript: 0.51 -> 0.91)")
    print(f"            best endpoint range: [{vv.best_endpoint_auc.min()}, {vv.best_endpoint_auc.max()}] "
          f"(manuscript: approx 0.49-0.51, i.e. near chance throughout)")


if __name__ == '__main__':
    main()

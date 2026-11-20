from pathlib import Path
import numpy as np
import pandas as pd
from simulation import RNG_SEED, cumulative_divergence
N_TOTAL = 2142
K_VALUES = [1, 2, 5, 10, 20]
N_TRIALS = 2000

def main(out_path=Path('appendix_e2_k_dependence.csv')):
    rows = []
    for k in K_VALUES:
        rng = np.random.default_rng(RNG_SEED + 81 + k)
        m = N_TOTAL // (k + 1)
        n = m * (k + 1)
        ramp = np.linspace(0.0, 1.0, n)
        vals = np.array([cumulative_divergence(rng.standard_normal(n) + ramp, m, k, 'iid') for _ in range(N_TRIALS)])
        rows.append({'k': k, 'm': m, 'n_trials': N_TRIALS, 'mean_debiased_CD': float(vals.mean()), 'se': float(vals.std(ddof=1) / np.sqrt(N_TRIALS))})
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))

if __name__ == '__main__':
    main()

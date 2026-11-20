import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'results')
REF = sys.argv[2] if len(sys.argv) > 2 else 'distributed'
DF = pd.concat([pd.read_csv(os.path.join(RES, f'cd_realdata_{d}_rho.csv')) for d in ('erpcore', 'ds006018')], ignore_index=True)
DF = DF[DF.reference == REF]
WINDOWS = ['baseline', 'early_post', 'mid_post', 'late_post']
WLAB = ['Baseline\n(−200–0 ms)', 'Early\n(100–250 ms)', 'Mid\n(200–300 ms)', 'Late\n(300–600 ms)']
C_NAT, C_DEC = ('#D55E00', '#0072B2')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8})
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
panels = [('erpcore', 'native', 'decim128', 'ERP CORE P3 (1024 Hz)'), ('ds006018', 'native', 'decim128', 'ds006018 (500 Hz)')]
for ax, (ds, rnat, rdec, title) in zip(axes, panels):
    x = np.arange(len(WINDOWS))
    for off, rate, col, lab in [(-0.16, rnat, C_NAT, 'native rate'), (0.16, rdec, C_DEC, 'decimated 128 Hz')]:
        sub = DF[(DF.dataset == ds) & (DF.rate == rate)].set_index('window')
        med = [sub.loc[w, 'median_rho'] for w in WINDOWS]
        lo = [sub.loc[w, 'median_rho'] - sub.loc[w, 'iqr_low'] for w in WINDOWS]
        hi = [sub.loc[w, 'iqr_high'] - sub.loc[w, 'median_rho'] for w in WINDOWS]
        ax.errorbar(x + off, med, yerr=[lo, hi], fmt='o', color=col, capsize=3, markersize=6, linewidth=1.4, label=lab)
    ax.axhspan(-0.4, 0.5, color='#2ca02c', alpha=0.06, zorder=0)
    ax.axhline(0.5, color='#2ca02c', lw=1.0, ls='--', zorder=1)
    ax.axhline(0.85, color='#888888', lw=1.0, ls=':', zorder=1)
    ax.text(3.45, 0.5, ' reliable\n (ρ≤0.5)', color='#2ca02c', fontsize=7.5, va='center', ha='left')
    ax.text(3.45, 0.85, ' empirical-\n null (ρ>0.85)', color='#888888', fontsize=7.5, va='center', ha='left')
    ax.set_xticks(x)
    ax.set_xticklabels(WLAB, fontsize=8.3)
    ax.set_title(title, fontsize=10.5)
    ax.set_ylim(-0.4, 1.02)
    ax.grid(True, axis='y', lw=0.4, alpha=0.3)
    if ax is axes[0]:
        ax.set_ylabel('per-trial median $\\hat{\\rho}$ (point = median, bars = IQR)')
    ax.legend(loc='lower left', fontsize=8.2, frameon=False)
fig.suptitle('First-differenced single-trial lag-1 autocorrelation at Fz, native rate and 128 Hz', fontsize=10.2, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RES, os.pardir, 'figures', 'figure_6_rho_distributions.png'), dpi=300, bbox_inches='tight', facecolor='white')
print('wrote figure_6_rho_distributions.png')

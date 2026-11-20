from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
DATA_DIR = Path('.')
OUT_DIR = Path('.')
DPI = 300
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'lines.linewidth': 1.5, 'lines.markersize': 5})
C_DATA = '#0072B2'
C_THEORY = '#D55E00'
C_IID = '#CC79A7'
C_BART = '#009E73'
C_PW = '#0072B2'
C_NULL = '#999999'
C_POS = '#0072B2'
C_TARGET = '#D55E00'

def figure_1_pipeline():
    from matplotlib.gridspec import GridSpec
    rng = np.random.default_rng(7)
    n_sub = 6
    m = 30
    mus = np.array([0.0, 0.25, 0.55, 1.05, 1.35, 1.7])
    sds = np.array([0.4, 0.5, 0.62, 0.8, 0.92, 1.05])
    segs = [rng.normal(mu, sd, m) for mu, sd in zip(mus, sds)]
    sig = np.concatenate(segs)
    t = np.arange(len(sig))
    bounds = [j * m for j in range(n_sub + 1)]
    fig = plt.figure(figsize=(8.4, 6.3))
    gs = GridSpec(2, 1, height_ratios=[1.02, 1.0], hspace=0.5)
    axA = fig.add_subplot(gs[0])
    ylo, yhi = (-2.6, 7.0)
    axA.set_xlim(0, len(sig))
    axA.set_ylim(ylo, yhi)
    axA.plot(t, sig, color='#94A3B2', linewidth=0.9, zorder=1)
    for j in range(n_sub):
        x0, x1 = (bounds[j], bounds[j + 1])
        if j % 2 == 0:
            axA.axvspan(x0, x1, color='#EEF2F6', zorder=0)
        if j < n_sub - 1:
            axA.axvline(x1, color='#CBD5DF', linewidth=0.8, zorder=1)
        axA.add_patch(plt.Rectangle((x0 + 1, mus[j] - sds[j]), x1 - x0 - 2, 2 * sds[j], facecolor=C_DATA, alpha=0.13, edgecolor='none', zorder=2))
        axA.hlines(mus[j], x0 + 1, x1 - 1, color=C_DATA, linewidth=2.2, zorder=4)
        axA.text((x0 + x1) / 2, ylo + 0.18, f'$W_{{{j + 1}}}$\n$(\\mu_{{{j + 1}}},\\sigma_{{{j + 1}}})$', ha='center', va='bottom', fontsize=8.2, color='#37474F', linespacing=1.25)
    yk = 5.9
    for j in range(n_sub - 1):
        c0 = (bounds[j] + bounds[j + 1]) / 2
        c1 = (bounds[j + 1] + bounds[j + 2]) / 2
        axA.annotate('', xy=(c1, yk), xytext=(c0, yk), arrowprops=dict(arrowstyle='-', color='#D55E00', linewidth=1.1, connectionstyle='arc3,rad=-0.35'), zorder=5)
        axA.text((c0 + c1) / 2, yk + 0.5, f'$D_{{{j + 1}}}$', ha='center', va='center', fontsize=8.6, color='#D55E00')
    axA.set_xlabel('samples within the single-trial analysis window', fontsize=9)
    axA.set_yticks([])
    axA.set_title('(a)  What CD measures: per-step Gaussian KL between consecutive sub-windows', fontsize=10.5, loc='left')
    for s in ('top', 'right', 'left'):
        axA.spines[s].set_visible(False)
    axB = fig.add_subplot(gs[1])
    axB.axis('off')
    axB.set_xlim(0, 100)
    axB.set_ylim(0, 100)

    def box(cx, w, label, y=58, h=26, fc='#F4F6F9', ec='#37474F', fs=8.4):
        axB.add_patch(FancyBboxPatch((cx - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.4,rounding_size=1.4', linewidth=1.0, edgecolor=ec, facecolor=fc, zorder=3))
        axB.text(cx, y, label, ha='center', va='center', fontsize=fs, linespacing=1.45, zorder=4)
    centres = [9.5, 30.5, 52.5, 73.5, 92.5]
    widths = [16, 19, 19, 20, 13]
    labels = ['Single-trial\nwindow\n$x=(x_1,\\dots,x_n)$', 'Adaptive split into\n$k{+}1$ sub-windows\n$k_{\\mathrm{eff}}=\\min(15,\\lfloor n/8\\rfloor-1)$', 'Per-step Gaussian KL\n$D_j$ between\n$W_j$ and $W_{j+1}$', 'Subtract bias floor\n$b(m_{\\mathrm{eff}})=\\frac{2m-1}{m(m-3)}$', 'Sum over\n$k$ steps']
    for cx, w, lab in zip(centres, widths, labels):
        box(cx, w, lab)
    edges = [(9.5 + 8, 30.5 - 9.5), (30.5 + 9.5, 52.5 - 9.5), (52.5 + 9.5, 73.5 - 10), (73.5 + 10, 92.5 - 6.5)]
    for x0, x1 in edges:
        axB.add_patch(FancyArrowPatch((x0, 58), (x1, 58), arrowstyle='-|>', mutation_scale=13, linewidth=1.0, color='#555', zorder=2))
    axB.add_patch(FancyArrowPatch((92.5, 45), (92.5, 33), arrowstyle='-|>', mutation_scale=13, linewidth=1.0, color='#555', zorder=2))
    axB.text(92.5, 28, '$\\widehat{CD}$', ha='center', va='center', fontsize=12, fontweight='bold', color=C_DATA)
    box(41.5, 22, 'AR(1) pre-whitening\n(applied iff $\\hat\\rho_{\\mathrm{med}}>0.20$)', y=86, h=15, fc='#FFF4E8', ec='#D55E00', fs=7.8)
    axB.add_patch(FancyArrowPatch((41.5, 78.5), (49, 71.5), arrowstyle='-|>', mutation_scale=11, linewidth=0.9, color='#D55E00', linestyle=(0, (4, 2)), zorder=2))
    axB.text(46, 20, '$\\widehat{CD}\\;=\\;\\sum_{j=1}^{k}\\left[\\,D_j-b(m_{\\mathrm{eff}})\\,\\right]$', ha='center', va='center', fontsize=11.5, color='#222')
    axB.text(46, 1, '§ 2.1 (pipeline) · § 2.2 (bias floor, Eq.\u20094) · § 2.5 (pre-whitening) · Eq.\u2009(1)', ha='center', va='center', fontsize=7.8, style='italic', color='#777')
    axB.set_title('(b)  How CD is computed', fontsize=10.5, loc='left', y=0.99)
    out = OUT_DIR / 'figure_1_pipeline.png'
    plt.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  -> {out}')

def figure_2_bias_floor():
    df = pd.read_csv(DATA_DIR / 'table_2_bias_floor.csv')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.0))
    ax1.plot(df['n'], df['closed_form_b'], '-', color=C_THEORY, linewidth=2, label='closed form $b(n_h)$', zorder=2)
    ax1.errorbar(df['n'], df['empirical_mean'], yerr=df['empirical_sd'] / np.sqrt(df['n_trials']), fmt='o', color=C_DATA, capsize=3, label='empirical mean (±SE)', zorder=3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('sub-window length $n_h$ (log)')
    ax1.set_ylabel('$\\widehat{CD}_{raw}$ under $H_0$ (log)')
    ax1.set_title('(a) Bias floor magnitude')
    ax1.legend(loc='upper right', frameon=False)
    ax1.grid(True, which='both', linewidth=0.4, alpha=0.3)
    ax2.plot(df['n'], df['rel_error_pct'], 'o-', color=C_DATA, markersize=6)
    ax2.axhline(3.0, color='#888888', linewidth=0.7, linestyle='--', label='3 % (P1a threshold)')
    ax2.axhline(1.0, color='#888888', linewidth=0.7, linestyle=':', label='1 % (P1b threshold, n_h≥256)')
    ax2.set_xscale('log')
    ax2.set_xlabel('sub-window length $n_h$ (log)')
    ax2.set_ylabel('relative error (%)')
    ax2.set_title('(b) Closed-form agreement')
    ax2.legend(loc='upper right', frameon=False)
    ax2.set_ylim(0, max(4, df['rel_error_pct'].max() * 1.15))
    ax2.grid(True, linewidth=0.4, alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / 'figure_2_bias_floor.png'
    plt.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  -> {out}')

def figure_3_ar1_sweep():
    df = pd.read_csv(DATA_DIR / 'table_3_ar1_sweep.csv').sort_values('rho')
    se_iid = df['iid_sd'] / np.sqrt(df['n_trials'])
    se_bart = df['bartlett_sd'] / np.sqrt(df['n_trials'])
    se_pw = df['prewhiten_sd'] / np.sqrt(df['n_trials'])
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.errorbar(df['rho'], df['iid_mean'], yerr=se_iid, fmt='o-', color=C_IID, capsize=3, label='iid (no correction)')
    ax.errorbar(df['rho'], df['bartlett_mean'], yerr=se_bart, fmt='s-', color=C_BART, capsize=3, label='Bartlett effective-$n$')
    ax.errorbar(df['rho'], df['prewhiten_mean'], yerr=se_pw, fmt='^-', color=C_PW, capsize=3, label='AR(1) pre-whitening')
    ax.axhline(0.0, color='#333333', linewidth=0.6, linestyle='-')
    ax.axhspan(-0.5, 0.5, color='#999999', alpha=0.1, label='P2a tolerance band (|mean| < 0.5)')
    ax.axvline(0.2, color='#888888', linewidth=0.7, linestyle='--', alpha=0.7, label='adaptive threshold $\\hat\\rho = 0.20$')
    ax.set_xlabel('true AR(1) coefficient $\\rho$')
    ax.set_ylabel('mean debiased $\\widehat{CD}$ on null trials')
    ax.set_title('AR(1) sweep across three correction strategies (path-integral $k = 20$, $m = 102$)')
    ax.legend(loc='upper left', frameon=False, ncol=1)
    ax.grid(True, linewidth=0.4, alpha=0.3)
    ax.set_xlim(-0.02, 1.0)
    plt.tight_layout()
    out = OUT_DIR / 'figure_3_ar1_sweep.png'
    plt.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  -> {out}')

def figure_4_kurtosis():
    df = pd.read_csv(DATA_DIR / 'table_4_kurtosis.csv').sort_values('df')
    se = df['debiased_cd_sd'] / np.sqrt(df['n_trials'])
    if df['empirical_kurtosis_excess'].isna().any():
        df['excess_kurtosis_theory'] = df['df'].apply(lambda v: 6.0 / (v - 4) if v > 4 else float('inf'))
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    finite_df = df[df['df'] < 100000.0]
    ax.errorbar(finite_df['df'], finite_df['debiased_cd_mean'], yerr=se[finite_df.index], fmt='o-', color=C_DATA, capsize=3, label='mean debiased CD (±SE)')
    gaussian_row = df[df['df'] >= 100000.0]
    if len(gaussian_row) > 0:
        gauss_mean = gaussian_row['debiased_cd_mean'].iloc[0]
        ax.axhline(gauss_mean, color=C_NULL, linewidth=0.7, linestyle='--', label=f'Gaussian limit (df → ∞): {gauss_mean:+.4f}')
    ax.axhline(0.0, color='#333333', linewidth=0.6, linestyle='-')
    ax.axvline(7.0, color='#888888', linewidth=0.7, linestyle=':', alpha=0.7, label='operating limit ν = 7 (excess kurtosis ≈ 1)')
    ax.set_xscale('log')
    ax.set_xlabel('Student-t degrees of freedom $\\nu$ (log)')
    ax.set_ylabel('mean debiased $\\widehat{CD}$ on null trials')
    ax.set_title('Kurtosis stress: Gaussian-based bias floor under heavy tails (path-integral $k = 20$, $m = 102$)')
    ax.legend(loc='upper right', frameon=False)
    ax.grid(True, which='both', linewidth=0.4, alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / 'figure_4_kurtosis.png'
    plt.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  -> {out}')

def figure_5_benchmark():
    df = pd.read_csv(DATA_DIR / 'table_7_benchmark.csv')
    NULL_FAMS = ['null_ar085', 'null_t4']
    REAL_FAMS = ['mean_ramp', 'var_ramp', 'v_variance', 'step_mean', 'oscillatory', 'mixed']
    ordered = [f for f in NULL_FAMS + REAL_FAMS if f in df['family'].values]
    df = df.set_index('family').loc[ordered].reset_index()
    methods = [('CD', 'auc_cd', C_DATA), ('Δ mean', 'auc_delta_mean', '#E69F00'), ('Δ SD', 'auc_delta_sd', '#56B4E9'), ('Δ RMS', 'auc_delta_rms', '#009E73'), ('Δ sample-entropy', 'auc_delta_samp_en', '#CC79A7')]
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    y = np.arange(len(df))
    bar_h = 0.16
    for i, (name, col, color) in enumerate(methods):
        offset = (i - 2) * bar_h
        ax.barh(y + offset, df[col], bar_h, color=color, label=name, edgecolor='white', linewidth=0.4)
    ax.axvline(0.5, color='#333333', linewidth=0.8, linestyle='-', label='AUC = 0.5 (chance)')
    ax.axvline(1.0, color='#888888', linewidth=0.5, linestyle='--', alpha=0.6)
    if len(NULL_FAMS) > 0 and len(REAL_FAMS) > 0:
        ax.axhline(len(NULL_FAMS) - 0.5, color='#666666', linewidth=0.7, linestyle='--')
        ax.text(-0.3, (len(NULL_FAMS) - 1) / 2, 'null-stress\n(want ≈ 0.5)', fontsize=8, color='#666666', style='italic', ha='center', va='center', rotation=90, transform=ax.get_yaxis_transform())
        ax.text(-0.3, len(NULL_FAMS) + (len(REAL_FAMS) - 1) / 2, 'real-change\n(want high)', fontsize=8, color='#666666', style='italic', ha='center', va='center', rotation=90, transform=ax.get_yaxis_transform())
    ax.set_yticks(y)
    ax.set_yticklabels(df['family'], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Discrimination AUC (vs. i.i.d. Gaussian null)')
    ax.set_title('Eight-family benchmark: CD vs. four endpoint summaries (n = 600 trials per class)')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8.5)
    ax.grid(True, axis='x', linewidth=0.4, alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / 'figure_5_benchmark.png'
    plt.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  -> {out}')

def main():
    print('CD figures 1-5')
    print(f'  data: {DATA_DIR}')
    print(f'  out : {OUT_DIR.resolve()}')
    print()
    for name, fn in [('Figure 1 (Pipeline schematic)', figure_1_pipeline), ('Figure 2 (Bias floor)', figure_2_bias_floor), ('Figure 3 (AR(1) sweep)', figure_3_ar1_sweep), ('Figure 4 (Kurtosis stress)', figure_4_kurtosis), ('Figure 5 (Benchmark)', figure_5_benchmark)]:
        print(f'{name}')
        try:
            fn()
        except FileNotFoundError as e:
            print(f'  SKIP: {e}')
        except Exception as e:
            print(f'  FAIL: {type(e).__name__}: {e}')
            raise
    print('\nDone.')
if __name__ == '__main__':
    main()

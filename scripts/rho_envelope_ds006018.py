from __future__ import annotations
import os, warnings, math
import numpy as np
import pandas as pd
from scipy.signal import resample as _sp_resample
CACHE_DIR = 'C:\\Users\\erkan\\Documents\\dof_validation\\data_ds006018'
OUT_DIR = os.environ.get('CD_OUT_DIR', os.path.dirname(os.path.abspath(__file__)))
TASK_FILTER = 'visualoddball'
MIN_WIN_SAMPLES = 8
N_STEPS_FIXED = 15
RHO_THRESHOLD = 0.2
FILTER_LOW, FILTER_HIGH, FILTER_DESIGN = (0.1, 30.0, 'firwin')
EPOCH_TMIN, EPOCH_TMAX = (-0.2, 0.8)
BASELINE = (-0.2, 0.0)
REJECT_THRESHOLD = 0.0001
DO_ICA = False
DECIM_SFREQ = 128.0
ICA_N_COMPONENTS, ICA_HIGHPASS_FOR_FIT = (0.99, 1.0)
ICA_METHOD, ICA_RANDOM_STATE = ('fastica', 42)
CD_CHANNEL = 'Fz'
CD_FALLBACK = ['FC1', 'FC2', 'F3', 'F4']
TARGET_CODES = [10001, 10007, 10013, 10019, 10025]
TRIAL_CODES = TARGET_CODES
EOG_CHANNELS = ['HEL', 'HER', 'VER']
MASTOID_CHANNELS = ['LM', 'RM']
MIN_TRIALS_REQUIRED = 10
WINDOWS = [('baseline', -0.2, 0.0), ('early_post', 0.1, 0.25), ('mid_post', 0.2, 0.3), ('late_post', 0.3, 0.6)]
AR1_SWEEP_MAX = 0.95
warnings.filterwarnings('ignore')
import mne
mne.set_log_level('WARNING')

def _bias_b(n: float) -> float:
    if n <= 3:
        return 0.0
    return (n - 1) / (2.0 * (n - 3)) - 0.5 + (n - 1) / (n * (n - 3))

def _ar1_prewhiten(x: np.ndarray):
    if len(x) < 4:
        return (x[1:], 0.0)
    xm = x - x.mean()
    denom = float(np.sum(xm[:-1] ** 2) + 1e-12)
    rho = float(np.sum(xm[:-1] * xm[1:]) / denom)
    rho = float(np.clip(rho, -0.999, 0.999))
    return (x[1:] - rho * x[:-1], rho)

def compute_cd(seg: np.ndarray, n_steps: int=N_STEPS_FIXED, rho_threshold: float=RHO_THRESHOLD) -> dict:
    seg = np.asarray(seg, dtype=float)
    _sd = float(np.std(seg))
    if _sd > 0.0:
        seg = seg / _sd
    T = len(seg)
    while n_steps > 2 and T < (n_steps + 1) * MIN_WIN_SAMPLES:
        n_steps -= 1
    n_win = T // (n_steps + 1)
    if n_win < 3:
        return dict(cd_raw=np.nan, cd_hat=np.nan, rho_med=0.0, n_win=n_win, n_steps=n_steps, n_steps_used=0, comp_A=np.nan, comp_B=np.nan, comp_C=np.nan, first_win=None, last_win=None)
    windows = [seg[k * n_win:(k + 1) * n_win] for k in range(n_steps + 1)]
    rho_hats = [_ar1_prewhiten(w)[1] if len(w) >= 4 else 0.0 for w in windows]
    rho_med = float(np.median(rho_hats))
    use_pw = rho_med > rho_threshold
    cd_raw = bias = sum_A = sum_B = sum_C = 0.0
    n_used = 0
    for t in range(n_steps):
        wo, wn = (windows[t], windows[t + 1])
        if use_pw:
            wo, _ = _ar1_prewhiten(wo)
            wn, _ = _ar1_prewhiten(wn)
            n_eff = len(wo)
        else:
            n_eff = n_win
        if n_eff < 3:
            continue
        mu_o = float(np.mean(wo))
        vo = max(float(np.var(wo, ddof=1)), 1e-12)
        mu_n = float(np.mean(wn))
        vn = max(float(np.var(wn, ddof=1)), 1e-12)
        A = 0.5 * np.log(vo / vn)
        B = (vn - vo) / (2.0 * vo)
        C = (mu_n - mu_o) ** 2 / (2.0 * vo)
        cd_raw += A + B + C
        bias += _bias_b(n_eff)
        sum_A += A
        sum_B += B
        sum_C += C
        n_used += 1
    return dict(cd_raw=float(cd_raw), cd_hat=float(cd_raw - bias), rho_med=rho_med, n_win=int(n_win), n_steps=int(n_steps), n_steps_used=int(n_used), comp_A=float(sum_A), comp_B=float(sum_B), comp_C=float(sum_C), first_win=windows[0], last_win=windows[n_steps])

def _auto_n_steps(n_smp: int) -> int:
    return min(N_STEPS_FIXED, max(2, n_smp // MIN_WIN_SAMPLES - 1))

def _sample_entropy(x: np.ndarray, m_emb: int=2, r: float | None=None) -> float:
    if x.size < m_emb + 2:
        return float('nan')
    if r is None:
        r = 0.2 * x.std(ddof=1)
    if r <= 0:
        return float('nan')
    n = x.size

    def _phi(mm: int) -> float:
        templates = np.array([x[i:i + mm] for i in range(n - mm + 1)])
        if templates.size == 0:
            return 0.0
        cnt, total = (0, 0)
        for i in range(templates.shape[0]):
            diffs = np.max(np.abs(templates - templates[i]), axis=1)
            cnt += int(np.sum((diffs < r) & (np.arange(templates.shape[0]) != i)))
            total += templates.shape[0] - 1
        return cnt / max(total, 1)
    A = _phi(m_emb + 1)
    B = _phi(m_emb)
    if A <= 0 or B <= 0:
        return float('nan')
    return -math.log(A / B)

def endpoint_summaries_real(first: np.ndarray, last: np.ndarray) -> dict:
    if first is None or last is None or len(first) < 3 or (len(last) < 3):
        return dict(delta_mean=np.nan, delta_sd=np.nan, delta_rms=np.nan, delta_samp_en=np.nan)
    return dict(delta_mean=float(abs(last.mean() - first.mean())), delta_sd=float(abs(last.std(ddof=1) - first.std(ddof=1))), delta_rms=float(abs(math.sqrt((last ** 2).mean()) - math.sqrt((first ** 2).mean()))), delta_samp_en=float(abs(_sample_entropy(last) - _sample_entropy(first))))

def _decimate_segment(seg: np.ndarray, native_sfreq: float, target_sfreq):
    if target_sfreq is None or native_sfreq <= target_sfreq:
        return seg
    n_new = max(2, int(round(len(seg) * target_sfreq / native_sfreq)))
    return _sp_resample(seg, n_new)

def get_channel_index(ch_names, primary, fallbacks):
    lower = [c.lower() for c in ch_names]
    if primary.lower() in lower:
        idx = lower.index(primary.lower())
        return (ch_names[idx], idx)
    for fb in fallbacks:
        if fb.lower() in lower:
            idx = lower.index(fb.lower())
            return (ch_names[idx], idx)
    return (None, None)

def load_visualoddball_recordings():
    from eegdash.dataset import DS006018
    print(f'Loading ds006018 via eegdash (cache: {CACHE_DIR}) ...')
    dataset = DS006018(cache_dir=CACHE_DIR)
    recs = []
    for rec in dataset.datasets:
        try:
            desc = rec.description
            task = desc.get('task', None) if hasattr(desc, 'get') else desc['task']
            subj = desc.get('subject', None) if hasattr(desc, 'get') else desc['subject']
        except Exception:
            task, subj = (None, None)
        if task == TASK_FILTER:
            recs.append((str(subj), rec))
    print(f'  visualoddball recordings: {len(recs)}')
    return recs

def preprocess_recording(rec):
    raw = rec.raw
    if raw is None:
        return None
    try:
        raw.load_data()
    except Exception:
        pass
    type_map = {}
    for ch in EOG_CHANNELS:
        if ch in raw.ch_names:
            type_map[ch] = 'eog'
    for ch in MASTOID_CHANNELS:
        if ch in raw.ch_names:
            type_map[ch] = 'misc'
    if type_map:
        try:
            raw.set_channel_types(type_map)
        except Exception:
            pass
    raw.filter(FILTER_LOW, FILTER_HIGH, fir_design=FILTER_DESIGN, verbose=False)
    if DO_ICA:
        try:
            raw_ica = raw.copy().filter(ICA_HIGHPASS_FOR_FIT, None, fir_design=FILTER_DESIGN, verbose=False)
            ica = mne.preprocessing.ICA(n_components=ICA_N_COMPONENTS, method=ICA_METHOD, random_state=ICA_RANDOM_STATE, max_iter='auto')
            ica.fit(raw_ica, verbose=False)
            try:
                eog_idx, _ = ica.find_bads_eog(raw, verbose=False)
                ica.exclude = eog_idx
            except Exception:
                ica.exclude = []
            raw = ica.apply(raw, verbose=False)
        except Exception:
            pass
    return raw

def process_recording(sub_label, rec):
    raw = preprocess_recording(rec)
    if raw is None:
        return (None, 'no_raw')
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    trial_event_id = {k: v for k, v in event_id.items() if v in TRIAL_CODES and v in events[:, 2]}
    if not trial_event_id:
        return (None, 'no_trial_events')
    epochs = mne.Epochs(raw, events, event_id=trial_event_id, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX, baseline=BASELINE, reject=dict(eeg=REJECT_THRESHOLD), preload=True, verbose=False)
    if len(epochs) < MIN_TRIALS_REQUIRED:
        return (None, f'too_few_{len(epochs)}')
    fz_name, fz_idx = get_channel_index(epochs.ch_names, CD_CHANNEL, CD_FALLBACK)
    if fz_name is None:
        return (None, 'missing_fz')
    times = epochs.times
    sfreq = float(epochs.info['sfreq'])
    eff_sfreq = float(DECIM_SFREQ) if DECIM_SFREQ is not None and sfreq > DECIM_SFREQ else sfreq
    data = epochs.get_data()
    rows = []
    for i in range(len(epochs)):
        for label, t0, t1 in WINDOWS:
            m = (times >= t0) & (times < t1)
            seg = data[i, fz_idx, m]
            seg = _decimate_segment(seg, sfreq, DECIM_SFREQ)
            if len(seg) < 6:
                rows.append(dict(subject=sub_label, trial_idx=i, condition='Target', sfreq=eff_sfreq, fz_used=fz_name, window=label, rho_med=np.nan, cd_hat=np.nan, n_win=np.nan, n_steps_used=np.nan, finite=0, comp_A=np.nan, comp_B=np.nan, comp_C=np.nan, delta_mean=np.nan, delta_sd=np.nan, delta_rms=np.nan, delta_samp_en=np.nan))
                continue
            seg_fd = np.diff(seg)
            res = compute_cd(seg_fd, n_steps=_auto_n_steps(len(seg_fd)))
            ep = endpoint_summaries_real(res['first_win'], res['last_win'])
            rows.append(dict(subject=sub_label, trial_idx=i, condition='Target', sfreq=eff_sfreq, fz_used=fz_name, window=label, rho_med=res['rho_med'], cd_hat=res['cd_hat'], n_win=res['n_win'], n_steps_used=res['n_steps_used'], finite=int(np.isfinite(res['cd_hat'])), comp_A=res['comp_A'], comp_B=res['comp_B'], comp_C=res['comp_C'], delta_mean=ep['delta_mean'], delta_sd=ep['delta_sd'], delta_rms=ep['delta_rms'], delta_samp_en=ep['delta_samp_en']))
    return (pd.DataFrame(rows), None)

def summarise(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for label, _, _ in WINDOWS:
        w = df[df.window == label]
        rho = w['rho_med'].dropna()
        out.append(dict(window=label, n_trials=int(len(w)), median_rho=round(float(rho.median()), 4), iqr_low=round(float(rho.quantile(0.25)), 4), iqr_high=round(float(rho.quantile(0.75)), 4), pct95=round(float(rho.quantile(0.95)), 4), maximum=round(float(rho.max()), 4), prewhiten_rate=round(float((rho > RHO_THRESHOLD).mean()), 4), pct_in_range=round(float(((rho >= 0) & (rho <= AR1_SWEEP_MAX)).mean()), 6)))
    return pd.DataFrame(out)

def _self_test_scale_invariance(verbose: bool=True) -> None:
    rng = np.random.default_rng(20260623)

    def _ar1(n, rho):
        e = rng.normal(0.0, 1.0, n)
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = rho * x[i - 1] + e[i]
        return x
    keys = ('rho_med', 'comp_A', 'comp_B', 'comp_C', 'cd_hat')
    n_checked = 0
    for length, rho in [(160, 0.6), (120, 0.3), (90, -0.2), (200, 0.8), (60, 0.0)]:
        seg_fd = np.diff(np.cumsum(_ar1(length, rho)))
        ns = _auto_n_steps(len(seg_fd))
        unit = compute_cd(seg_fd, n_steps=ns)
        volt = compute_cd(seg_fd * 1e-06, n_steps=ns)
        for k in keys:
            u, v = (unit[k], volt[k])
            if np.isnan(u) and np.isnan(v):
                continue
            assert abs(u - v) <= 1e-09 + 1e-06 * abs(u), f'scale-invariance FAILED on {k}: unit={u!r} vs volt-scaled={v!r} (len={length}, rho={rho}). Has the normalisation in compute_cd been removed?'
        n_checked += 1
    _ = compute_cd(np.zeros(5), n_steps=2)
    if verbose:
        print(f'[self-test] scale-invariance OK across {n_checked} signals (rho_med/comp_A/comp_B/comp_C/cd_hat identical at unit and 1e-6 scale).')

def main():
    _self_test_scale_invariance()
    recs = load_visualoddball_recordings()
    print(f'ds006018 rho-envelope | MIN_WIN={MIN_WIN_SAMPLES} N_STEPS={N_STEPS_FIXED} rho_thr={RHO_THRESHOLD} | DO_ICA={DO_ICA}')
    frames, kept, skipped = ([], 0, [])
    for sub_label, rec in recs:
        try:
            df, err = process_recording(sub_label, rec)
        except Exception as e:
            skipped.append((sub_label, f'err:{type(e).__name__}'))
            continue
        if df is None:
            skipped.append((sub_label, err))
            continue
        frames.append(df)
        kept += 1
        print(f'  sub-{sub_label}: {df.trial_idx.nunique()} trials  (sfreq={df.sfreq.iloc[0]:.0f})')
    if not frames:
        raise RuntimeError('No recordings processed -- check eegdash cache / task filter.')
    per = pd.concat(frames, ignore_index=True)
    _tag = f'_decim{int(DECIM_SFREQ)}' if DECIM_SFREQ is not None else ''
    per.to_csv(os.path.join(OUT_DIR, f'rho_ds006018{_tag}_pertrial.csv'), index=False)
    summ = summarise(per)
    summ.to_csv(os.path.join(OUT_DIR, f'rho_ds006018{_tag}_summary.csv'), index=False)
    finite_rate = per['finite'].mean()
    in_range = ((per['rho_med'].dropna() >= 0) & (per['rho_med'].dropna() <= AR1_SWEEP_MAX)).mean()
    lines = [f'ds006018  subjects kept = {kept}  trials = {per.trial_idx.count()}  (sfreq = {per.sfreq.iloc[0]:.0f} Hz, DO_ICA={DO_ICA})', f'P3b rho in [0,0.95]           : {in_range:.6f}  ({('PASS' if in_range == 1.0 else 'FAIL')} ==1.0)', f'finite-value rate             : {finite_rate:.4f}', '', 'Per-window summary:', summ.to_string(index=False), '', f'Skipped: {skipped}']
    txt = '\n'.join(lines)
    open(os.path.join(OUT_DIR, f'rho_ds006018{_tag}_p3.txt'), 'w').write(txt)
    print('\n' + txt)
    print(f'\nWrote rho_ds006018{_tag}_pertrial.csv, rho_ds006018{_tag}_summary.csv, rho_ds006018{_tag}_p3.txt')
if __name__ == '__main__':
    import sys
    if '--self-test' in sys.argv:
        _self_test_scale_invariance()
    else:
        main()

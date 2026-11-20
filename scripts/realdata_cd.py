from __future__ import annotations
import argparse
import gc
import glob
import math
import os
import re
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.signal import resample as _sp_resample
warnings.filterwarnings('ignore')
import mne
mne.set_log_level('WARNING')

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
RESULTS_DIR = os.path.join(BASE, 'results', 'P300')
ERPCORE_DIR = os.path.join(BASE, 'data', 'erp_core_P3')
DS006018_DIR = os.path.join(BASE, 'data_ds006018')
DS006018_SUBDIR = 'ds006018'
DS006018_TASK = 'visualoddball'

FILTER_LOW, FILTER_HIGH, FILTER_DESIGN = (0.1, 30.0, 'firwin')
TMIN_EPOCH, TMAX_EPOCH = (-0.2, 0.8)
BASELINE = (-0.2, 0.0)
REJECT_UV = 100.0
DO_ICA = True
ICA_N_COMPONENTS = 20
ICA_HIGHPASS_FOR_FIT = 1.0
ICA_METHOD = 'fastica'
ICA_RANDOM_STATE = 42
TARGET_CODES = [11, 22, 33, 44, 55]
STANDARD_CODES = [12, 13, 14, 15, 21, 23, 24, 25, 31, 32, 34, 35, 41, 42, 43, 45, 51, 52, 53, 54]
MIN_TRIALS_REQUIRED = 10
MIN_TRIALS_PER_CONDITION = 10

MIN_WIN_SAMPLES = 8
N_STEPS_FIXED = 15
RHO_THRESHOLD = 0.2
WINDOWS = [('baseline', -0.2, 0.0), ('early_post', 0.1, 0.25), ('mid_post', 0.2, 0.3), ('late_post', 0.3, 0.6)]
RATES = {'native': None, 'decim128': 128.0}

EOG_CANDIDATES = ['HEOG', 'VEOG', 'HEOG_left', 'HEOG_right', 'VEOG_lower', 'heog', 'veog', 'EOG', 'LHEOG', 'RHEOG', 'LVEOG', 'UVEOG', 'HEL', 'HER', 'VEL', 'VER']
MISC_CANDIDATES = ['LM', 'RM', 'M1', 'M2', 'A1', 'A2']
MASTOID = {'erpcore': ('P9', 'P10'), 'ds006018': ('LM', 'RM')}
NON_SCALP = {'erpcore': (), 'ds006018': ('LM', 'RM')}
ONLINE = {'ds006018': 'Cz'}
REFS = {'erpcore': ('distributed', 'cz', 'mastoid', 'average', 'csd'), 'ds006018': ('distributed', 'mastoid', 'average', 'csd')}
EP = ['delta_mean', 'delta_sd', 'delta_rms', 'delta_samp_en']
AUC_COLS = ['cd_hat', 'comp_A', 'comp_B', 'comp_C', 'delta_mean', 'delta_sd', 'delta_rms', 'delta_samp_en', 'amp_fz', 'amp_pz']
SUBJECT_AUC_COLS = ['cd_hat', 'comp_A', 'comp_B', 'comp_C', 'delta_mean', 'delta_sd', 'delta_rms', 'amp_fz', 'amp_pz']
MIN_TRIALS_SUBJECT_AUC = 5
SETTINGS = ('ERPCORE_DIR', 'DS006018_DIR', 'DS006018_SUBDIR', 'DS006018_TASK', 'REJECT_UV', 'DO_ICA', 'ICA_N_COMPONENTS')


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


def erpcore_subjects():
    pat = os.path.join(ERPCORE_DIR, 'sub-*', 'ses-P3', 'eeg', 'sub-*_ses-P3_task-P3_eeg.set')
    return [os.path.basename(p).split('_')[0].replace('sub-', '') for p in sorted(glob.glob(pat))]


def _ds006018_root():
    p = os.path.join(DS006018_DIR, DS006018_SUBDIR) if DS006018_SUBDIR else DS006018_DIR
    return p if os.path.isdir(p) else DS006018_DIR


def _has_payload(path):
    if not path.endswith('.vhdr'):
        return True
    return any(os.path.exists(path[:-5] + ext) for ext in ('.eeg', '.dat', '.seg'))


def _ds006018_files():
    root = _ds006018_root()
    out = []
    for ext in ('.vhdr', '.set', '.fif', '.edf', '.bdf'):
        out.extend(glob.glob(os.path.join(root, '**', '*' + ext), recursive=True))
    keep = []
    for p in out:
        rel = os.path.relpath(p, root).replace('\\', '/').split('/')
        if any(part in ('derivatives', 'sourcedata', 'stimuli', 'code') for part in rel):
            continue
        if not _has_payload(p):
            continue
        if DS006018_TASK and f'task-{DS006018_TASK}' not in os.path.basename(p):
            continue
        keep.append(p)
    return sorted(keep)


def _subject_of(path):
    for part in path.replace('\\', '/').split('/'):
        if part.startswith('sub-'):
            return part[4:].split('_')[0]
    return None


def ds006018_subjects():
    seen = []
    for p in _ds006018_files():
        s = _subject_of(p)
        if s and s not in seen:
            seen.append(s)
    return seen


def _read_any(path):
    if path.endswith('.vhdr'):
        return mne.io.read_raw_brainvision(path, preload=True, verbose=False)
    if path.endswith('.set'):
        return mne.io.read_raw_eeglab(path, preload=True, verbose=False)
    if path.endswith('.fif'):
        return mne.io.read_raw_fif(path, preload=True, verbose=False)
    if path.endswith('.bdf'):
        return mne.io.read_raw_bdf(path, preload=True, verbose=False)
    return mne.io.read_raw_edf(path, preload=True, verbose=False)


def load_raw(sub, dataset):
    if dataset == 'erpcore':
        p = os.path.join(ERPCORE_DIR, f'sub-{sub}', 'ses-P3', 'eeg', f'sub-{sub}_ses-P3_task-P3_eeg.set')
        if not os.path.exists(p):
            return (None, 'file_not_found')
        return (mne.io.read_raw_eeglab(p, preload=True, verbose=False), None)
    cands = [p for p in _ds006018_files() if _subject_of(p) == sub]
    if not cands:
        return (None, 'file_not_found')
    if len(cands) > 1:
        rels = [os.path.relpath(c, _ds006018_root()).replace('\\', '/') for c in cands]
        return (None, 'ambiguous_%d_recordings: %s' % (len(cands), ' | '.join(rels)))
    try:
        return (_read_any(cands[0]), None)
    except Exception as exc:
        return (None, 'read_failed_%s' % type(exc).__name__)


def prepare_channel_types(raw):
    m = {}
    for ch in raw.ch_names:
        if ch in EOG_CANDIDATES:
            m[ch] = 'eog'
        elif ch in MISC_CANDIDATES:
            m[ch] = 'misc'
    if m:
        raw.set_channel_types(m, verbose=False)
    return raw


def run_ica(raw):
    if not DO_ICA:
        return (raw, 0, 0, None)
    try:
        raw_fit = raw.copy().filter(ICA_HIGHPASS_FOR_FIT, None, fir_design=FILTER_DESIGN, verbose=False)
        ica = mne.preprocessing.ICA(n_components=ICA_N_COMPONENTS, method=ICA_METHOD, random_state=ICA_RANDOM_STATE, max_iter='auto')
        ica.fit(raw_fit, verbose=False)
        n_comp = ica.n_components_
        if len(mne.pick_types(raw.info, eeg=False, eog=True)) == 0:
            print('    WARNING: no channel is typed eog; ICA removes nothing. Add the EOG channel names to EOG_CANDIDATES.')
            ica.exclude = []
        else:
            try:
                eog_idx, _ = ica.find_bads_eog(raw, verbose=False)
                ica.exclude = eog_idx
            except Exception as exc:
                print(f'    WARNING: find_bads_eog failed ({type(exc).__name__}: {exc}); ICA removes nothing.')
                ica.exclude = []
        n_exc = len(ica.exclude)
        del raw_fit
        raw = ica.apply(raw, verbose=False)
        return (raw, n_comp, n_exc, None)
    except Exception as e:
        return (raw, 0, 0, f'ica_failed_{type(e).__name__}')


def clean_continuous(sub, dataset):
    raw, err = load_raw(sub.replace('sub-', ''), dataset)
    if err:
        return (None, err)
    if int(raw.first_samp) != 0:
        return (None, f'first_samp_{raw.first_samp}_nonzero')
    raw = prepare_channel_types(raw)
    mast = [c for c in MASTOID[dataset] if c in raw.ch_names]
    if len(mast) != 2:
        return (None, 'mastoid_channels_missing')
    raw.set_channel_types({c: 'eeg' for c in mast}, verbose=False)
    raw.filter(FILTER_LOW, FILTER_HIGH, fir_design=FILTER_DESIGN, verbose=False)
    raw, n_comp, n_exc, err = run_ica(raw)
    if err:
        return (None, err)
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    code_of = {}
    for desc, val in event_id.items():
        m = re.search('(\\d+)\\s*$', str(desc))
        code_of[int(val)] = int(m.group(1)) if m else None
    eeg = mne.pick_types(raw.info, eeg=True, eog=False, misc=False, exclude=[])
    names = [raw.ch_names[i] for i in eeg]
    V = raw.get_data()[eeg]
    online = ONLINE.get(dataset)
    if online and online not in names:
        V = np.vstack([V, np.zeros((1, V.shape[1]))])
        names = names + [online]
    for c in ('Fz', 'Pz'):
        if c not in names:
            return (None, f'missing_{c}')
    return (dict(V=V, names=names, sfreq=float(raw.info['sfreq']), events=events, code_of=code_of, n=V.shape[1], n_ica=int(n_comp), n_excluded=int(n_exc)), None)


def reference_signals(c, dataset, refs):
    V, names = c['V'], c['names']
    idx = {n: i for i, n in enumerate(names)}
    scalp = [i for i, n in enumerate(names) if n not in NON_SCALP[dataset]]
    out = {}
    for r in refs:
        if r == 'distributed':
            out[r] = np.zeros(c['n'])
        elif r == 'cz':
            if 'Cz' not in idx:
                return (None, 'cz_missing')
            out[r] = V[idx['Cz']].copy()
        elif r == 'mastoid':
            out[r] = V[[idx[m] for m in MASTOID[dataset]]].mean(axis=0)
        elif r == 'average':
            out[r] = V[scalp].mean(axis=0)
    return (out, scalp)


def rolling_ptp_max(V, rows, s, n_window, lo):
    from scipy.ndimage import maximum_filter1d, minimum_filter1d
    n_s = V.shape[1]
    org = -(n_window // 2)
    out = np.zeros(n_s, dtype=np.float64)
    for ch in rows:
        x = V[ch] - s
        mx = maximum_filter1d(x, size=n_window, origin=org, mode='nearest')
        mn = minimum_filter1d(x, size=n_window, origin=org, mode='nearest')
        np.maximum(out, mx - mn, out=out)
    shifted = np.full(n_s, np.inf)
    if lo >= 0:
        shifted[:n_s - lo] = out[lo:]
    else:
        shifted[-lo:] = out[:n_s + lo]
    return shifted


def combined_ptp(c, refs, scalp):
    sf = c['sfreq']
    lo = int(round(TMIN_EPOCH * sf))
    nw = int(round(TMAX_EPOCH * sf)) - lo + 1
    worst = None
    for s in refs.values():
        p = rolling_ptp_max(c['V'], scalp, s, nw, lo)
        worst = p if worst is None else np.maximum(worst, p)
    return worst


def csd_pair(c, chans):
    names = c['names']
    low = {n.lower(): i for i, n in enumerate(names)}
    miss = [x for x in chans if x.lower() not in low]
    if miss:
        return None, 'csd_missing_' + '_'.join(miss)
    ix = [low[x.lower()] for x in chans]
    info = mne.create_info([names[i] for i in ix], c['sfreq'], 'eeg')
    raw = mne.io.RawArray(c['V'][ix], info, verbose=False)
    raw.set_montage(mne.channels.make_standard_montage('standard_1020'), match_case=False, verbose=False)
    out = mne.preprocessing.compute_current_source_density(raw, verbose=False).get_data()
    sub = [names[i].lower() for i in ix]
    return (out[sub.index('fz')], out[sub.index('pz')]), None


def trial_set(c, ptp):
    sf = c['sfreq']
    keep = set(TARGET_CODES + STANDARD_CODES)
    ev = c['events']
    codes = np.array([c['code_of'].get(int(v)) for v in ev[:, 2]], dtype=object)
    sel = np.array([x is not None and x in keep for x in codes])
    samples = ev[sel, 0].astype(np.int64)
    codes = codes[sel]
    lo = int(round(TMIN_EPOCH * sf))
    hi = int(round(TMAX_EPOCH * sf))
    inb = (samples + lo >= 0) & (samples + hi < c['n'])
    samples, codes = samples[inb], codes[inb]
    ok = ptp[samples] <= REJECT_UV * 1e-06
    samples, codes = samples[ok], codes[ok]
    return samples, np.array([int(x) for x in codes]), np.array([int(x in TARGET_CODES) for x in codes])


def epochs(sig, samples, sf):
    lo = int(round(TMIN_EPOCH * sf))
    hi = int(round(TMAX_EPOCH * sf))
    x = sig[samples[:, None] + np.arange(lo, hi + 1)[None, :]]
    times = np.arange(lo, hi + 1) / sf
    bm = (times >= BASELINE[0]) & (times <= BASELINE[1])
    return x - x[:, bm].mean(axis=1, keepdims=True), times


def closed_form_b(n):
    return (2 * n - 1) / (n * (n - 3)) if n > 3 else float('nan')


def cd_rows(seg_native, sf, base):
    rows = []
    for rate, target in RATES.items():
        seg = _decimate_segment(seg_native, sf, target)
        eff = float(target) if target is not None and sf > target else sf
        b0 = dict(base, rate=rate, sfreq=eff)
        if len(seg) < 6:
            rows.append(dict(b0, n_samples=len(seg), finite=0, cd_hat=np.nan, **{k: np.nan for k in EP}))
            continue
        seg_fd = np.diff(seg)
        res = compute_cd(seg_fd, n_steps=_auto_n_steps(len(seg_fd)))
        use_pw = bool(res['rho_med'] > RHO_THRESHOLD)
        n_eff = (res['n_win'] - 1) if use_pw else res['n_win']
        bias_impl = res['cd_raw'] - res['cd_hat'] if np.isfinite(res['cd_hat']) else np.nan
        bias_cf = res['n_steps_used'] * closed_form_b(n_eff) if res['n_steps_used'] else np.nan
        ep = endpoint_summaries_real(res['first_win'], res['last_win'])
        rows.append(dict(b0, n_samples=len(seg_fd), n_win=res['n_win'], n_steps=res['n_steps'], n_steps_used=res['n_steps_used'],
                         prewhitened=int(use_pw), n_eff=n_eff, below_min_m=int(res['n_win'] < MIN_WIN_SAMPLES), rho_med=res['rho_med'],
                         cd_raw=res['cd_raw'], cd_hat=res['cd_hat'], bias_subtracted=bias_impl, bias_closed_form=bias_cf,
                         finite=int(np.isfinite(res['cd_hat'])), comp_A=res['comp_A'], comp_B=res['comp_B'], comp_C=res['comp_C'], **ep))
    return rows


def save_cache(res, cp):
    tmp = cp + '.tmp'
    pd.to_pickle(res, tmp)
    os.replace(tmp, cp)


def subject(args):
    sub, dataset, refs_wanted, cache_dir, state = args
    globals().update(state)
    tag = f"r{int(round(REJECT_UV))}_{'ic%d' % ICA_N_COMPONENTS if DO_ICA else 'noica'}_{'-'.join(refs_wanted)}"
    cp = os.path.join(cache_dir, f'cd_realdata_{dataset}_{tag}_{sub}.pkl')
    res = None
    if os.path.exists(cp):
        try:
            res = pd.read_pickle(cp)
        except Exception:
            res = None
        if res is not None:
            stale = res['rows'] is not None and 'amp_pz' not in res['rows'].columns
            if not (DO_ICA and not res['info'].get('n_ica')) and not stale:
                return res, 'cached', 0.0
        os.remove(cp)
    t0 = time.time()
    try:
        return _subject(sub, dataset, refs_wanted, cp, t0)
    except Exception as ex:
        return dict(rows=None, info=dict(subject=sub, status=f'raised_{type(ex).__name__}: {ex}')), 'failed', time.time() - t0
    finally:
        gc.collect()


def _subject(sub, dataset, refs_wanted, cp, t0):
    c, err = clean_continuous(sub, dataset)
    if err:
        return dict(rows=None, info=dict(subject=sub, status=err)), 'excluded', time.time() - t0
    refs, scalp = reference_signals(c, dataset, [r for r in REFS[dataset] if r != 'csd'])
    if refs is None:
        return dict(rows=None, info=dict(subject=sub, status=scalp)), 'excluded', time.time() - t0
    ptp = combined_ptp(c, refs, scalp)
    sf = c['sfreq']
    fz = c['names'].index('Fz')
    pz = c['names'].index('Pz')
    sig = {r: (c['V'][fz] - s, c['V'][pz] - s) for r, s in refs.items()}
    if 'csd' in refs_wanted:
        try:
            pair, err = csd_pair(c, [c['names'][i] for i in scalp])
        except Exception as ex:
            pair, err = None, f'csd_failed_{type(ex).__name__}'
        if err:
            return dict(rows=None, info=dict(subject=sub, status=err)), 'excluded', time.time() - t0
        sig['csd'] = pair
    sig = {r: sig[r] for r in refs_wanted}
    samples, codes, is_t = trial_set(c, ptp)
    info = dict(subject=sub, n_ica=c['n_ica'], n_ica_excluded=c['n_excluded'], n_trials=len(samples), n_target=int(is_t.sum()), n_standard=int(len(is_t) - is_t.sum()))
    if is_t.sum() < MIN_TRIALS_PER_CONDITION or (len(is_t) - is_t.sum()) < MIN_TRIALS_PER_CONDITION or len(samples) < MIN_TRIALS_REQUIRED:
        info['status'] = 'below_minimum_trials'
        res = dict(rows=None, info=info)
        save_cache(res, cp)
        return res, 'excluded', time.time() - t0
    rows = []
    for r, (sf_z, sp_z) in sig.items():
        x, times = epochs(sf_z, samples, sf)
        xp, _ = epochs(sp_z, samples, sf)
        for label, t0w, t1w in WINDOWS:
            m = (times >= t0w) & (times < t1w)
            afz = x[:, m].mean(axis=1) * 1e6
            apz = xp[:, m].mean(axis=1) * 1e6
            for i in range(len(samples)):
                base = dict(subject=sub, dataset=dataset, reference=r, window=label, trial=i, code=int(codes[i]),
                            condition='Target' if is_t[i] else 'Standard', amp_fz=float(afz[i]), amp_pz=float(apz[i]))
                rows.extend(cd_rows(x[i, m], sf, base))
    info['status'] = 'ok'
    res = dict(rows=pd.DataFrame(rows), info=info)
    save_cache(res, cp)
    return res, 'computed', time.time() - t0


def auc(x, y):
    from scipy.stats import mannwhitneyu
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    return float(mannwhitneyu(x, y, alternative='two-sided').statistic / (len(x) * len(y)))


def summarise(per):
    g = ['dataset', 'reference', 'rate', 'window']
    rho, bias, fin, disc = [], [], [], []
    for key, w in per.groupby(g, sort=False):
        k = dict(zip(g, key))
        ok = w[w.finite == 1]
        r = ok.rho_med
        rho.append(dict(k, n_trials=len(w), n_subjects=w.subject.nunique(), median_rho=r.median(), iqr_low=r.quantile(0.25),
                        iqr_high=r.quantile(0.75), pct95=r.quantile(0.95), maximum=r.max(), prewhiten_rate=(r > RHO_THRESHOLD).mean(),
                        frac_rho_le_050=(r <= 0.5).mean()))
        step = ok.bias_subtracted / ok.n_steps_used
        cf = ok.bias_closed_form / ok.n_steps_used
        rel = (step - cf).abs() / cf
        bias.append(dict(k, median_m=ok.n_win.median(), min_m=ok.n_win.min(), median_n_eff=ok.n_eff.median(), min_n_eff=ok.n_eff.min(),
                         median_k_eff=ok.n_steps_used.median(), closed_form_b_at_median_n_eff=closed_form_b(ok.n_eff.median()),
                         median_bias_per_step=step.median(), max_rel_error=rel.max(), n_rows_bias_zero_n_eff_le3=int((ok.n_eff <= 3).sum()),
                         n_rows_m_below_8=int(ok.below_min_m.sum())))
        fin.append(dict(k, finite=int(w.finite.sum()), total=len(w), rate_finite=w.finite.mean()))
        for subset, d in (('all', ok), ('m_ge_8', ok[ok.below_min_m == 0])):
            t, s_ = d[d.condition == 'Target'], d[d.condition == 'Standard']
            row = dict(k, subset=subset, n_target=len(t), n_standard=len(s_))
            for col in AUC_COLS:
                a = auc(t[col], s_[col])
                row[col] = a
                row[col + '_folded'] = max(a, 1 - a) if np.isfinite(a) else np.nan
            disc.append(row)
    return {'rho': pd.DataFrame(rho), 'bias': pd.DataFrame(bias), 'finite': pd.DataFrame(fin), 'auc': pd.DataFrame(disc), 'auc_subject': subject_auc(per)}


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj


def subject_auc(per):
    from scipy.stats import wilcoxon
    g = ['dataset', 'reference', 'rate', 'window']
    out = []
    for key, w in per[per.finite == 1].groupby(g, sort=False):
        for col in SUBJECT_AUC_COLS:
            a = []
            for _, h in w.groupby('subject', sort=True):
                t = h.loc[h.condition == 'Target', col].dropna()
                u = h.loc[h.condition == 'Standard', col].dropna()
                if len(t) >= MIN_TRIALS_SUBJECT_AUC and len(u) >= MIN_TRIALS_SUBJECT_AUC:
                    a.append(auc(t, u))
            a = np.array(a)
            p = float(wilcoxon(a - 0.5).pvalue) if len(a) > 0 and np.any(a != 0.5) else np.nan
            out.append(dict(zip(g, key), metric=col, n_subjects=len(a), mean_auc=a.mean() if len(a) else np.nan,
                            sd_auc=a.std(ddof=1) if len(a) > 1 else np.nan, p_wilcoxon=p))
    df = pd.DataFrame(out)
    df['p_holm'] = np.nan
    for _, d in df.groupby('dataset'):
        ok = d.p_wilcoxon.notna()
        df.loc[d.index[ok], 'p_holm'] = holm(d.p_wilcoxon[ok].values)
    return df


def write_summary(per, out, stem):
    for name, df in summarise(per).items():
        f = os.path.join(out, f'{stem}_{name}.csv')
        df.to_csv(f, index=False)
        print(f'wrote {os.path.basename(f)} ({len(df)} rows)')
    return 0


def main():
    global ERPCORE_DIR, DS006018_DIR, DS006018_SUBDIR, DS006018_TASK, REJECT_UV, DO_ICA, ICA_N_COMPONENTS
    ap = argparse.ArgumentParser(description='Cumulative divergence on single-trial Fz of ERP CORE P3 and OpenNeuro ds006018 under five derivations at the native rate and at 128 Hz.')
    ap.add_argument('--dataset', required=True, choices=['erpcore', 'ds006018'])
    ap.add_argument('--erpcore', default=ERPCORE_DIR)
    ap.add_argument('--ds006018', default=DS006018_DIR)
    ap.add_argument('--ds006018-subdir', default=DS006018_SUBDIR)
    ap.add_argument('--out', default=RESULTS_DIR)
    ap.add_argument('--subjects', default=None)
    ap.add_argument('--refs', default=None)
    ap.add_argument('--reject', type=float, default=REJECT_UV)
    ap.add_argument('--no-ica', action='store_true')
    ap.add_argument('--ica-n', type=int, default=ICA_N_COMPONENTS)
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--summary-only', action='store_true')
    ap.add_argument('--part', default=None)
    a = ap.parse_args()
    part = None
    if a.part:
        m = re.fullmatch(r'\s*(\d+)\s*/\s*(\d+)\s*', a.part)
        if not m or not 1 <= int(m.group(1)) <= int(m.group(2)):
            print('--part takes K/N with 1 <= K <= N, for example --part 1/8')
            return 2
        part = (int(m.group(1)), int(m.group(2)))
    ERPCORE_DIR, DS006018_DIR, DS006018_SUBDIR = os.path.abspath(a.erpcore), os.path.abspath(a.ds006018), a.ds006018_subdir
    REJECT_UV, DO_ICA, ICA_N_COMPONENTS = float(a.reject), not a.no_ica, int(a.ica_n)
    refs_wanted = [r.strip() for r in a.refs.split(',') if r.strip()] if a.refs else list(REFS[a.dataset])
    bad = [r for r in refs_wanted if r not in REFS[a.dataset]]
    if bad:
        print(f'derivations not available for {a.dataset}: {bad}; choose from {list(REFS[a.dataset])}')
        return 2
    want = sorted(x.strip().replace('sub-', '').zfill(3) for x in a.subjects.split(',') if x.strip()) if a.subjects else None
    stem = f'cd_realdata_{a.dataset}' + ('' if DO_ICA else '_noica') + ('_sub-' + '-'.join(want) if want else '')
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    parts = lambda: sorted(glob.glob(os.path.join(out, f'{stem}_pertrial_*.csv.gz')))
    if a.summary_only:
        order = {r: i for i, r in enumerate(REFS[a.dataset])}
        files = sorted(parts(), key=lambda f: order.get(f.rsplit('_pertrial_', 1)[1][:-7], len(order)))
        if not files:
            print(f'no per-trial files {stem}_pertrial_*.csv.gz in {out}')
            return 2
        return write_summary(pd.concat([pd.read_csv(f, float_precision='round_trip') for f in files], ignore_index=True), out, stem)
    print(f'dataset {a.dataset} | derivations {refs_wanted} | reject {REJECT_UV:g} uV | ICA {ICA_N_COMPONENTS if DO_ICA else "off"} | out {out}')
    _self_test_scale_invariance()
    subs = erpcore_subjects() if a.dataset == 'erpcore' else ds006018_subjects()
    if not subs:
        print(f'no recordings found under {ERPCORE_DIR if a.dataset == "erpcore" else _ds006018_root()}')
        return 2
    subs = ['sub-' + s for s in subs]
    if want:
        subs = [s for s in subs if s.replace('sub-', '') in set(want)]
    if part:
        k, n = part
        subs = [subs[i] for i in np.array_split(np.arange(len(subs)), n)[k - 1]]
        print(f'part {k}/{n}: {", ".join(subs) if subs else "no participants"}')
    if a.jobs > 2:
        print(f'note: each job holds one recording, its ICA and its CSD in memory (about 1.5 GB for ERP CORE); {a.jobs} jobs need about {1.5 * a.jobs:.0f} GB')
    cache_dir = os.path.join(out, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    state = {k: globals()[k] for k in SETTINGS}
    tasks = [(s, a.dataset, refs_wanted, cache_dir, state) for s in subs]
    it = ProcessPoolExecutor(max_workers=a.jobs).map(subject, tasks) if a.jobs > 1 else map(subject, tasks)
    rows, infos = [], []
    for i, (res, what, dt) in enumerate(it, 1):
        infos.append(res['info'])
        if res['rows'] is not None and not part:
            rows.append(res['rows'])
        print(f"  [{i}/{len(subs)}] {tasks[i - 1][0]}: {res['info'].get('status')} | {what} {dt:.0f}s", flush=True)
    failed = [d['subject'] for d in infos if str(d.get('status', '')).startswith(('raised_', 'ica_failed_', 'read_failed_'))]
    if part:
        k, n = part
        print(f'part {k}/{n} finished: {len(infos) - len(failed)} participant(s) in the cache, {len(failed)} failed' + (f' ({", ".join(failed)})' if failed else '') + '.')
        print(f'When every part is finished, run without --part to write the output files; any participant still missing is computed then.')
        return 1 if failed else 0
    pd.DataFrame(infos).to_csv(os.path.join(out, f'{stem}_subjects.csv'), index=False)
    if failed:
        print(f'{len(failed)} participant(s) failed: {", ".join(failed)} (reasons in {stem}_subjects.csv).')
        print('No per-trial or summary files were written. Run the same command again, with fewer --jobs if memory ran out;')
        print('participants already finished are read from the cache.')
        return 1
    if not rows:
        print(f'no subject processed; see {stem}_subjects.csv')
        return 1
    per = pd.concat(rows, ignore_index=True)
    for f in parts():
        os.remove(f)
    for r in refs_wanted:
        f = os.path.join(out, f'{stem}_pertrial_{r}.csv.gz')
        per[per.reference == r].to_csv(f, index=False, float_format='%.17g', compression={'method': 'gzip', 'mtime': 0})
        print(f'wrote {os.path.basename(f)} ({int((per.reference == r).sum())} rows)')
    return write_summary(per, out, stem)


if __name__ == '__main__':
    sys.exit(main())

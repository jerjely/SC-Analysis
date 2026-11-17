"""
Baden-battery (B1) analyses: QI, feature vectors, DS/OS, colour opponency,
speed-tuning block, SparsePCA + GMM clustering, cluster profile and the
multi-session export (cells 42-64 of the notebook).
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from . import sync
from .psth import compute_trial_psths, compute_trial_psths_windowed

NOMINAL_DUR = {'step': 1.0, 'bar': 3.85, 'colour': 1.0, 'freq_chirp': 8.0, 'contr_chirp': 8.0}
ROUTINE_COLORS = {'step': 'tab:blue', 'freq_chirp': 'tab:orange', 'contr_chirp': 'tab:green',
                  'bar': 'tab:red', 'colour': 'tab:purple'}
BAR_DIR_COLORS = {0: 'tab:blue', 90: 'tab:orange', 180: 'tab:green', 270: 'tab:red'}


class B1Timing(object):
    """All the B1-derived constants from cells 42/43/47 for one session."""

    def __init__(self, sess):
        cfg = sess.cfg
        self.df = sess.df_b1
        self.scale, _ = sync.b1_scale(self.df, cfg)
        s = self.scale
        nd = dict(NOMINAL_DUR)
        nd['bar'] = float(cfg.b1_bar_dur)
        self.nominal = nd
        self.bin = float(cfg.bin_s_feature)
        self.sigma = float(cfg.smooth_sigma_bins)
        self.dirs = list(cfg.b1_bar_directions)
        self.colors = list(cfg.b1_color_names)
        self.chirp_dur = nd['freq_chirp'] * s
        self.contr_dur = nd['contr_chirp'] * s
        self.step_dur = nd['step'] * s
        self.bar_dur = nd['bar'] * s
        self.color_dur = nd['colour'] * s
        STEP_GAP = COLOR_GAP = 0.5
        self.step_gap = STEP_GAP
        self.color_gap = COLOR_GAP
        self.freq_pre, self.freq_post = 0.5 * s, (nd['freq_chirp'] + 0.5) * s
        self.contr_pre, self.contr_post = 0.5 * s, (nd['contr_chirp'] + 0.5) * s
        self.step_pre, self.step_post = 0.5 * s, (2 * nd['step'] + 2 * STEP_GAP) * s
        self.color_pre, self.color_post = 0.5 * s, (2 * nd['colour'] + 2 * COLOR_GAP) * s
        self.bar_align_pre, self.bar_align_post = 1.0 * s, 1.5 * s
        self.bar_align_n_bins = len(np.arange(0, self.bar_align_pre + self.bar_align_post + self.bin,
                                              self.bin)) - 1
        self.bar_speed = float(cfg.bar_speed_deg)
        self.bar_half_travel = (float(cfg.screen_diag_deg) + 20.0) / 2.0
        self.y_flip = bool(cfg.y_sign_flip)
        self.x_flip = bool(cfg.x_sign_flip)
        self.baseline_s = 0.5 * s
        self.chirp_freq = self.get('freq_chirp')
        self.chirp_contr = self.get('contr_chirp')
        self.step_on = self.get('step', 'ON')
        self.bar_by_dir = {d: self.get('bar', '{}deg'.format(d)) for d in self.dirs}
        self.blue = self.get('colour', self.colors[0])
        self.green = self.get('colour', self.colors[1]) if len(self.colors) > 1 else np.array([])

    def get(self, routine, condition=''):
        m = self.df['routine'] == routine
        if condition != '':
            m &= (self.df['condition'] == condition)
        on = np.sort(self.df.loc[m, 'onset_s'].values)
        if len(on) == 0:
            raise ValueError('No B1 events for routine={!r} condition={!r}. Available: {} | {}'.format(
                routine, condition, sorted(self.df['routine'].unique()),
                sorted(self.df['condition'].astype(str).unique())))
        return on

    def bar_cross(self, x_deg, y_deg, direction):
        rad = np.radians(direction)
        xc = -x_deg if self.x_flip else x_deg
        yc = -y_deg if self.y_flip else y_deg
        return (xc * np.cos(rad) + yc * np.sin(rad) + self.bar_half_travel) / self.bar_speed

    def n_bins(self, pre, post):
        return len(np.arange(0, pre + post + self.bin, self.bin)) - 1

    def segment_bounds(self):
        nb = [self.n_bins(self.freq_pre, self.freq_post), self.n_bins(self.contr_pre, self.contr_post),
              self.n_bins(self.step_pre, self.step_post),
              self.n_bins(self.bar_align_pre, self.bar_align_post),
              self.n_bins(self.color_pre, self.color_post)]
        e = np.cumsum([0] + nb)
        names = ['freq_chirp', 'contr_chirp', 'step', 'bar (pooled directions)',
                 'colour (blue → green)']
        return [(int(e[i]), int(e[i + 1]), names[i]) for i in range(5)]

    def internal_edges(self):
        b = self.bin
        s = self.scale
        return {
            'freq_chirp': [('baseline→chirp', self.freq_pre / b)],
            'contr_chirp': [('baseline→chirp', self.contr_pre / b)],
            'step': [('ON', self.step_pre / b), ('gap', (self.step_pre + self.step_dur) / b),
                     ('OFF', (self.step_pre + self.step_dur + self.step_gap * s) / b)],
            'bar (pooled directions)': [('RF crossing', self.bar_align_pre / b)],
            'colour (blue → green)': [
                ('blue', self.color_pre / b), ('gap', (self.color_pre + self.color_dur) / b),
                ('green', (self.color_pre + self.color_dur + self.color_gap * s) / b)],
        }


def get_timing(sess):
    if 'b1timing' not in sess.cache:
        sess.cache['b1timing'] = B1Timing(sess)
    return sess.cache['b1timing']


# ---------------------------------------------------------------------------
# QI (cell 43)
# ---------------------------------------------------------------------------
def compute_qi(trial_mat, sigma):
    if trial_mat.shape[0] < 2:
        return np.nan
    sm = gaussian_filter1d(trial_mat, sigma=sigma, axis=1)
    den = np.mean(np.var(sm, axis=1))
    if den <= 0:
        return np.nan
    return np.var(sm.mean(axis=0)) / den


def zscore_safe_smoothed(trace, sigma):
    sm = gaussian_filter1d(trace, sigma=sigma)
    sd = sm.std()
    if not np.isfinite(sd) or sd < 1e-8:
        return sm, False
    return (sm - sm.mean()) / sd, True


def run_qi(sess, log=print, progress=None):
    T = get_timing(sess)
    qi = {}
    ids = list(sess.good_ids)
    for i, cid in enumerate(ids):
        st = sess.spike_trains[cid]
        q1 = compute_qi(compute_trial_psths(st, T.chirp_freq, T.chirp_dur, T.bin), T.sigma)
        q2 = compute_qi(compute_trial_psths(st, T.chirp_contr, T.contr_dur, T.bin), T.sigma)
        q3 = compute_qi(compute_trial_psths(st, T.step_on, T.step_dur, T.bin), T.sigma)
        bm = np.concatenate([compute_trial_psths(st, T.bar_by_dir[d], T.bar_dur, T.bin)
                             for d in T.dirs], axis=0)
        q4 = compute_qi(bm, T.sigma)
        vals = [q for q in (q1, q2, q3, q4) if np.isfinite(q)]
        qi[int(cid)] = float(np.max(vals)) if vals else np.nan
        if progress:
            progress(i + 1, len(ids))
    thr = float(sess.cfg.qi_threshold)
    log('B1 scale factor: {:.4f}'.format(T.scale))
    log('qi_by_unit computed for {} units | threshold = {}'.format(len(qi), thr))
    log('  units passing QI: {}'.format(sum(v >= thr for v in qi.values() if np.isfinite(v))))
    sess.results['qi'] = qi
    return qi


# ---------------------------------------------------------------------------
# feature vectors (cell 45 + 47)
# ---------------------------------------------------------------------------
def run_features(sess, log=print, progress=None):
    T = get_timing(sess)
    qi = sess.results.get('qi') or run_qi(sess, log)
    thr = float(sess.cfg.qi_threshold)
    rfc = sess.rf_clean
    qi_pass = {c for c, q in qi.items() if np.isfinite(q) and q >= thr}
    rf_pass = {int(c) for c in sess.good_ids if c in rfc and rfc[c].get('fit_ok', False)}
    combined = qi_pass | rf_pass
    log('QI-only: {} | RF-only: {} | overlap: {} | union: {}'.format(
        len(qi_pass), len(rf_pass), len(qi_pass & rf_pass), len(combined)))
    feats, valid, is_qi, has_bar, has_col = {}, [], [], [], []
    comb = sorted(combined)
    for i, cid in enumerate(comb):
        st = sess.spike_trains[cid]
        fz, a = zscore_safe_smoothed(compute_trial_psths_windowed(
            st, T.chirp_freq, T.freq_pre, T.freq_post, T.bin).mean(axis=0), T.sigma)
        cz, b = zscore_safe_smoothed(compute_trial_psths_windowed(
            st, T.chirp_contr, T.contr_pre, T.contr_post, T.bin).mean(axis=0), T.sigma)
        sz, c = zscore_safe_smoothed(compute_trial_psths_windowed(
            st, T.step_on, T.step_pre, T.step_post, T.bin).mean(axis=0), T.sigma)
        if not (a and b and c):
            continue
        rf = rfc.get(cid)
        if rf and rf.get('fit_ok'):
            bt = np.concatenate([compute_trial_psths_windowed(
                st, T.bar_by_dir[d] + T.bar_cross(rf['x_deg'], rf['y_deg'], d) * T.scale,
                T.bar_align_pre, T.bar_align_post, T.bin) for d in T.dirs], axis=0)
            bz, okb = zscore_safe_smoothed(bt.mean(axis=0), T.sigma)
        else:
            bz, okb = np.zeros(T.bar_align_n_bins), False
        if not okb:
            bz = np.zeros_like(bz)
        colz, okc = zscore_safe_smoothed(compute_trial_psths_windowed(
            st, T.blue, T.color_pre, T.color_post, T.bin).mean(axis=0), T.sigma)
        if not okc:
            colz = np.zeros_like(colz)
        feats[cid] = np.concatenate([fz, cz, sz, bz, colz])
        valid.append(cid); is_qi.append(cid in qi_pass); has_bar.append(okb); has_col.append(okc)
        if progress:
            progress(i + 1, len(comb))
    X = np.array([feats[c] for c in valid]) if valid else np.zeros((0, 0))
    r = {'valid_units': valid, 'is_qi_pass': np.array(is_qi, dtype=bool),
         'has_bar_trace': np.array(has_bar, dtype=bool), 'has_color_trace': np.array(has_col, dtype=bool),
         'X': X, 'qi_pass': qi_pass, 'rf_pass': rf_pass,
         'rf_only_flag': {c: (c not in qi_pass) for c in valid}}
    log('{} / {} units produced valid feature vectors | QI-pass {} | RF-only {}'.format(
        len(valid), len(combined), int(r['is_qi_pass'].sum()), int((~r['is_qi_pass']).sum())))
    log('  reliable bar trace: {} | reliable colour trace: {}'.format(
        int(r['has_bar_trace'].sum()), int(r['has_color_trace'].sum())))
    log('Feature matrix shape: {}'.format(X.shape))
    sess.results['features'] = r
    return r


# ---------------------------------------------------------------------------
# DS / OS
# ---------------------------------------------------------------------------
def vector_selectivity(mean_resp, dirs):
    theta = np.deg2rad(dirs)
    r = np.clip(mean_resp, 0, None)
    tot = r.sum()
    if tot <= 0:
        return 0.0, np.nan, 0.0, np.nan
    vd = np.sum(r * np.exp(1j * theta))
    vo = np.sum(r * np.exp(1j * 2 * theta))
    return (np.abs(vd) / tot, np.degrees(np.angle(vd)) % 360,
            np.abs(vo) / tot, (np.degrees(np.angle(vo)) / 2) % 180)


def _vector_sel_batch(means, dirs):
    """means: (n_perm, n_dirs) -> dsi, osi arrays (same maths as vector_selectivity)."""
    theta = np.deg2rad(np.asarray(dirs, dtype=float))
    r = np.clip(means, 0, None)
    tot = r.sum(axis=1)
    vd = np.abs((r * np.exp(1j * theta)).sum(axis=1))
    vo = np.abs((r * np.exp(1j * 2 * theta)).sum(axis=1))
    with np.errstate(invalid='ignore', divide='ignore'):
        dsi = np.where(tot > 0, vd / np.where(tot > 0, tot, 1), 0.0)
        osi = np.where(tot > 0, vo / np.where(tot > 0, tot, 1), 0.0)
    return dsi, osi


_PERM_CACHE = {}


def _perms(n, n_perm, seed=0):
    key = (n, n_perm, seed)
    if key not in _PERM_CACHE:
        rng = np.random.RandomState(seed)
        _PERM_CACHE[key] = np.array([rng.permutation(n) for _ in range(n_perm)])
    return _PERM_CACHE[key]


def permutation_pvalues(trial_resp_by_dir, dirs, obs_dsi, obs_osi, n_perm):
    """Identical to the notebook's permutation_pvalue (seed 0, same shuffles)
    but vectorised; returns (p_dsi, p_osi)."""
    allt = np.concatenate([trial_resp_by_dir[d] for d in dirs])
    nper = [len(trial_resp_by_dir[d]) for d in dirs]
    P = _perms(len(allt), n_perm)
    sh = allt[P]
    means = []
    i0 = 0
    for n in nper:
        means.append(sh[:, i0:i0 + n].mean(axis=1)); i0 += n
    means = np.column_stack(means)
    dsi, osi = _vector_sel_batch(means, dirs)
    return float(np.mean(dsi >= obs_dsi)), float(np.mean(osi >= obs_osi))


def _spike_count(st, t0s, dur):
    t0s = np.asarray(t0s)
    return np.searchsorted(st, t0s + dur, side='left') - np.searchsorted(st, t0s, side='left')


def run_dsos(sess, log=print, progress=None, pooled=True):
    """pooled=True: cell 49/61 (pooled baseline, used for clustering).
       pooled=False: cell 58 (per-trial baseline, supplementary histograms)."""
    T = get_timing(sess)
    n_perm = int(sess.cfg.n_permutations)
    res = {}
    ids = list(sess.good_ids)
    for i, cid in enumerate(ids):
        st = sess.spike_trains[cid]
        if pooled:
            cnt = [_spike_count(st, T.bar_by_dir[d] - T.baseline_s, T.baseline_s) for d in T.dirs]
            allc = np.concatenate(cnt)
            pb = allc.sum() / (len(allc) * T.baseline_s)
            tr = {d: _spike_count(st, T.bar_by_dir[d], T.bar_dur) / T.bar_dur for d in T.dirs}
            mr = np.array([tr[d].mean() for d in T.dirs]) - pb
        else:
            tr = {d: (_spike_count(st, T.bar_by_dir[d], T.bar_dur) / T.bar_dur -
                      _spike_count(st, T.bar_by_dir[d] - T.baseline_s, T.baseline_s) / T.baseline_s)
                  for d in T.dirs}
            mr = np.array([tr[d].mean() for d in T.dirs])
            pb = np.nan
        dsi, pdir, osi, pori = vector_selectivity(mr, T.dirs)
        p_d, p_o = permutation_pvalues(tr, T.dirs, dsi, osi, n_perm)
        tot = int(sum(_spike_count(st, T.bar_by_dir[d], T.bar_dur).sum() for d in T.dirs))
        res[int(cid)] = {'dsi': dsi, 'pref_dir': pdir, 'p_dsi': p_d, 'osi': osi, 'pref_ori': pori,
                         'p_osi': p_o, 'mean_resp': mr, 'pooled_baseline': pb, 'total_spikes': tot}
        if progress:
            progress(i + 1, len(ids))
    mn_s, mn_r = int(sess.cfg.ds_min_total_spikes), float(sess.cfg.ds_min_peak_rate)
    filt = {c: r for c, r in res.items() if r['total_spikes'] >= mn_s and r['mean_resp'].max() >= mn_r}
    key = 'dsos_v2' if pooled else 'dsos_v1'
    sess.results[key] = res
    sess.results[key + '_filtered'] = filt
    log('DS/OS ({}) reliable (activity-filtered): {} / {} units'.format(
        'pooled baseline' if pooled else 'per-trial baseline', len(filt), len(res)))
    return res, filt


# ---------------------------------------------------------------------------
# colour opponency (cell 51)
# ---------------------------------------------------------------------------
def run_colour(sess, log=print, progress=None):
    T = get_timing(sess)
    n_perm = int(sess.cfg.n_permutations)
    dur, base = T.color_dur, T.baseline_s
    res = {}
    ids = list(sess.good_ids)
    P_cache = {}
    for i, cid in enumerate(ids):
        st = sess.spike_trains[cid]
        bt = _spike_count(st, T.blue, dur) / dur - _spike_count(st, T.blue - base, base) / base
        gt = _spike_count(st, T.green, dur) / dur - _spike_count(st, T.green - base, base) / base
        br_, gr_ = max(bt.mean(), 0), max(gt.mean(), 0)
        tot = br_ + gr_
        opp = (gr_ - br_) / tot if tot > 0 else 0.0
        pooled = np.concatenate([bt, gt])
        nb = len(bt)
        P = _perms(len(pooled), n_perm)
        sh = pooled[P]
        bm = np.clip(sh[:, :nb].mean(axis=1), 0, None)
        gm = np.clip(sh[:, nb:].mean(axis=1), 0, None)
        tt = bm + gm
        with np.errstate(invalid='ignore', divide='ignore'):
            null = np.where(tt > 0, (gm - bm) / np.where(tt > 0, tt, 1), 0.0)
        p = float(np.mean(np.abs(null) >= abs(opp)))
        tsp = int(_spike_count(st, np.concatenate([T.blue, T.green]), dur).sum())
        pk = max(br_, gr_)
        res[int(cid)] = {'blue_resp': br_, 'green_resp': gr_, 'opponency': opp, 'p_opponency': p,
                         'total_spikes': tsp, 'peak_resp': pk,
                         'reliable': tsp >= int(sess.cfg.color_min_total_spikes) and
                         pk >= float(sess.cfg.color_min_peak_resp)}
        if progress:
            progress(i + 1, len(ids))
    nr = sum(r['reliable'] for r in res.values())
    ns = sum(r['reliable'] and r['p_opponency'] < 0.05 for r in res.values())
    log('Colour-reliable units: {} / {} | significantly opponent (p<0.05): {}'.format(nr, len(res), ns))
    sess.results['colour'] = res
    return res


# ---------------------------------------------------------------------------
# speed block (cell 50)
# ---------------------------------------------------------------------------
SPEEDS = [10, 20, 25, 30, 40, 50]


def speed_block(sess, valid_units):
    df = sess.picks_df('sweep')
    n = len(valid_units)
    fr = np.full((n, len(SPEEDS)), np.nan)
    frac = np.zeros(n)
    if df is not None and len(df):
        df = df.set_index('cid')
        for i, cid in enumerate(valid_units):
            if cid not in df.index:
                continue
            row = df.loc[cid]
            k = 0
            for j, spd in enumerate(SPEEDS):
                nc, fc = 'n_picks_{}dps'.format(spd), 'p1_fr_{}dps'.format(spd)
                if nc in row and pd.notna(row[nc]) and row[nc] >= 1 and pd.notna(row.get(fc, np.nan)):
                    fr[i, j] = row[fc]; k += 1
            frac[i] = k / len(SPEEDS)
    with np.errstate(invalid='ignore', all='ignore'):
        rm = np.nanmax(np.where(np.isnan(fr), -np.inf, fr), axis=1, keepdims=True)
    rm = np.where(np.isfinite(rm) & (rm > 0), rm, np.nan)
    shape = fr / rm
    import warnings
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cm = np.nanmean(shape, axis=0)
    inds = np.where(np.isnan(shape))
    shape[inds] = np.take(cm, inds[1])
    shape = np.nan_to_num(shape)
    return fr, frac, shape


# ---------------------------------------------------------------------------
# clustering (cell 52)
# ---------------------------------------------------------------------------
def run_clustering(sess, log=print, progress=None):
    from sklearn.decomposition import SparsePCA
    from sklearn.mixture import GaussianMixture
    F = sess.results.get('features') or run_features(sess, log)
    X, isq = F['X'], F['is_qi_pass']
    if X.shape[0] < 4 or isq.sum() < 3:
        raise ValueError('Too few units with feature vectors ({}, {} QI-pass) to cluster'.format(
            X.shape[0], int(isq.sum())))
    cfg = sess.cfg
    ncomp = min(int(cfg.n_sparse_components), int(isq.sum()) - 1, X.shape[1])
    spca = SparsePCA(n_components=ncomp, random_state=0)
    Xf = spca.fit_transform(X[isq])
    log('Fit set reduced shape: {}'.format(Xf.shape))
    kmax = min(int(cfg.max_clusters), Xf.shape[0] - 1)
    ks = list(range(2, max(3, kmax + 1)))
    bics = []
    for i, k in enumerate(ks):
        g = GaussianMixture(n_components=k, covariance_type='diag', n_init=int(cfg.gmm_n_init_bic),
                            random_state=0)
        g.fit(Xf)
        bics.append(g.bic(Xf))
        if progress:
            progress(i + 1, len(ks) + 1)
    best_k = ks[int(np.argmin(bics))]
    log('BIC-selected k: {}'.format(best_k))
    gm = GaussianMixture(n_components=best_k, covariance_type='diag',
                         n_init=int(cfg.gmm_n_init_final), random_state=0)
    lf = gm.fit_predict(Xf)
    labels = np.empty(len(F['valid_units']), dtype=int)
    labels[isq] = lf
    if (~isq).sum():
        labels[~isq] = gm.predict(spca.transform(X[~isq]))
    for k in range(best_k):
        mem = [F['valid_units'][i] for i in range(len(labels)) if labels[i] == k]
        log('  cluster {}: {} units ({} RF-only)'.format(k, len(mem), sum(F['rf_only_flag'][c] for c in mem)))
    r = {'ks': ks, 'bics': bics, 'best_k': best_k, 'labels': labels, 'spca': spca, 'gmm': gm}
    sess.results['clustering'] = r
    if progress:
        progress(1, 1)
    return r


def cluster_profile(sess):
    F = sess.results['features']
    C = sess.results['clustering']
    ds = sess.results.get('dsos_v2_filtered', {})
    col = sess.results.get('colour', {})
    vu = F['valid_units']
    _, frac, shape = speed_block(sess, vu)
    rows = []
    for k in range(C['best_k']):
        mem = [vu[i] for i in range(len(vu)) if C['labels'][i] == k]
        dv = [ds[c] for c in mem if c in ds]
        cv = [col[c] for c in mem if col.get(c, {}).get('reliable', False)]
        si = [i for i, c in enumerate(vu) if c in mem and frac[i] == 1.0]
        rows.append({'cluster': k, 'n_units': len(mem), 'n_ds': len(dv),
                     'mean_DSI': np.mean([r['dsi'] for r in dv]) if dv else np.nan,
                     'mean_OSI': np.mean([r['osi'] for r in dv]) if dv else np.nan,
                     'n_color': len(cv),
                     'mean_color_opp (blue-/green+)': np.mean([r['opponency'] for r in cv]) if cv else np.nan,
                     'n_speed_full': len(si),
                     'pref_speed_dps': SPEEDS[int(np.argmax(shape[si].mean(axis=0)))] if si else np.nan})
    return pd.DataFrame(rows).set_index('cluster')


def availability_table(sess):
    F = sess.results['features']
    C = sess.results['clustering']
    ds = sess.results.get('dsos_v2_filtered', {})
    vu = F['valid_units']
    _, frac, _ = speed_block(sess, vu)
    has_ds = np.array([1.0 if c in ds else 0.0 for c in vu])
    rows = []
    for k in range(C['best_k']):
        m = C['labels'] == k
        rows.append({'cluster': k, 'n': int(m.sum()),
                     'frac_has_ds': has_ds[m].mean() if m.sum() else np.nan,
                     'frac_has_speed': frac[m].mean() if m.sum() else np.nan,
                     'frac_has_bar_trace': F['has_bar_trace'][m].mean() if m.sum() else np.nan,
                     'frac_has_color_trace': F['has_color_trace'][m].mean() if m.sum() else np.nan})
    return pd.DataFrame(rows).set_index('cluster')


# ---------------------------------------------------------------------------
# export (cell 56)
# ---------------------------------------------------------------------------
COMMON = {'chirp': 400, 'contr': 400, 'step': 150, 'bar': 150, 'color': 150}


def resample_to_common(trace, n):
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(trace)), trace)


def run_export(sess, log=print):
    F = sess.results.get('features')
    if F is None:
        raise ValueError('Compute feature vectors first')
    T = get_timing(sess)
    X, vu = F['X'], F['valid_units']
    sizes = [COMMON['chirp'], COMMON['contr'], COMMON['step'], COMMON['bar'], COMMON['color']]
    sb = T.segment_bounds()
    Xc = np.zeros((X.shape[0], sum(sizes)))
    for i in range(X.shape[0]):
        o = 0
        for (a, b, _), n in zip(sb, sizes):
            Xc[i, o:o + n] = resample_to_common(X[i, a:b], n); o += n
    rfc = sess.rf_clean
    bdc = np.full((len(vu), len(T.dirs), COMMON['bar']), np.nan)
    for i, cid in enumerate(vu):
        rf = rfc.get(cid)
        if not (rf and rf.get('fit_ok')):
            continue
        for j, d in enumerate(T.dirs):
            tm = compute_trial_psths_windowed(sess.spike_trains[cid],
                                              T.bar_by_dir[d] + T.bar_cross(rf['x_deg'], rf['y_deg'], d) * T.scale,
                                              T.bar_align_pre, T.bar_align_post, T.bin)
            z, ok = zscore_safe_smoothed(tm.mean(axis=0), T.sigma)
            if ok:
                bdc[i, j] = resample_to_common(z, COMMON['bar'])
    cfg = sess.cfg
    out = os.path.join(cfg.save_dir, 'clustering_export')
    os.makedirs(out, exist_ok=True)
    qi = sess.results.get('qi', {})
    ds = sess.results.get('dsos_v2_filtered', {})
    sv = lambda name, arr: np.save(os.path.join(out, name + '.npy'), arr)
    sv('X_common', Xc); sv('bar_dir_common', bdc); sv('valid_units', np.array(vu))
    sv('is_qi_pass', F['is_qi_pass'])
    sv('qi_values', np.array([qi.get(c, np.nan) for c in vu]))
    sv('rf_only', np.array([F['rf_only_flag'][c] for c in vu]))
    sv('depths', np.array([sess.unit_depths.get(c, np.nan) for c in vu]))
    sv('has_bar_trace', F['has_bar_trace']); sv('has_color_trace', F['has_color_trace'])
    for k in ['x_deg', 'y_deg', 'diameter']:
        sv('rf_' + k, np.array([rfc.get(c, {}).get(k, np.nan) for c in vu]))
    sv('rf_fit_ok', np.array([bool(rfc.get(c, {}).get('fit_ok', False)) for c in vu]))
    for k in ['dsi', 'osi', 'pref_dir', 'pref_ori', 'p_dsi', 'p_osi']:
        sv(k, np.array([ds.get(c, {}).get(k, np.nan) for c in vu]))
    col = sess.results.get('colour', {})
    sv('color_opponency', np.array([col.get(c, {}).get('opponency', np.nan) for c in vu]))
    sv('color_p', np.array([col.get(c, {}).get('p_opponency', np.nan) for c in vu]))
    sv('color_reliable', np.array([bool(col.get(c, {}).get('reliable', False)) for c in vu]))
    _, sfrac, sshape = speed_block(sess, vu)
    sv('speed_shape', sshape); sv('speed_frac', sfrac)
    C = sess.results.get('clustering')
    if C is not None:
        sv('cluster_labels', C['labels'])
    meta = {'session_id': cfg.session_id, 'recording_date': cfg.recording_date,
            'mouse_line': cfg.mouse_line, 'recording_region': cfg.recording_region,
            'n_units': len(vu), 'qi_threshold': float(cfg.qi_threshold),
            'bin_s_feature': float(cfg.bin_s_feature), 'smooth_sigma_bins': float(cfg.smooth_sigma_bins),
            'b1_scale': float(T.scale), 'common_n_chirp': COMMON['chirp'], 'common_n_contr': COMMON['contr'],
            'common_n_step': COMMON['step'], 'common_n_bar': COMMON['bar'], 'common_n_color': COMMON['color'],
            'bar_directions': list(T.dirs), 'x_sign_flip': bool(cfg.x_sign_flip),
            'y_sign_flip': bool(cfg.y_sign_flip), 'speeds_dps': SPEEDS,
            'session_file': cfg.path(), 'exported_by': 'SC Analysis'}
    with open(os.path.join(out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    log('Exported {} units to {}'.format(len(vu), out))
    log('X_common {} | bar_dir_common {}'.format(Xc.shape, bdc.shape))
    return out

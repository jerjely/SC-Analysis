"""
Multi-recording classification.

A *recording list* (JSON) holds the analysis folders of completed recordings.
Each recording contributes its  <SAVE_DIR>/clustering_export/  (written by the
Baden clustering tab or the notebook's export cell).  Units from all included
recordings are pooled; SparsePCA + GMM (BIC model order) is fitted on the
pooled QI-pass units and RF-only units are assigned with the same model.
"""
import datetime
import json
import os

import numpy as np
import pandas as pd

from .config import SessionConfig, SESSION_FILENAME

EXPORT = 'clustering_export'
# internal event positions as a fraction of each common-length segment
# (independent of the B1 scale factor: every duration scales together)
SEG_EVENTS = {
    'freq_chirp': [('chirp', 0.5 / 9.0)],
    'contr_chirp': [('chirp', 0.5 / 9.0)],
    'step': [('ON', 0.5 / 3.5), ('gap', 1.5 / 3.5), ('OFF', 2.0 / 3.5)],
    'bar': [('RF crossing', 1.0 / 2.5)],
    'colour': [('blue', 0.5 / 3.5), ('gap', 1.5 / 3.5), ('green', 2.0 / 3.5)],
}
SEG_NAMES = ['freq_chirp', 'contr_chirp', 'step', 'bar', 'colour']


# ---------------------------------------------------------------------------
# recording list
# ---------------------------------------------------------------------------
class RecordingList(object):
    def __init__(self, path=None):
        self.path = path
        self.items = []          # dicts: folder, include, note
        if path and os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            self.items = d.get('recordings', [])
            self.settings = d.get('settings', {})
        else:
            self.settings = {}

    @property
    def out_dir(self):
        if not self.path:
            return ''
        return os.path.join(os.path.dirname(self.path),
                            os.path.splitext(os.path.basename(self.path))[0] + '_pooled')

    def save(self, path=None):
        self.path = path or self.path
        if not self.path:
            raise ValueError('No recording-list file chosen')
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump({'recordings': self.items, 'settings': self.settings}, f, indent=2)

    def folders(self):
        return [it['folder'] for it in self.items]

    def add(self, folder):
        folder = os.path.normpath(folder)
        if folder.lower().endswith('.json'):
            folder = os.path.dirname(folder)
        if os.path.basename(folder) == EXPORT:
            folder = os.path.dirname(folder)
        if folder in self.folders():
            return False
        self.items.append({'folder': folder, 'include': True, 'note': ''})
        self.items.sort(key=lambda it: os.path.basename(it['folder']))
        return True

    def remove(self, folders):
        self.items = [it for it in self.items if it['folder'] not in set(folders)]

    def included(self):
        return [it['folder'] for it in self.items if it.get('include', True)]


def scan(root, max_depth=4):
    """Analysis folders below root: those with sc_session.json or clustering_export/meta.json."""
    out = []
    base = os.path.normpath(root).count(os.sep)
    for r, dirs, files in os.walk(root):
        if os.path.normpath(r).count(os.sep) - base >= max_depth:
            dirs[:] = []
        dirs[:] = [d for d in dirs if d not in ('sorter_output', 'sweep_cell_plots') and not d.startswith('.')]
        if SESSION_FILENAME in files or os.path.exists(os.path.join(r, EXPORT, 'meta.json')):
            out.append(os.path.normpath(r))
            dirs[:] = [d for d in dirs if d == EXPORT]
    return sorted(set(out))


def status(folder):
    """Summary row for the recording table."""
    st = {'folder': folder, 'name': os.path.basename(folder), 'date': '', 'mouse_line': '', 'region': '',
          'has_session': os.path.exists(os.path.join(folder, SESSION_FILENAME)),
          'pipeline_ready': False, 'exported': False, 'n_units': 0, 'n_qi': 0, 'n_rf': 0,
          'export_time': '', 'has_colour_speed': False}
    if st['has_session']:
        try:
            cfg = SessionConfig.load(os.path.join(folder, SESSION_FILENAME))
            st.update(date=cfg.recording_date, mouse_line=cfg.mouse_line, region=cfg.recording_region,
                      name=cfg.session_id)
        except Exception:
            pass
    st['pipeline_ready'] = all(os.path.exists(os.path.join(folder, f)) for f in
                               ['rf_results_clean.npy', 'B1_photodiode_events_labeled.csv'])
    ed = os.path.join(folder, EXPORT)
    mp = os.path.join(ed, 'meta.json')
    if os.path.exists(mp):
        try:
            with open(mp) as f:
                m = json.load(f)
            st['exported'] = True
            st['date'] = st['date'] or m.get('recording_date', '')
            st['mouse_line'] = st['mouse_line'] or m.get('mouse_line', '')
            st['region'] = st['region'] or m.get('recording_region', '')
            if not st['has_session'] and m.get('session_id'):
                st['name'] = m['session_id']
            st['n_units'] = int(m.get('n_units', 0))
            q = np.load(os.path.join(ed, 'is_qi_pass.npy'))
            st['n_qi'] = int(q.sum()); st['n_rf'] = int((~q).sum())
            st['export_time'] = datetime.datetime.fromtimestamp(os.path.getmtime(mp)).strftime('%Y-%m-%d %H:%M')
            st['has_colour_speed'] = os.path.exists(os.path.join(ed, 'speed_shape.npy'))
        except Exception as e:
            st['error'] = str(e)
    return st


def export_recording(folder, log=print, progress=None, overwrite=False):
    """Run QI -> features -> DS/OS -> colour -> export for one recording (headless)."""
    from .session import Session
    from . import baden
    if os.path.exists(os.path.join(folder, EXPORT, 'meta.json')) and not overwrite:
        log('  already exported'); return False
    p = os.path.join(folder, SESSION_FILENAME)
    if not os.path.exists(p):
        raise ValueError('No sc_session.json in {} - open it once in SC Analysis (or export from the notebook)'.format(folder))
    cfg = SessionConfig.load(p)
    if not cfg.save_dir:
        cfg['save_dir'] = folder
    sess = Session(cfg)
    for k in ('b1', 'rfmode'):
        sess.require(k)
    baden.run_qi(sess, log)
    baden.run_features(sess, log)
    baden.run_dsos(sess, log)
    baden.run_colour(sess, log)
    baden.run_export(sess, log)
    return True


def export_missing(folders, overwrite=False, log=print, progress=None, cancel=None):
    done, failed = [], []
    for i, f in enumerate(folders):
        if cancel is not None and cancel():
            raise RuntimeError('Cancelled')
        log('\n[{}/{}] {}'.format(i + 1, len(folders), os.path.basename(f)))
        try:
            if export_recording(f, log, overwrite=overwrite):
                done.append(f)
        except Exception as e:
            log('  FAILED: {}'.format(e)); failed.append((f, str(e)))
        if progress:
            progress(i + 1, len(folders))
    log('\nExported {} | failed {}'.format(len(done), len(failed)))
    return done, failed


# ---------------------------------------------------------------------------
# pooling
# ---------------------------------------------------------------------------
PER_UNIT = ['is_qi_pass', 'qi_values', 'rf_only', 'depths', 'has_bar_trace', 'has_color_trace', 'rf_x_deg',
            'rf_y_deg', 'rf_diameter', 'rf_fit_ok', 'dsi', 'osi', 'pref_dir', 'pref_ori', 'p_dsi', 'p_osi',
            'color_opponency', 'color_p', 'color_reliable', 'speed_frac']


def load_pooled(folders, log=print):
    Xs, rows, bars, speeds, metas = [], [], [], [], []
    ref = None
    seen = {}
    for f in folders:
        ed = os.path.join(f, EXPORT)
        if not os.path.exists(os.path.join(ed, 'meta.json')):
            log('  skip (no export): {}'.format(f)); continue
        with open(os.path.join(ed, 'meta.json')) as fh:
            m = json.load(fh)
        sid = m.get('session_id', os.path.basename(f))
        if sid in seen:
            log('  SKIP {}: same recording ({}) already included from {}'.format(f, sid, seen[sid])); continue
        seen[sid] = f
        sizes = [m.get('common_n_' + k) for k in ('chirp', 'contr', 'step', 'bar', 'color')]
        if ref is None:
            ref = (sizes, m.get('bar_directions'), m.get('x_sign_flip'), m.get('y_sign_flip'))
        elif sizes != ref[0]:
            log('  SKIP {}: feature lengths {} differ from {}'.format(m.get('session_id'), sizes, ref[0])); continue
        elif (m.get('x_sign_flip'), m.get('y_sign_flip')) != (ref[2], ref[3]):
            log('  WARNING {}: bar sign flips differ from the first recording'.format(m.get('session_id')))
        X = np.load(os.path.join(ed, 'X_common.npy'))
        vu = np.load(os.path.join(ed, 'valid_units.npy'))
        n = len(vu)
        d = {'recording': m.get('session_id', os.path.basename(f)), 'folder': f,
             'mouse_line': m.get('mouse_line', ''), 'date': m.get('recording_date', ''), 'unit': vu}
        for k in PER_UNIT:
            p = os.path.join(ed, k + '.npy')
            d[k] = np.load(p) if os.path.exists(p) else np.full(n, np.nan)
        rows.append(pd.DataFrame(d))
        Xs.append(X)
        bp = os.path.join(ed, 'bar_dir_common.npy')
        bars.append(np.load(bp) if os.path.exists(bp) else np.full((n, 4, sizes[3]), np.nan))
        sp = os.path.join(ed, 'speed_shape.npy')
        speeds.append(np.load(sp) if os.path.exists(sp) else np.full((n, 6), np.nan))
        metas.append(m)
        log('  {}: {} units ({} QI-pass)'.format(d['recording'], n, int(np.nansum(d['is_qi_pass']))))
    if not Xs:
        raise ValueError('No exported recordings to pool')
    df = pd.concat(rows, ignore_index=True)
    for k in ('is_qi_pass', 'rf_only', 'rf_fit_ok', 'has_bar_trace', 'has_color_trace', 'color_reliable'):
        df[k] = df[k].fillna(0).astype(bool)
    P = {'X': np.vstack(Xs), 'df': df, 'bar_dir': np.concatenate(bars), 'speed_shape': np.vstack(speeds),
         'sizes': ref[0], 'metas': metas, 'bar_directions': ref[1] or [0, 90, 180, 270]}
    log('Pooled: {} units from {} recordings | {} QI-pass'.format(len(df), len(metas), int(df['is_qi_pass'].sum())))
    return P


def segment_bounds(sizes):
    e = np.cumsum([0] + list(sizes))
    return [(int(e[i]), int(e[i + 1]), SEG_NAMES[i]) for i in range(5)]


def run_pooled(P, n_components=8, k_min=2, k_max=30, n_init_bic=10, n_init_final=20, fixed_k=0, seed=0,
               log=print, progress=None):
    from sklearn.decomposition import SparsePCA
    from sklearn.mixture import GaussianMixture
    X, df = P['X'], P['df']
    q = df['is_qi_pass'].values
    if q.sum() < 4:
        raise ValueError('Only {} QI-pass units in the pooled set'.format(int(q.sum())))
    nc = int(min(n_components, q.sum() - 1, X.shape[1]))
    spca = SparsePCA(n_components=nc, random_state=seed)
    Z = spca.fit_transform(X[q])
    log('SparsePCA: {} QI-pass units -> {} components'.format(int(q.sum()), nc))
    ks = list(range(max(1, k_min), max(k_min, min(k_max, int(q.sum()) - 1)) + 1))
    bics = []
    if fixed_k:
        best = int(fixed_k)
    else:
        for i, k in enumerate(ks):
            g = GaussianMixture(n_components=k, covariance_type='diag', n_init=n_init_bic, random_state=seed)
            g.fit(Z); bics.append(g.bic(Z))
            if progress:
                progress(i + 1, len(ks) + 1)
        best = ks[int(np.argmin(bics))]
        log('BIC-selected k = {}'.format(best))
        if best == ks[-1]:
            log('  WARNING: BIC minimum is at the top of the k range - raise "k max" or set a fixed k')
    gm = GaussianMixture(n_components=best, covariance_type='diag', n_init=n_init_final, random_state=seed)
    lab = np.empty(len(df), dtype=int)
    lab[q] = gm.fit_predict(Z)
    prob = np.zeros(len(df))
    prob[q] = gm.predict_proba(Z).max(axis=1)
    if (~q).sum():
        Zr = spca.transform(X[~q])
        lab[~q] = gm.predict(Zr)
        prob[~q] = gm.predict_proba(Zr).max(axis=1)
    # order clusters by size (largest = 0) so labels are stable to read
    order = np.argsort(-np.bincount(lab[q], minlength=best))
    remap = {int(o): i for i, o in enumerate(order)}
    lab = np.array([remap[int(l)] for l in lab])
    df = df.copy()
    df['cluster'] = lab
    df['cluster_prob'] = prob
    R = {'df': df, 'labels': lab, 'ks': ks if not fixed_k else [], 'bics': bics, 'best_k': best, 'spca': spca,
         'at_k_max': bool(not fixed_k and best == ks[-1]),
         'gmm': gm, 'n_components': nc, 'fixed_k': fixed_k, 'seed': seed}
    for k in range(best):
        m = lab == k
        log('  cluster {:2d}: {:4d} units ({} RF-only) from {} recordings'.format(
            k, int(m.sum()), int((m & ~q).sum()), df.loc[m, 'recording'].nunique()))
    if progress:
        progress(1, 1)
    return R


def save_pooled(rl, P, R, log=print):
    out = rl.out_dir
    if not out:
        raise ValueError('Save the recording list first (its folder holds the pooled results)')
    os.makedirs(out, exist_ok=True)
    df = R['df']
    df.drop(columns=[]).to_csv(os.path.join(out, 'pooled_units.csv'), index=False)
    np.save(os.path.join(out, 'X_pooled.npy'), P['X'])
    np.save(os.path.join(out, 'cluster_labels.npy'), R['labels'])
    meta = {'created': datetime.datetime.now().isoformat(timespec='seconds'),
            'recordings': list(df['recording'].unique()), 'folders': list(df['folder'].unique()),
            'n_units': int(len(df)), 'n_qi_pass': int(df['is_qi_pass'].sum()), 'best_k': int(R['best_k']),
            'fixed_k': int(R['fixed_k'] or 0), 'n_components': int(R['n_components']), 'seed': int(R['seed']),
            'ks': R['ks'], 'bics': [float(b) for b in R['bics']], 'common_sizes': P['sizes']}
    with open(os.path.join(out, 'pooled_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    # per-recording copy of its pooled labels
    for folder, g in df.groupby('folder'):
        try:
            g[['unit', 'cluster', 'cluster_prob', 'is_qi_pass']].to_csv(
                os.path.join(folder, EXPORT, 'pooled_cluster_labels.csv'), index=False)
        except Exception as e:
            log('  could not write labels into {}: {}'.format(folder, e))
    log('Saved pooled results to {}'.format(out))
    return out


def profile(P, R):
    df = R['df']
    rows = []
    sp = P['speed_shape']
    speeds = [10, 20, 25, 30, 40, 50]
    for k in range(R['best_k']):
        m = (df['cluster'] == k).values
        g = df[m]
        ds = g[np.isfinite(g['dsi'])]
        col = g[g['color_reliable']]
        full = m & (df['speed_frac'].values == 1.0)
        rows.append({'cluster': k, 'n': int(m.sum()), 'n_QI': int(g['is_qi_pass'].sum()),
                     'recordings': g['recording'].nunique(),
                     'depth_um': g['depths'].median(), 'rf_diam': g['rf_diameter'].median(),
                     'DSI': ds['dsi'].mean() if len(ds) else np.nan,
                     'frac_DS(p<.05)': (ds['p_dsi'] < 0.05).mean() if len(ds) else np.nan,
                     'OSI': ds['osi'].mean() if len(ds) else np.nan,
                     'colour_opp': col['color_opponency'].mean() if len(col) else np.nan,
                     'pref_speed': speeds[int(np.nanargmax(np.nanmean(sp[full], axis=0)))] if full.sum() else np.nan})
    return pd.DataFrame(rows).set_index('cluster')

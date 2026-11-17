"""
Cell 0C - parse PsychoPy CSVs into condition onsets/offsets (imec time),
plus the grating helpers from Cell 1.
"""
import glob
import os

import numpy as np
import pandas as pd


def align_to_imec(psychopy_ttl, imec_onsets, log=print):
    n = min(len(psychopy_ttl), len(imec_onsets))
    if n < 2:
        log('    WARNING: not enough points to align')
        return psychopy_ttl, 1.0
    scale = (imec_onsets[n - 1] - imec_onsets[0]) / (psychopy_ttl[n - 1] - psychopy_ttl[0])
    aligned = (psychopy_ttl - psychopy_ttl[0]) * scale + imec_onsets[0]
    res = aligned[:n] - imec_onsets[:n]
    log('    scale={:.7f} | res mean={:.2f}ms std={:.2f}ms'.format(
        scale, res.mean() * 1000, res.std() * 1000))
    return aligned, scale


def parse_p1(cfg, folder, imec_onsets, log=print):
    log('  Loading P1...')
    csv_files = glob.glob(os.path.join(folder, '*project_stimulus*.csv'))
    df_full = None
    if csv_files:
        df_full = pd.read_csv(csv_files[0], low_memory=False, encoding='utf-8-sig')
        if 'stimType' not in df_full.columns:
            log('    WARNING: stimType not found in CSV')
            df_full = None
        else:
            log('    CSV loaded: {} rows'.format(len(df_full)))

    if df_full is not None:
        df = df_full[df_full['stimType'].notna() & df_full['ttl_time'].notna()].copy()
        df['ttl_time'] = df['ttl_time'].astype(float)
        df['stim_duration'] = df['stim_duration'].astype(float)
        df = df.sort_values('ttl_time').reset_index(drop=True)
        if len(imec_onsets) >= 2:
            aligned, scale = align_to_imec(df['ttl_time'].values, imec_onsets, log)
        else:
            aligned, scale = df['ttl_time'].values, 1.0
        df['onset_imec'] = aligned
        df['offset_imec'] = aligned + df['stim_duration'].values
        t_ref_py = df['ttl_time'].values[0]
        t_ref_im = df['onset_imec'].values[0]
        cond_on, cond_off = {}, {}

        for stype, name in [('LARGE_MOVE', 'large_sweep'), ('BLINK_WHITE', 'white_blink'),
                            ('BLINK_BLACK', 'black_blink')]:
            m = df['stimType'] == stype
            if m.sum() == 0:
                continue
            cond_on[name] = df.loc[m, 'onset_imec'].values
            cond_off[name] = df.loc[m, 'offset_imec'].values

        offset_deg = float(cfg.loom_azi_offset_deg)
        for stype, base, c_start, c_stop in [
                ('LOOM_BLACK', 'black_loom', 'looming_black_circle_2.started',
                 'looming_black_circle_2.stopped'),
                ('LOOM_WHITE', 'white_loom', 'looming_white_circle_2.started',
                 'looming_white_circle_2.stopped')]:
            m_main = df['stimType'] == stype
            if m_main.sum() == 0 and c_start not in df_full.columns:
                continue
            on_main = df.loc[m_main, 'onset_imec'].values
            off_main = df.loc[m_main, 'offset_imec'].values
            if 'azi' in df.columns:
                azi_main = df.loc[m_main, 'azi'].astype(float).values
            else:
                azi_main = np.zeros(len(on_main))
            n_off = 0
            if c_start in df_full.columns:
                m_off = df_full['stimType'].isna() & df_full[c_start].notna()
                n_off = int(m_off.sum())
            if n_off > 0:
                t_start = df_full.loc[m_off, c_start].astype(float).values
                t_stop = (df_full.loc[m_off, c_stop].astype(float).values
                          if c_stop in df_full.columns else np.full(n_off, np.nan))
                dur = t_stop - t_start
                n_missing = int(np.isnan(dur).sum())
                if n_missing > 0:
                    valid = dur[~np.isnan(dur)]
                    med = np.median(valid) if len(valid) else 0.0
                    log('      WARNING: {} offset rows - {}/{} missing .stopped, imputing '
                        'duration={:.4f}s'.format(base, n_missing, len(dur), med))
                    dur = np.where(np.isnan(dur), med, dur)
                azi_off_base = (df_full.loc[m_off, 'azi'].astype(float).values
                                if 'azi' in df_full.columns else np.zeros(n_off))
                on_off = (t_start - t_ref_py) * scale + t_ref_im
                off_off = on_off + dur
                azi_off = azi_off_base + offset_deg
                all_on = np.concatenate([on_main, on_off])
                all_off = np.concatenate([off_main, off_off])
                all_azi = np.concatenate([azi_main, azi_off])
                log('      {}: {} main + {} offset = {} total'.format(base, len(on_main), n_off, len(all_on)))
            else:
                all_on, all_off, all_azi = on_main, off_main, azi_main
                log('      {}: {} main only'.format(base, len(on_main)))
            idx = np.argsort(all_on)
            all_on, all_off, all_azi = all_on[idx], all_off[idx], all_azi[idx]
            cond_on[base] = all_on
            cond_off[base] = all_off
            for azi_val in np.unique(np.round(all_azi, 1)):
                m = np.abs(all_azi - azi_val) < 0.5
                key = '{}_azi{:+d}'.format(base, int(round(azi_val)))
                cond_on[key] = all_on[m]
                cond_off[key] = all_off[m]

        small = df[df['stimType'] == 'SMALL_MOVE'].copy()
        for name, dur_val in cfg.small_sweep_durations.items():
            m = np.abs(small['stim_duration'].values - float(dur_val)) < 0.1
            if m.sum() == 0:
                continue
            cond_on[name] = small.loc[m, 'onset_imec'].values
            cond_off[name] = small.loc[m, 'offset_imec'].values

        log('    {} main trials -> {} conditions'.format(len(df), len(cond_on)))
        for name, on in sorted(cond_on.items()):
            log('      {:32s}: {:3d} trials | dur={:.3f}s'.format(
                name, len(on), np.mean(cond_off[name] - on)))
        return cond_on, cond_off

    log('  No valid CSV - trying OBX-decoded npy...')
    on_path = os.path.join(cfg.save_dir, 'p1_condition_onsets.npy')
    off_path = os.path.join(cfg.save_dir, 'p1_condition_offsets.npy')
    if os.path.exists(on_path):
        cond_on = np.load(on_path, allow_pickle=True).item()
        cond_off = np.load(off_path, allow_pickle=True).item()
        log('  Loaded from OBX npy: {} conditions'.format(len(cond_on)))
        return cond_on, cond_off
    log('  WARNING: no CSV and no p1_condition_onsets.npy found')
    return {}, {}


def parse_rfm(cfg, seg, rfm_all_edges, log=print):
    log('  Loading {}...'.format(seg))
    empty = ({}, {}, np.array([]), np.array([]), np.array([]))
    if seg not in rfm_all_edges:
        log('    WARNING: no edges for {}'.format(seg))
        return empty
    ft = rfm_all_edges[seg].copy()
    log('    Raw edges: {}'.format(len(ft)))
    if len(ft) > 1:
        gap = (ft[1] - ft[0]) * 1000
        if gap > 150:
            ft = ft[1:]
            log('    Skipped pre-stim trigger (gap={:.0f}ms)'.format(gap))
    n_frames = int(cfg.n_frames_rfm)
    n_use = len(ft)
    if n_use < n_frames:
        log('    WARNING: only {} frames (< {}) - skipping'.format(n_use, n_frames))
        return empty
    fi = np.arange(n_use) % n_frames
    ri = np.arange(n_use) // n_frames
    log('    Using {} frames | {} complete repeats{} | median interval {:.1f}ms'.format(
        n_use, n_use // n_frames,
        (' + {} partial'.format(n_use % n_frames) if n_use % n_frames else ''),
        np.median(np.diff(ft)) * 1000))
    return {'rfm_frames': ft}, {'rfm_frames': ft + 0.1}, ft, fi, ri


GRATING_COLS = ('ttl_time', 'degrees', 'phase')


def _read_grating_csv(folder):
    """The PsychoPy grating log = the CSV that has ttl_time, degrees and phase
    columns (Bonsai *gv*.csv, DLC csv etc. in the same folder are skipped)."""
    csvs = sorted(glob.glob(os.path.join(folder, '*.csv')))
    good = []
    for c in csvs:
        try:
            cols = set(pd.read_csv(c, nrows=2, low_memory=False).columns)
        except Exception:
            continue
        if all(k in cols for k in GRATING_COLS):
            good.append(c)
    if not good:
        return None, csvs
    return pd.read_csv(good[0], low_memory=False), good


def parse_grating(cfg, folder, seg, imec_onsets, log=print):
    log('  Loading {}...'.format(seg))
    df, csvs = _read_grating_csv(folder)
    if df is None:
        log('    WARNING: no CSV with ttl_time/degrees/phase columns in {} (found: {})'.format(
            folder, [os.path.basename(c) for c in csvs])); return {}, {}
    if len(csvs) > 1:
        log('    {} CSVs found - using first ({})'.format(len(csvs), os.path.basename(csvs[0])))
    df = df[df['ttl_time'].notna()].copy()
    df['ttl_time'] = df['ttl_time'].astype(float)
    df['stim_duration'] = (df['grating.stopped'] - df['ttl_time']).fillna(10.0)
    df = df.sort_values('ttl_time').reset_index(drop=True)
    sub = int(cfg.grating_csv_subsample)
    py_sub = df['ttl_time'].values[::sub]
    n = min(len(py_sub), len(imec_onsets))
    if n < 2:
        log('    WARNING: not enough matching onsets ({})'.format(n)); return {}, {}
    scale = (imec_onsets[n - 1] - imec_onsets[0]) / (py_sub[n - 1] - py_sub[0])
    aligned = (df['ttl_time'].values - df['ttl_time'].values[0]) * scale + imec_onsets[0]
    df['onset_imec'] = aligned
    df['offset_imec'] = aligned + df['stim_duration'].values
    log('    {} trials | scale={:.6f} | subsample={} | sizes: {}'.format(
        len(df), scale, sub, [round(float(d), 1) for d in sorted(df['degrees'].unique())]))
    cond_on, cond_off = {}, {}
    for deg in sorted(df['degrees'].unique()):
        m = df['degrees'] == deg
        key = 'grating_{:.1f}deg'.format(deg)
        cond_on[key] = df.loc[m, 'onset_imec'].values
        cond_off[key] = df.loc[m, 'offset_imec'].values
    for sign, label in [(1, 'drift_pos'), (-1, 'drift_neg')]:
        m = np.sign(df['phase'].values) == sign
        cond_on[label] = df.loc[m, 'onset_imec'].values
        cond_off[label] = df.loc[m, 'offset_imec'].values
    cond_on['all_gratings'] = df['onset_imec'].values
    cond_off['all_gratings'] = df['offset_imec'].values
    return cond_on, cond_off


def parse_all(cfg, order, seg_onsets, log=print):
    rfm_all_edges = {}
    for seg in order:
        p = os.path.join(cfg.save_dir, '{}_all_edges_concat_s.npy'.format(seg))
        if os.path.exists(p):
            rfm_all_edges[seg] = np.load(p)
            log('  {}: {} edges'.format(seg, len(rfm_all_edges[seg])))
    all_on, all_off, rfm_data = {}, {}, {}
    for seg in order:
        folder = cfg.seg_psychopy_folder(seg)
        imec_on = seg_onsets.get(seg, np.array([]))
        log('\n{} ({} OBX onsets)'.format(seg, len(imec_on)))
        if seg.startswith('RFM') or seg in rfm_all_edges:
            on, off, ft, fi, ri = parse_rfm(cfg, seg, rfm_all_edges, log)
            if len(ft) > 0:
                rfm_data[seg] = {'frame_times': ft, 'ttl_imec': ft,
                                 'frame_idx': fi, 'repeat_idx': ri}
        elif not os.path.exists(folder):
            log('  WARNING: folder not found - skipping ({})'.format(folder))
            continue
        elif seg == 'P1':
            on, off = parse_p1(cfg, folder, imec_on, log)
        elif seg.startswith('G') and seg[1:].isdigit():
            on, off = parse_grating(cfg, folder, seg, imec_on, log)
        elif seg.startswith('S'):
            log('  Spontaneous - no stimuli'); on, off = {}, {}
        else:
            log('  No parser for {}'.format(seg)); continue
        all_on[seg] = on
        all_off[seg] = off
    np.save(os.path.join(cfg.save_dir, 'all_condition_onsets.npy'), all_on, allow_pickle=True)
    np.save(os.path.join(cfg.save_dir, 'all_condition_offsets.npy'), all_off, allow_pickle=True)
    np.save(os.path.join(cfg.save_dir, 'rfm_data.npy'), rfm_data, allow_pickle=True)
    log('\nSaved all_condition_onsets.npy, all_condition_offsets.npy, rfm_data.npy')
    log('rfm_data keys: {}'.format(list(rfm_data.keys())))
    for seg in order:
        if seg in all_on:
            c = all_on[seg]
            log('  {:6s}: {:2d} conditions | {:5d} trials'.format(
                seg, len(c), sum(len(v) for v in c.values())))
    return all_on, all_off, rfm_data


# ---------------------------------------------------------------------------
# Cell 1 grating helpers
# ---------------------------------------------------------------------------
def load_grating_final(folder, deg, direction):
    df, csvs = _read_grating_csv(folder)
    if df is None:
        raise ValueError('No grating CSV (ttl_time/degrees/phase columns) in {}'.format(folder))
    df = df[df['ttl_time'].notna()].copy()
    df['ttl_time'] = df['ttl_time'].astype(float)
    df['stim_duration'] = (df['grating.stopped'] - df['ttl_time']).fillna(10.0)
    m = (np.isclose(df['degrees'].astype(float), float(deg))) & \
        (df['phase'] > 0 if direction == 'pos' else df['phase'] < 0)
    return df[m].copy()


def align_grating_onsets(cfg, df_filtered, seg, seg_onsets, boundaries, log=print):
    imec = seg_onsets[seg]
    df_all, _ = _read_grating_csv(cfg.seg_psychopy_folder(seg))
    if df_all is None:
        raise ValueError('No grating CSV in {}'.format(cfg.seg_psychopy_folder(seg)))
    df_all = df_all[df_all['ttl_time'].notna()].copy()
    df_all['ttl_time'] = df_all['ttl_time'].astype(float)
    df_all = df_all.sort_values('ttl_time').reset_index(drop=True)
    py_sub = df_all['ttl_time'].values[::int(cfg.grating_csv_subsample)]
    n2 = min(len(py_sub), len(imec))
    scale = (imec[n2 - 1] - imec[0]) / (py_sub[n2 - 1] - py_sub[0])
    aligned = (df_filtered['ttl_time'].values - df_all['ttl_time'].values[0]) * scale + imec[0]
    ok = (aligned >= boundaries[seg]['start_s']) & (aligned < boundaries[seg]['end_s'])
    if not np.all(ok):
        log('  dropping {} out-of-range trials'.format((~ok).sum()))
    return aligned[ok]


def grating_sizes(cfg, seg):
    df, _ = _read_grating_csv(cfg.seg_psychopy_folder(seg))
    if df is None or 'degrees' not in df.columns:
        return []
    return sorted(float(d) for d in df['degrees'].dropna().unique())

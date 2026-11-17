"""
Synchronisation steps of the notebook:
  0A  segment boundaries from per-segment .ap.meta
  0B  TTL / photodiode edge extraction from OBX files
  B1  protocol reference timeline + event labelling
  PD  photodiode-corrected P1 onsets
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from . import npx_io


# ---------------------------------------------------------------------------
# 0A
# ---------------------------------------------------------------------------
def compute_boundaries(cfg, order, log=print):
    boundaries = {}
    t = 0.0
    fs = None
    for seg in order:
        dur, fs = npx_io.get_duration_from_meta(cfg.seg_npx_folder(seg))
        boundaries[seg] = {'start_s': t, 'end_s': t + dur}
        t += dur
    boundaries['fs'] = fs
    for k, v in boundaries.items():
        if k != 'fs':
            log('  {}: {:.1f} -> {:.1f} s'.format(k, v['start_s'], v['end_s']))
    os.makedirs(cfg.save_dir, exist_ok=True)
    with open(os.path.join(cfg.save_dir, 'segment_boundaries.json'), 'w') as f:
        json.dump(boundaries, f, indent=2)
    log('Saved segment_boundaries.json')
    return boundaries


def load_boundaries(save_dir):
    with open(os.path.join(save_dir, 'segment_boundaries.json')) as f:
        b = json.load(f)
    fs = b.pop('fs', 30000.0)
    return b, (fs if fs else 30000.0)


# ---------------------------------------------------------------------------
# 0B
# ---------------------------------------------------------------------------
def extract_ttl_edges(obx_bin_path, fs_obx=None, n_chans=14, analog_ch=4,
                      threshold_v=1.0, min_gap_s=0.05, fallback_fs=30303.0, log=print):
    if fs_obx is None:
        fs_obx = npx_io.get_obx_sample_rate(obx_bin_path, fallback_fs, log)
    raw = npx_io.obx_memmap(obx_bin_path, n_chans)
    n_samp = raw.shape[0]
    xa = raw[:, analog_ch].astype(np.float32) * (5.0 / 32768)
    sync = raw[:, n_chans - 1].astype(np.float32) * (5.0 / 32768)
    above = (xa > threshold_v).astype(np.int8)
    d = np.diff(above)
    rising_raw = np.where(d == 1)[0] + 1
    falling_raw = np.where(d == -1)[0] + 1
    min_gap = int(min_gap_s * fs_obx)
    rising_clean = (rising_raw[np.concatenate([[True], np.diff(rising_raw) > min_gap])]
                    if len(rising_raw) > 1 else rising_raw)
    falling_clean = (falling_raw[np.concatenate([[True], np.diff(falling_raw) > min_gap])]
                     if len(falling_raw) > 1 else falling_raw)
    n = min(len(rising_clean), len(falling_clean))
    all_edges_raw = np.sort(np.concatenate([rising_raw, falling_raw]))
    return {
        'rising_s': rising_clean[:n] / fs_obx,
        'falling_s': falling_clean[:n] / fs_obx,
        'all_edges_s': all_edges_raw / fs_obx,
        'duration_s': n_samp / fs_obx,
        'has_sync': bool(sync.max() > 1.0),
        'fs_obx': fs_obx,
    }


def get_dynamic_threshold(obx_bin_path, analog_ch, n_chans=14):
    ch = npx_io.obx_channel_volts(obx_bin_path, analog_ch, n_chans)
    return float((ch.max() + ch.min()) / 2.0)


def extract_all_onsets(cfg, order, boundaries, log=print):
    obx_files = npx_io.discover_obx_files(cfg, order, log)
    on_c, off_c, edges_c = {}, {}, {}
    overrides = cfg.segment_ttl_overrides or {}
    for name, path in obx_files.items():
        kw = {'analog_ch': cfg.ttl_analog_ch, 'threshold_v': cfg.ttl_threshold_v,
              'min_gap_s': cfg.ttl_min_gap_s}
        kw.update(overrides.get(name, {}))
        if kw.get('threshold_v') == 'dynamic':
            kw['threshold_v'] = get_dynamic_threshold(path, kw['analog_ch'], cfg.obx_n_chans)
            log('  {:6s}: dynamic threshold={:.3f}V on ch{} (override)'.format(
                name, kw['threshold_v'], kw['analog_ch']))
        ev = extract_ttl_edges(path, n_chans=cfg.obx_n_chans,
                               fallback_fs=cfg.obx_fallback_fs, log=log, **kw)
        seg = boundaries[name]
        scale = (seg['end_s'] - seg['start_s']) / ev['duration_s']
        if len(ev['rising_s']) == 0:
            on_c[name] = np.array([]); off_c[name] = np.array([]); edges_c[name] = np.array([])
            log('  {:6s}: 0 onsets - spontaneous'.format(name))
            continue
        on_c[name] = ev['rising_s'] * scale + seg['start_s']
        off_c[name] = ev['falling_s'] * scale + seg['start_s']
        edges_c[name] = ev['all_edges_s'] * scale + seg['start_s']
        log('  {:6s}: {:4d} rising | {:5d} all edges | scale={:.6f}'.format(
            name, len(on_c[name]), len(edges_c[name]), scale))

    for name in [s for s in order if s.startswith('RFM')]:
        e = edges_c.get(name, np.array([]))
        if len(e) == 0:
            log('  {}: no edges'.format(name)); continue
        log('  {}: {} edges | {:.2f}s - {:.2f}s | median interval={:.2f}ms'.format(
            name, len(e), e[0], e[-1], np.median(np.diff(e)) * 1000))

    segs_on = [on_c[n] for n in order if len(on_c.get(n, [])) > 0]
    segs_off = [off_c[n] for n in order if len(off_c.get(n, [])) > 0]
    flat_on = np.sort(np.concatenate(segs_on)) if segs_on else np.array([])
    flat_off = np.sort(np.concatenate(segs_off)) if segs_off else np.array([])
    np.save(os.path.join(cfg.save_dir, 'stim_onsets_concat_s.npy'), flat_on)
    np.save(os.path.join(cfg.save_dir, 'stim_offsets_concat_s.npy'), flat_off)
    for name in edges_c:
        if name.startswith('RFM'):
            np.save(os.path.join(cfg.save_dir, '{}_all_edges_concat_s.npy'.format(name)),
                    edges_c[name])
            log('  Saved {}_all_edges_concat_s.npy: {} edges'.format(name, len(edges_c[name])))
    np.save(os.path.join(cfg.save_dir, 'onsets_by_segment.npy'),
            {'onsets': on_c, 'offsets': off_c, 'obx_files': obx_files}, allow_pickle=True)
    log('Total onsets: {} | Saved'.format(len(flat_on)))
    return on_c, off_c, edges_c, obx_files


def onsets_by_segment_from_flat(save_dir, boundaries):
    flat = np.load(os.path.join(save_dir, 'stim_onsets_concat_s.npy'), allow_pickle=True)
    out = {}
    for seg, b in boundaries.items():
        m = (flat >= b['start_s']) & (flat < b['end_s'])
        out[seg] = flat[m]
    return out, flat


# ---------------------------------------------------------------------------
# B1 protocol
# ---------------------------------------------------------------------------
def build_expected_rising_times(
        N_REPEATS=5, INNER_REPS_SHORT=2,
        STEP_PRE_DUR=0.5, STEP_DUR=1.0, STEP_GAP=0.5,
        PRE_FREQ_DUR=0.5, FREQ_CHIRP_DUR=8.0, POST_FREQ_DUR=0.5,
        PRE_CONTR_DUR=0.5, CONTR_CHIRP_DUR=8.0, POST_CONTR_DUR=0.5,
        BAR_DIRECTIONS=(0, 90, 180, 270), BAR_DUR=3.85, BAR_GAP=0.5,
        COLOR_PRE_DUR=0.5, COLOR_DUR=1.0, COLOR_GAP=0.5, N_COLORS=2,
        INTER_REPEAT_DUR=4.0):
    t = 0.0
    rising, labels = [], []
    for rep in range(N_REPEATS):
        for trial in range(INNER_REPS_SHORT):
            t += STEP_PRE_DUR
            rising.append(t); labels.append('rep{}_step{}_ON'.format(rep, trial))
            t += STEP_DUR + STEP_GAP
            rising.append(t); labels.append('rep{}_step{}_OFF'.format(rep, trial))
            t += STEP_DUR + STEP_GAP
        t += PRE_FREQ_DUR
        rising.append(t); labels.append('rep{}_freq_chirp'.format(rep))
        t += FREQ_CHIRP_DUR + POST_FREQ_DUR
        t += PRE_CONTR_DUR
        rising.append(t); labels.append('rep{}_contr_chirp'.format(rep))
        t += CONTR_CHIRP_DUR + POST_CONTR_DUR
        for trial in range(INNER_REPS_SHORT):
            for d in BAR_DIRECTIONS:
                rising.append(t); labels.append('rep{}_bar{}_dir{}'.format(rep, trial, d))
                t += BAR_DUR + BAR_GAP
        for trial in range(INNER_REPS_SHORT):
            t += COLOR_PRE_DUR
            for c in range(N_COLORS):
                rising.append(t); labels.append('rep{}_colour{}_c{}'.format(rep, trial, c))
                t += COLOR_DUR + COLOR_GAP
        if rep < N_REPEATS - 1:
            t += INTER_REPEAT_DUR
    return np.array(rising), labels


def b1_expected(cfg):
    return build_expected_rising_times(
        N_REPEATS=cfg.b1_n_repeats, INNER_REPS_SHORT=cfg.b1_inner_reps,
        BAR_DIRECTIONS=tuple(cfg.b1_bar_directions), BAR_DUR=cfg.b1_bar_dur,
        N_COLORS=len(cfg.b1_color_names), INTER_REPEAT_DUR=cfg.b1_inter_repeat)


def label_b1_events(real_rising_s, cfg, log=print):
    """Cell 11 version (blue/green, trailing extra edge trimmed)."""
    real = np.asarray(real_rising_s)
    _, expected_labels = b1_expected(cfg)
    if len(real) == len(expected_labels) + 1:
        log('  B1: {} edges vs {} expected - dropping trailing edge'.format(
            len(real), len(expected_labels)))
        real = real[:-1]
    if len(real) != len(expected_labels):
        raise ValueError('Length mismatch: {} real B1 edges vs {} expected labels - '
                         'cannot safely pair them. Re-check edge extraction.'.format(
                             len(real), len(expected_labels)))
    colors = list(cfg.b1_color_names)
    rows = []
    for t, label in zip(real, expected_labels):
        parts = label.split('_')
        rep = int(parts[0].replace('rep', ''))
        second = parts[1]
        if second.startswith('step'):
            routine = 'step'; trial = int(second.replace('step', '')); cond = parts[2]
        elif second == 'freq':
            routine = 'freq_chirp'; trial = 0; cond = ''
        elif second == 'contr':
            routine = 'contr_chirp'; trial = 0; cond = ''
        elif second.startswith('bar'):
            routine = 'bar'; trial = int(second.replace('bar', ''))
            cond = parts[2].replace('dir', '') + 'deg'
        elif second.startswith('colour'):
            routine = 'colour'; trial = int(second.replace('colour', ''))
            cond = colors[int(parts[2].replace('c', ''))]
        else:
            routine = second; trial = 0; cond = '_'.join(parts[2:])
        rows.append({'onset_s': t, 'repeat': rep, 'routine': routine,
                     'trial_in_repeat': trial, 'condition': cond, 'label_raw': label})
    return pd.DataFrame(rows)


def run_b1_labelling(cfg, b1_onsets, log=print):
    df = label_b1_events(b1_onsets, cfg, log)
    log('{} labeled B1 events'.format(len(df)))
    log(str(df['routine'].value_counts().to_dict()))
    df.to_csv(os.path.join(cfg.save_dir, 'B1_photodiode_events_labeled.csv'), index=False)
    log('Saved B1_photodiode_events_labeled.csv')
    return df


def b1_scale(df_b1, cfg):
    exp_times, _ = b1_expected(cfg)
    real_norm = df_b1['onset_s'].values - df_b1['onset_s'].values[0]
    exp_norm = exp_times - exp_times[0]
    n = min(len(real_norm), len(exp_norm))
    slope, intercept = np.polyfit(exp_norm[:n], real_norm[:n], 1)
    return float(slope), float(intercept)


# ---------------------------------------------------------------------------
# PD correction
# ---------------------------------------------------------------------------
def detect_pd_onset_peaks(pd_signal, ttl_t, fs, search_window_s=0.25,
                          baseline_window_s=0.2, amp_threshold_mult=1.5, distance_ms=8.0):
    base_start = max(0, int((ttl_t - baseline_window_s) * fs))
    base_end = int(ttl_t * fs)
    base_sig = np.asarray(pd_signal[base_start:base_end])
    dist = max(1, int(distance_ms * 1e-3 * fs))
    if len(base_sig) < 3:
        return np.nan
    bp, _ = find_peaks(base_sig, distance=dist)
    bt, _ = find_peaks(-base_sig, distance=dist)
    if len(bp) < 2 or len(bt) < 2:
        return np.nan
    thr = (np.median(base_sig[bp]) - np.median(base_sig[bt])) * amp_threshold_mult
    s0 = int(ttl_t * fs)
    s1 = min(int((ttl_t + search_window_s) * fs), len(pd_signal))
    sig = np.asarray(pd_signal[s0:s1])
    if len(sig) < 3:
        return np.nan
    pk, _ = find_peaks(sig, distance=dist)
    tr, _ = find_peaks(-sig, distance=dist)
    if len(pk) == 0 or len(tr) == 0:
        return np.nan
    for p in pk:
        nt = tr[np.argmin(np.abs(tr - p))]
        if sig[p] - sig[nt] > thr:
            return (s0 + p) / fs
    return np.nan


def run_pd_correction(cfg, seg_onsets, boundaries, obx_files, log=print):
    seg = cfg.pd_segment
    path = obx_files[seg]
    fs_obx = npx_io.get_obx_sample_rate(path, cfg.obx_fallback_fs, log)
    raw = npx_io.obx_memmap(path, cfg.obx_n_chans)
    pd_sig = raw[:, cfg.pd_analog_ch].astype(np.float32) * (5.0 / 32768)
    b = boundaries[seg]
    scale = (b['end_s'] - b['start_s']) / (len(pd_sig) / fs_obx)
    raw_on = seg_onsets[seg]
    ttl_obx = (raw_on - b['start_s']) / scale
    pd_obx = np.array([detect_pd_onset_peaks(
        pd_sig, t, fs_obx, cfg.pd_search_window_s, cfg.pd_baseline_window_s,
        cfg.pd_amp_threshold_mult, cfg.pd_distance_ms) for t in ttl_obx])
    pd_imec = np.array([raw_on[i] + (pd_obx[i] - ttl_obx[i]) * scale
                        if np.isfinite(pd_obx[i]) else np.nan
                        for i in range(len(raw_on))])
    delays = (pd_imec - raw_on) * 1000
    v = delays[np.isfinite(delays)]
    log('{} PD detection: {}/{} detected'.format(seg, np.isfinite(delays).sum(), len(delays)))
    if len(v):
        log('  delay mean={:.1f} median={:.1f} std={:.1f} min={:.1f} max={:.1f} ms | NaNs={}'.format(
            v.mean(), np.median(v), v.std(), v.min(), v.max(), np.isnan(delays).sum()))
    np.save(os.path.join(cfg.save_dir, '{}_pd_corrected_onsets_s.npy'.format(seg)), pd_imec)
    log('Saved {}_pd_corrected_onsets_s.npy'.format(seg))
    return pd_imec

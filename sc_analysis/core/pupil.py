"""
Pupil / eye-tracking: screen-video luminance, screen->imec alignment
(grating segments: cell 152; P1: blink+loom matching, cells 155-158),
DLC pupil position + diameter, epoching, OKN measures.
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks, welch
from scipy.stats import linregress


def _cv2():
    import cv2
    return cv2


def seg_files(cfg, seg):
    folder = cfg.seg_psychopy_folder(seg)
    pupil = [p for p in glob.glob(os.path.join(folder, '*{}.avi'.format(seg)))
             if not p.endswith('_screen.avi')]
    screen = glob.glob(os.path.join(folder, '*{}_screen.avi'.format(seg)))
    h5 = glob.glob(os.path.join(folder, '*.h5'))
    return folder, (pupil[0] if pupil else None), (screen[0] if screen else None), (h5[0] if h5 else None)


def frame_count(path):
    cv2 = _cv2()
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def screen_luminance(cfg, seg, screen_path, force=False, log=print, progress=None, cancel=None):
    path = os.path.join(cfg.save_dir, 'screen_luminance_{}.npy'.format(seg))
    if os.path.exists(path) and not force:
        log('  Loading cached luminance ({})'.format(os.path.basename(path)))
        return np.load(path)
    cv2 = _cv2()
    x1, x2, y1, y2 = [int(v) for v in cfg.pupil_crop]
    cap = cv2.VideoCapture(screen_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lum = np.zeros(n, dtype=np.float32)
    log('  Extracting screen luminance ({} frames)...'.format(n))
    for i in range(n):
        ok, fr = cap.read()
        if not ok:
            lum = lum[:i]; break
        lum[i] = fr[y1:y2, x1:x2].mean()
        if progress and i % 200 == 0:
            progress(i, n)
        if cancel is not None and i % 500 == 0 and cancel():
            cap.release()
            raise RuntimeError('Cancelled')
    cap.release()
    np.save(path, lum)
    log('  Saved {}'.format(path))
    return lum


def dlc_pupil(cfg, h5_path, fps):
    df = pd.read_hdf(h5_path)
    scorer = df.columns[0][0]
    thr = float(cfg.pupil_like_thresh)
    bps = [bp for bp in cfg.pupil_bodyparts if (scorer, bp, 'x') in df.columns]
    xa, ya, la = [], [], []
    for bp in bps:
        x = df[(scorer, bp, 'x')].values.astype(float).copy()
        y = df[(scorer, bp, 'y')].values.astype(float).copy()
        l = df[(scorer, bp, 'likelihood')].values.astype(float)
        x[l < thr] = np.nan; y[l < thr] = np.nan
        xa.append(x); ya.append(y); la.append(l)
    xa, ya, la = np.array(xa), np.array(ya), np.array(la)
    with np.errstate(invalid='ignore', divide='ignore'):
        w = np.where(la >= thr, la, np.nan)
        ws = np.nansum(w, axis=0)
        px = np.nansum(xa * w, axis=0) / ws
        py = np.nansum(ya * w, axis=0) / ws
    for arr in (px, py):
        nm = ~np.isfinite(arr)
        if nm.any() and (~nm).any():
            idx = np.arange(len(arr))
            arr[nm] = np.interp(idx[nm], idx[~nm], arr[~nm])

    # diameter (cells 163-165)
    def col(bp, c):
        k = (scorer, bp, c)
        return df[k].values.astype(float) if k in df.columns else None
    dx, dy, dl = col('Dorsal', 'x'), col('Dorsal', 'y'), col('Dorsal', 'likelihood')
    vx, vy, vl = col('Ventral', 'x'), col('Ventral', 'y'), col('Ventral', 'likelihood')
    nx, nl = col('Nasal', 'x'), col('Nasal', 'likelihood')
    tx, tl = col('Temporal', 'x'), col('Temporal', 'likelihood')
    diam = None
    if dx is not None and vx is not None and nx is not None and tx is not None:
        dv = np.sqrt((dx - vx) ** 2 + (dy - vy) ** 2)
        dh = np.abs(nx - tx)
        dv[(dl < thr) | (vl < thr)] = np.nan
        dh[(nl < thr) | (tl < thr)] = np.nan
        with np.errstate(all='ignore'):
            diam = np.nanmean(np.stack([dv, dh]), axis=0)
        nm = ~np.isfinite(diam)
        if nm.any() and (~nm).any():
            idx = np.arange(len(diam))
            diam[nm] = np.interp(idx[nm], idx[~nm], diam[~nm])
    sig = fps * 0.1
    return {'pupil_x': px, 'pupil_y': py,
            'pupil_x_sm': gaussian_filter1d(px, sigma=sig), 'pupil_y_sm': gaussian_filter1d(py, sigma=sig),
            'diam': diam, 'diam_sm': gaussian_filter1d(diam, sigma=fps * 0.05) if diam is not None else None,
            'nan_frac': float(np.mean(~np.isfinite(ws)) if len(ws) else 0)}


def _camera_offset(cfg, folder, slope, intercept, fps_p, fps_s, log):
    gv = [f for f in glob.glob(os.path.join(folder, '*gv*.csv')) if 'screen_gv' not in f]
    sgv = glob.glob(os.path.join(folder, '*screen_gv*.csv'))
    default = float(cfg.pupil_cam_offset_s)
    if not (gv and sgv):
        log('  No Bonsai gv files, using default camera offset {:.1f} ms'.format(default * 1000))
        return default
    try:
        gm = pd.read_csv(gv[0], header=0, names=['frame', 'mean'])['mean'].astype(float).values
        sm = pd.read_csv(sgv[0], header=0, names=['frame', 'mean'])['mean'].astype(float).values
        t_gv = slope * (np.arange(len(gm)) / fps_p) + intercept
        t_sc = slope * (np.arange(len(sm)) / fps_s) + intercept
        gz = (gm - np.median(gm)) / (gm.std() + 1e-6)
        sz = (sm - np.median(sm)) / (sm.std() + 1e-6)
        pg = np.concatenate([find_peaks(np.abs(gz), height=4, distance=max(1, int(fps_p)))[0],
                             find_peaks(-gz, height=4, distance=max(1, int(fps_p)))[0]])
        ps = np.concatenate([find_peaks(np.abs(sz), height=4, distance=max(1, int(fps_s)))[0],
                             find_peaks(-sz, height=4, distance=max(1, int(fps_s)))[0]])
        tg, ts_ = t_gv[pg], t_sc[ps]
        offs = []
        for t in ts_:
            df_ = np.abs(tg - t)
            if len(df_) and df_.min() < 1.0:
                o = tg[np.argmin(df_)] - t
                if abs(o - default) < 0.05:
                    offs.append(o)
        if offs:
            v = float(np.median(offs))
            log('  Camera offset: {:.1f} ms ({} pairs)'.format(v * 1000, len(offs)))
            return v
    except Exception as e:
        log('  Camera offset failed: {}'.format(e))
    log('  Camera offset: using default {:.1f} ms'.format(default * 1000))
    return default


def align_grating_segment(sess, seg, force=False, log=print, progress=None, cancel=None):
    """Cell 152 align_pupil_segment."""
    cfg = sess.cfg
    folder, pupil_avi, screen_avi, h5 = seg_files(cfg, seg)
    log('\n{}\nPupil alignment - {} ({})'.format('=' * 50, seg, folder))
    for name, f in [('pupil AVI', pupil_avi), ('screen AVI', screen_avi), ('DLC h5', h5)]:
        if not f:
            raise FileNotFoundError('No {} in {}'.format(name, folder))
    b = sess.boundaries[seg]
    dur = b['end_s'] - b['start_s']
    n_p, n_s = frame_count(pupil_avi), frame_count(screen_avi)
    fps_p, fps_s = n_p / dur, n_s / dur
    log('  n_pupil={} fps={:.4f} | n_screen={} fps={:.4f}'.format(n_p, fps_p, n_s, fps_s))
    lum = screen_luminance(cfg, seg, screen_avi, force, log, progress, cancel)
    win = max(int(fps_s * 0.5), 3)
    cent = lum - uniform_filter1d(lum, win)
    rms = np.sqrt(uniform_filter1d(cent ** 2, win))
    sus = int(1.0 * fps_s)
    first = None
    above = rms > 0.3
    if len(above) > sus:
        run = np.convolve(above.astype(int), np.ones(sus, dtype=int), 'valid')
        hits = np.nonzero(run == sus)[0]
        first = int(hits[0]) if len(hits) else None
    if first is None:
        log('  WARNING: no onset detected, using frame 0'); first = 0
    log('  First onset frame: {} ({:.2f}s)'.format(first, first / fps_s))
    flat = sess.stim_onsets_flat
    obx = flat[(flat >= b['start_s']) & (flat < b['end_s'])]
    log('  OBX onsets: {}'.format(len(obx)))
    if len(obx) < 2:
        slope, icpt = 1.0, (obx[0] - first / fps_s if len(obx) else b['start_s'])
    else:
        mint = np.mean(np.diff(obx)) * fps_s
        first_imec = first / fps_s + b['start_s']
        nb = int(np.sum(obx < first_imec))
        pred = first + np.arange(-nb, len(obx) - nb) * mint
        sc_t = pred / fps_s
        n = min(len(sc_t), len(obx))
        slope, icpt, r, _, _ = linregress(sc_t[:n], obx[:n])
        res = obx[:n] - (slope * sc_t[:n] + icpt)
        log('  Alignment: R2={:.6f} std={:.1f}ms'.format(r ** 2, res.std() * 1000))
    t_seg = slope * (np.arange(n_p) / fps_p) + icpt
    off = _camera_offset(cfg, folder, slope, icpt, fps_p, fps_s, log)
    t_final = t_seg - off
    np.save(os.path.join(cfg.save_dir, 'pupil_frame_times_imec_{}.npy'.format(seg)), t_final)
    log('  Saved pupil_frame_times_imec_{}.npy  ({:.2f}-{:.2f}s, segment {:.2f}-{:.2f}s)'.format(
        seg, t_final[0], t_final[-1], b['start_s'], b['end_s']))
    return finish_pupil(sess, seg, t_final, fps_p, h5, lum, fps_s, slope, icpt, off, log)


def match_events(screen_t, ttl_times, max_diff_s=5.0):
    used, pairs = set(), []
    for t in ttl_times:
        d = np.abs(screen_t - t)
        for idx in np.argsort(d):
            if idx not in used and d[idx] < max_diff_s:
                pairs.append((screen_t[idx], t)); used.add(idx); break
    return np.array(pairs).reshape(-1, 2)


def align_p1_segment(sess, seg='P1', force=False, white_thresh=128.0, blink_max=0.6, loom_min=1.0,
                     log=print, progress=None, cancel=None):
    """Cells 155-158: white blink + white loom luminance events matched to TTLs."""
    cfg = sess.cfg
    folder, pupil_avi, screen_avi, h5 = seg_files(cfg, seg)
    log('\n{}\nPupil alignment - {} (blink/loom matching)'.format('=' * 50, seg))
    for name, f in [('pupil AVI', pupil_avi), ('screen AVI', screen_avi), ('DLC h5', h5)]:
        if not f:
            raise FileNotFoundError('No {} in {}'.format(name, folder))
    b = sess.boundaries[seg]
    dur = b['end_s'] - b['start_s']
    n_s, n_p = frame_count(screen_avi), frame_count(pupil_avi)
    fps_s, fps_p = n_s / dur, n_p / dur
    lum = screen_luminance(cfg, seg, screen_avi, force, log, progress, cancel)
    above = (lum > white_thresh).astype(np.int8)
    ons = np.where(np.diff(above) == 1)[0] + 1
    offs = np.where(np.diff(above) == -1)[0] + 1
    mind = int(0.1 * fps_s)
    ev = []
    for o in ons:
        k = np.searchsorted(offs, o)
        if k >= len(offs):
            continue
        f = offs[k]
        if f - o < mind:
            continue
        ev.append({'onset_s': o / fps_s, 'dur_s': (f - o) / fps_s})
    ev = ev[1:]  # skip start-up flash
    sb = np.array([e['onset_s'] for e in ev if e['dur_s'] < blink_max])
    sl = np.array([e['onset_s'] for e in ev if e['dur_s'] > loom_min])
    log('  screen events: {} blinks, {} looms'.format(len(sb), len(sl)))
    on = sess.cond_onsets[seg]
    tb = np.sort(on.get('white_blink', []))
    tl = np.sort(on.get('white_loom', []))
    # rough initial offset from the first blink so match window is meaningful
    if len(sb) and len(tb):
        shift = tb[0] - sb[0]
    else:
        shift = b['start_s']
    pb = match_events(sb + shift, tb)
    pl = match_events(sl + shift, tl)
    pairs = np.vstack([pb, pl]) if len(pb) or len(pl) else np.zeros((0, 2))
    if len(pairs) < 2:
        raise ValueError('Could not match enough screen events to TTLs (got {})'.format(len(pairs)))
    o = np.argsort(pairs[:, 1])
    st, it = pairs[o, 0] - shift, pairs[o, 1]
    slope, icpt, r, _, _ = linregress(st, it)
    res = it - (slope * st + icpt)
    log('  {} pairs | slope={:.8f} R2={:.8f} std={:.2f}ms max={:.2f}ms'.format(
        len(st), slope, r ** 2, res.std() * 1000, np.abs(res).max() * 1000))
    off = float(cfg.pupil_cam_offset_s)
    t_final = slope * (np.arange(n_p) / fps_p) + icpt - off
    np.save(os.path.join(cfg.save_dir, 'pupil_frame_times_imec_{}.npy'.format(seg)), t_final)
    log('  Saved pupil_frame_times_imec_{}.npy'.format(seg))
    out = finish_pupil(sess, seg, t_final, fps_p, h5, lum, fps_s, slope, icpt, off, log)
    out['align_pairs'] = (st, it, res)
    return out


def finish_pupil(sess, seg, t_final, fps_p, h5, lum, fps_s, slope, icpt, off, log):
    d = dlc_pupil(sess.cfg, h5, fps_p)
    n = min(len(t_final), len(d['pupil_x']))
    for k in ['pupil_x', 'pupil_y', 'pupil_x_sm', 'pupil_y_sm', 'diam', 'diam_sm']:
        if d.get(k) is not None:
            d[k] = d[k][:n]
    log('  Pupil x: {} frames | NaN {:.1f}% | range {:.1f}-{:.1f}px'.format(
        n, d['nan_frac'] * 100, np.nanmin(d['pupil_x']), np.nanmax(d['pupil_x'])))
    d.update({'seg': seg, 'pupil_imec_t': t_final[:n], 'pupil_fps': fps_p, 'slope': slope,
              'intercept': icpt, 'cam_offset_s': off, 'lum': lum, 'fps_screen': fps_s})
    sess.results.setdefault('pupil', {})[seg] = d
    return d


def load_pupil(sess, seg):
    """Re-create pupil data from saved frame times + DLC (no video reading)."""
    cfg = sess.cfg
    p = os.path.join(cfg.save_dir, 'pupil_frame_times_imec_{}.npy'.format(seg))
    if not os.path.exists(p):
        return None
    t = np.load(p)
    _, _, _, h5 = seg_files(cfg, seg)
    if not h5:
        return None
    b = sess.boundaries[seg]
    fps = len(t) / (b['end_s'] - b['start_s'])
    lp = os.path.join(cfg.save_dir, 'screen_luminance_{}.npy'.format(seg))
    lum = np.load(lp) if os.path.exists(lp) else None
    return finish_pupil(sess, seg, t, fps, h5, lum, None, None, None, None, lambda *a: None)


def get_pupil(sess, seg):
    pdict = sess.results.get('pupil', {})
    if seg in pdict:
        return pdict[seg]
    return load_pupil(sess, seg)


# ---------------------------------------------------------------------------
# epoching
# ---------------------------------------------------------------------------
def grating_trials(sess, seg):
    out = []
    for cond, ons in sess.cond_onsets.get(seg, {}).items():
        if cond.startswith('grating_') and 'deg' in cond:
            deg = float(cond.replace('grating_', '').replace('deg', ''))
            for t in ons:
                out.append({'cond': cond, 'deg': deg, 't_on': float(t)})
    out.sort(key=lambda e: e['t_on'])
    return out


def epoch(pd_, trials, pre_s, post_s, key='pupil_x_sm'):
    t = pd_['pupil_imec_t']
    fps = pd_['pupil_fps']
    tc = np.linspace(-pre_s, post_s, int((pre_s + post_s) * fps))
    eps, info = [], []
    sig = pd_.get(key)
    if sig is None:
        return tc, np.zeros((0, len(tc))), []
    for tr in trials:
        m = (t >= tr['t_on'] - pre_s) & (t <= tr['t_on'] + post_s)
        fr = np.nonzero(m)[0]
        if len(fr) < 5:
            continue
        eps.append(np.interp(tc, t[fr] - tr['t_on'], sig[fr]))
        info.append(tr)
    E = np.array(eps) if eps else np.zeros((0, len(tc)))
    return tc, E, info


def baseline_sub(tc, E):
    if len(E) == 0:
        return E
    b = tc < 0
    return E - E[:, b].mean(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# OKN measures (cells 142-148)
# ---------------------------------------------------------------------------
def okn_psd(pd_, trials, fmin=0.1, fmax=12.0, n_top=3, drift_s=10.0):
    fps = pd_['pupil_fps']
    nper = int(fps * 4)
    t = pd_['pupil_imec_t']
    x = pd_['pupil_x_sm']
    res = {}
    degs = sorted(set(tr['deg'] for tr in trials))
    for deg in degs:
        ps, fax = [], None
        for tr in [tr for tr in trials if tr['deg'] == deg]:
            m = (t >= tr['t_on'] + 0.5) & (t < tr['t_on'] + drift_s - 0.5)
            xs = x[m]
            if len(xs) < nper:
                continue
            ii = np.arange(len(xs))
            xd = xs - np.polyval(np.polyfit(ii, xs, 1), ii)
            f, p = welch(xd, fs=fps, nperseg=nper)
            ps.append(p); fax = f
        if not ps:
            continue
        mp = np.mean(ps, axis=0)
        msk = (fax >= fmin) & (fax <= fmax)
        res[deg] = _top_peaks(fax[msk], mp[msk], n_top, 0.2)
        res[deg]['n'] = len(ps)
    return res


def _top_peaks(f, p, n_top, min_sep_hz):
    if len(f) < 3:
        return {'f': f, 'psd': p, 'freqs': np.array([]), 'pows': np.array([])}
    pk, _ = find_peaks(p, height=p.max() * 0.1, distance=max(1, int(min_sep_hz / (f[1] - f[0]))))
    top = pk[np.argsort(-p[pk])[:n_top]]
    return {'f': f, 'psd': p, 'freqs': f[top], 'pows': p[top]}


def neural_psd(sess, trials, cids, bin_s=0.033, fmin=0.2, fmax=6.0, n_top=3, drift_s=10.0):
    bins = np.arange(0, drift_s + bin_s, bin_s)
    bc = bins[:-1] + bin_s / 2
    fs = 1.0 / bin_s
    res = {}
    for deg in sorted(set(tr['deg'] for tr in trials)):
        tt = [tr for tr in trials if tr['deg'] == deg]
        pop = np.zeros((len(tt), len(bins) - 1))
        for i, tr in enumerate(tt):
            for c in cids:
                st = sess.spike_trains[c]
                s = st[(st >= tr['t_on']) & (st < tr['t_on'] + drift_s)] - tr['t_on']
                cnt, _ = np.histogram(s, bins=bins)
                pop[i] += cnt / bin_s
            pop[i] = pop[i] - np.polyval(np.polyfit(bc, pop[i], 1), bc)
        nper = min(int(fs * 5), pop.size)
        f, p = welch(pop.flatten(), fs=fs, nperseg=nper, noverlap=nper // 2)
        msk = (f >= fmin) & (f <= fmax)
        res[deg] = _top_peaks(f[msk], p[msk], n_top, 0.3)
        res[deg]['n_trials'] = len(tt)
    return res


def okn_reset_rates(pd_, trials, vel_thresh_std=3.0, min_sep=0.5, drift_s=10.0):
    t, x, fps = pd_['pupil_imec_t'], pd_['pupil_x'], pd_['pupil_fps']
    out = {}
    for deg in sorted(set(tr['deg'] for tr in trials)):
        rates = []
        for tr in [tr for tr in trials if tr['deg'] == deg]:
            m = (t >= tr['t_on']) & (t < tr['t_on'] + drift_s)
            xs = x[m]
            if len(xs) < 10:
                continue
            v = np.diff(xs) * fps
            rm = v < (v.mean() - vel_thresh_std * v.std())
            ro = np.where(np.diff(rm.astype(int)) == 1)[0] / fps
            clean = [ro[0]] if len(ro) else []
            for r in ro[1:]:
                if r - clean[-1] >= min_sep:
                    clean.append(r)
            rates.append(len(clean) / drift_s)
        out[deg] = np.array(rates)
    return out


def slow_phase_velocity(pd_, trials, drift_s=10.0):
    t, x, fps = pd_['pupil_imec_t'], pd_['pupil_x'], pd_['pupil_fps']
    out = {}
    for deg in sorted(set(tr['deg'] for tr in trials)):
        vs = []
        for tr in [tr for tr in trials if tr['deg'] == deg]:
            m = (t >= tr['t_on']) & (t < tr['t_on'] + drift_s)
            xs = x[m]
            if len(xs) < 10:
                continue
            v = np.diff(xs) * fps
            slow = v[v > v.mean() - 2 * v.std()]
            if len(slow):
                vs.append(float(np.median(slow)))
        out[deg] = np.array(vs)
    return out


def gaze_corrections(pd_, onsets, speed_dps, cam_deg_per_px, seg_window=None):
    t, x = pd_['pupil_imec_t'], pd_['pupil_x_sm']
    if seg_window is not None:
        m = (t >= seg_window[0]) & (t < seg_window[1])
        med = np.median(x[m]) if m.any() else np.median(x)
    else:
        med = np.median(x)
    corr, gaze = [], []
    for t0 in onsets:
        i = int(np.argmin(np.abs(t - t0)))
        sd = (x[i] - med) * cam_deg_per_px
        corr.append(sd / speed_dps); gaze.append(sd)
    return np.array(corr), np.array(gaze)

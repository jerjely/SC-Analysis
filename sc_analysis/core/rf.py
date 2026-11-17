"""
Cells 7, 8, 1B - spike-triggered averages, 2-D Gaussian RF fits and the RF
mode selector (RFM1 | RFM2 | combined | consistent).
"""
import os

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# screen geometry
# ---------------------------------------------------------------------------
class ScreenGeom(object):
    def __init__(self, cfg):
        self.px_w = int(cfg.screen_px_w)
        self.px_h = int(cfg.screen_px_h)
        self.width_cm = float(cfg.screen_width_cm)
        self.height_cm = self.width_cm * (self.px_h / float(self.px_w))
        self.dist_cm = float(cfg.screen_dist_cm)
        self.width_deg = 2 * np.degrees(np.arctan(self.width_cm / (2 * self.dist_cm)))
        self.height_deg = 2 * np.degrees(np.arctan(self.height_cm / (2 * self.dist_cm)))
        self.deg_per_px_x = self.width_deg / self.px_w
        self.deg_per_px_y = self.height_deg / self.px_h

    def ds_axes(self, ds):
        x_px = np.arange(self.px_w // ds) * ds + ds // 2
        y_px = np.arange(self.px_h // ds) * ds + ds // 2
        return (x_px * self.deg_per_px_x - self.width_deg / 2,
                y_px * self.deg_per_px_y - self.height_deg / 2)

    def extent(self, ds):
        dx, dy = self.ds_axes(ds)
        return [dx.min(), dx.max(), dy.max(), dy.min()]


# ---------------------------------------------------------------------------
# STA
# ---------------------------------------------------------------------------
def load_frames(cfg, log=print):
    frames = np.load(cfg.frames_path, mmap_mode='r')
    ds = int(cfg.sta_ds)
    # frames may be stored at full screen resolution (1600x900) or already small
    frames_ds = np.asarray(frames[:, ::ds, ::ds], dtype=np.float32)
    mean_frame = frames_ds.mean(axis=0)
    log('DS={}: frame={}x{}'.format(ds, frames_ds.shape[1], frames_ds.shape[2]))
    return frames_ds, mean_frame


def compute_sta(spike_times, frame_times, frame_idx, frames_flat, mean_frame_flat, lags_s):
    """Vectorised version of the notebook's compute_sta (identical result):
    for each lag, every spike adds the frame shown at (spike + lag)."""
    n_lags = len(lags_s)
    n_pix = frames_flat.shape[1]
    sta = np.zeros((n_lags, n_pix), dtype=np.float64)
    n_per_lag = np.zeros(n_lags, dtype=np.int64)
    n_fr = frames_flat.shape[0]
    for li, lag in enumerate(lags_s):
        idx = np.searchsorted(frame_times, spike_times + lag, side='right') - 1
        ok = (idx >= 0) & (idx < len(frame_idx))
        if not ok.any():
            continue
        fids = frame_idx[idx[ok]]
        counts = np.bincount(fids, minlength=n_fr).astype(np.float64)
        nz = np.nonzero(counts)[0]
        sta[li] = counts[nz] @ frames_flat[nz]
        n_per_lag[li] = ok.sum()
        sta[li] /= n_per_lag[li]
        sta[li] -= mean_frame_flat
    return sta, n_per_lag


def build_rfm_configs(rfm_data, rfm_good, log=print):
    cfgs = {}
    for seg in rfm_good:
        if seg not in rfm_data or not rfm_data[seg]:
            log('Skipping {} - not in rfm_data'.format(seg)); continue
        ft = rfm_data[seg].get('frame_times', rfm_data[seg].get('ttl_imec', []))
        if len(ft) == 0:
            log('Skipping {} - empty frame_times'.format(seg)); continue
        cfgs[seg] = {'frame_times': np.asarray(ft), 'frame_idx': np.asarray(rfm_data[seg]['frame_idx']),
                     'start_s': ft[0], 'end_s': ft[-1]}
    if len(cfgs) > 1:
        ft = np.concatenate([cfgs[s]['frame_times'] for s in cfgs])
        fi = np.concatenate([cfgs[s]['frame_idx'] for s in cfgs])
        o = np.argsort(ft)
        cfgs['combined'] = {'frame_times': ft[o], 'frame_idx': fi[o],
                            'start_s': ft[o][0], 'end_s': ft[o][-1]}
        log('Combined: {} frames from {}'.format(len(ft), [s for s in cfgs if s != 'combined']))
    return cfgs


def run_sta(cfg, rfm_data, good_ids, spike_trains, overwrite=False, log=print,
            progress=None, cancel=None):
    frames_ds, mean_frame = load_frames(cfg, log)
    H, W = frames_ds.shape[1], frames_ds.shape[2]
    flat = frames_ds.reshape(frames_ds.shape[0], -1)
    mflat = mean_frame.ravel()
    lags = list(cfg.sta_lags_s)
    configs = build_rfm_configs(rfm_data, list(cfg.rfm_good), log)
    for seg in configs:
        fi = configs[seg]['frame_idx']
        if fi.max() >= frames_ds.shape[0]:
            raise ValueError('{}: frame_idx max {} >= n frames {}'.format(seg, fi.max(), frames_ds.shape[0]))
    log('Computing STAs for: {}'.format(list(configs)))
    total = len(configs) * len(good_ids)
    done = 0
    for name, c in configs.items():
        path = os.path.join(cfg.save_dir, 'all_sta_hires_{}.npy'.format(name))
        if os.path.exists(path) and not overwrite:
            log('{}: already exists - skipping (tick "overwrite" to recompute)'.format(name))
            done += len(good_ids)
            continue
        log('\nComputing STA - {} ({} units) | {} frames'.format(name, len(good_ids), len(c['frame_idx'])))
        sta_dict = {}
        for i, cid in enumerate(good_ids):
            if cancel is not None and cancel():
                raise RuntimeError('Cancelled')
            st = spike_trains[cid]
            st = st[(st >= c['start_s']) & (st < c['end_s'])]
            if len(st) < int(cfg.sta_min_spikes):
                sta_dict[cid] = None
            else:
                sta, npl = compute_sta(st, c['frame_times'], c['frame_idx'], flat, mflat, lags)
                sta_dict[cid] = sta.reshape(len(lags), H, W)
            done += 1
            if progress:
                progress(done, total)
        np.save(path, sta_dict, allow_pickle=True)
        log('  Saved {}'.format(path))


# ---------------------------------------------------------------------------
# RF fitting
# ---------------------------------------------------------------------------
def gaussian_2d(xy, amp, x0, y0, sigma_x, sigma_y, offset):
    x, y = xy
    return (offset + amp * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) +
                                    ((y - y0) ** 2) / (2 * sigma_y ** 2)))).ravel()


def sta_snr(sta):
    return float(np.abs(sta).max() / (sta.std() + 1e-8))


def peak_lag_idx(sta):
    return int(np.argmax(np.abs(sta).max(axis=(1, 2))))


def fit_rf_hires(frame_deg, deg_x, deg_y, search_radius=20.0):
    if abs(frame_deg.min()) >= abs(frame_deg.max()):
        data = -frame_deg; polarity = 'OFF'
    else:
        data = frame_deg; polarity = 'ON'
    dp = data - np.median(data)
    pk = np.unravel_index(np.argmax(dp), dp.shape)
    x0i, y0i = deg_x[pk[1]], deg_y[pk[0]]
    amp_i = max(dp[pk], 1e-8)
    X, Y = np.meshgrid(deg_x, deg_y)
    x_lo = max(x0i - search_radius, deg_x.min()); x_hi = min(x0i + search_radius, deg_x.max())
    y_lo = max(y0i - search_radius, deg_y.min()); y_hi = min(y0i + search_radius, deg_y.max())
    try:
        popt, pcov = curve_fit(gaussian_2d, (X.ravel(), Y.ravel()), data.ravel(),
                               p0=[amp_i, x0i, y0i, 5.0, 5.0, np.median(data)],
                               bounds=([0, x_lo, y_lo, 0.5, 0.5, -np.inf],
                                       [np.inf, x_hi, y_hi, 50., 50., np.inf]),
                               maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        amp, x0, y0, sx, sy, off = popt
        ok = (np.all(np.isfinite(popt)) and np.all(np.isfinite(perr)) and amp > 0 and
              0.5 <= sx <= 50 and 0.5 <= sy <= 50 and perr[3] < sx * 2 and perr[4] < sy * 2)
        return popt, polarity, bool(ok)
    except Exception:
        return None, polarity, False


def rf_fit_configs(cfg):
    names = list(cfg.rfm_good)
    if os.path.exists(os.path.join(cfg.save_dir, 'all_sta_hires_combined.npy')):
        names.append('combined')
    return names


def run_rf_fits(cfg, unit_depths, overwrite=False, log=print, progress=None, cancel=None):
    geom = ScreenGeom(cfg)
    lags = list(cfg.sta_lags_s)
    names = rf_fit_configs(cfg)
    log('Fitting RFs for: {}'.format(names))
    for name in names:
        sta_path = os.path.join(cfg.save_dir, 'all_sta_hires_{}.npy'.format(name))
        out_path = os.path.join(cfg.save_dir, 'rf_results_hires_{}.npy'.format(name))
        if not os.path.exists(sta_path):
            log('{}: STA file not found - skipping'.format(name)); continue
        if os.path.exists(out_path) and not overwrite:
            log('{}: RF results already exist - skipping (tick "overwrite" to refit)'.format(name)); continue
        log('\nFitting RFs - {}...'.format(name))
        stas = np.load(sta_path, allow_pickle=True).item()
        res = {}
        n_fit = 0
        first = next((s for s in stas.values() if s is not None), None)
        if first is None:
            log('  no STAs'); continue
        H, W = first.shape[1], first.shape[2]
        ds = int(cfg.sta_ds)
        deg_x, deg_y = geom.ds_axes(ds)
        deg_x, deg_y = deg_x[:W], deg_y[:H]
        items = list(stas.items())
        for i, (cid, sta) in enumerate(items):
            if cancel is not None and cancel():
                raise RuntimeError('Cancelled')
            depth = float(unit_depths.get(cid, np.nan))
            if sta is None:
                res[cid] = {'fit_ok': False, 'snr': 0, 'depth': depth}
                continue
            snr = sta_snr(sta)
            li = peak_lag_idx(sta)
            frame = gaussian_filter(sta[li], sigma=1.5)
            popt, pol, ok = fit_rf_hires(frame, deg_x, deg_y, float(cfg.rf_search_radius))
            if ok and popt is not None:
                amp, x0, y0, sx, sy, off = popt
                res[cid] = {'fit_ok': True, 'x_deg': float(x0), 'y_deg': float(y0),
                            'sigma_x': float(sx), 'sigma_y': float(sy),
                            'diameter': float(2 * np.mean([sx, sy])), 'polarity': pol,
                            'snr': float(snr), 'lag_ms': float(lags[li] * 1000), 'depth': depth,
                            'amp': float(amp), 'source': name}
                n_fit += 1
            else:
                res[cid] = {'fit_ok': False, 'snr': float(snr), 'lag_ms': float(lags[li] * 1000),
                            'polarity': pol, 'depth': depth, 'source': name}
            if progress:
                progress(i + 1, len(items))
        log('  Successful fits: {}/{}'.format(n_fit, len(stas)))
        np.save(out_path, res, allow_pickle=True)
        log('  Saved {}'.format(out_path))


# ---------------------------------------------------------------------------
# RF mode selector (1B)
# ---------------------------------------------------------------------------
def load_rf(save_dir, name):
    p = os.path.join(save_dir, 'rf_results_hires_{}.npy'.format(name))
    if os.path.exists(p):
        return np.load(p, allow_pickle=True).item()
    return {}


def passes_quality(r, min_snr, max_diam):
    if not r.get('fit_ok', False):
        return False
    for k in ['x_deg', 'y_deg', 'sigma_x', 'sigma_y', 'diameter', 'snr', 'depth']:
        if not np.isfinite(r.get(k, np.nan)):
            return False
    return r['snr'] >= min_snr and r['diameter'] <= max_diam


def build_rf_clean(cfg, log=print):
    good = list(cfg.rfm_good)
    r1 = load_rf(cfg.save_dir, good[0])
    r2 = load_rf(cfg.save_dir, good[1]) if len(good) > 1 else r1
    rc = load_rf(cfg.save_dir, 'combined')
    mode = cfg.rf_mode
    mn, mx = float(cfg.min_snr), float(cfg.max_rf_diam)
    if mode in ('RFM1', 'RFM2', 'combined'):
        src = {'RFM1': r1, 'RFM2': r2, 'combined': rc}[mode]
        if mode == 'combined' and not rc:
            raise ValueError("RF_MODE='combined' but rf_results_hires_combined.npy not found")
        clean = {cid: dict(r, source=mode, shift_deg=0.0) for cid, r in src.items()
                 if passes_quality(r, mn, mx)}
    elif mode == 'consistent':
        clean = {}
        for cid in r1:
            if cid not in r2:
                continue
            a, b = r1[cid], r2[cid]
            if not passes_quality(a, mn, mx) or not passes_quality(b, mn, mx):
                continue
            dist = np.sqrt((a['x_deg'] - b['x_deg']) ** 2 + (a['y_deg'] - b['y_deg']) ** 2)
            if dist > float(cfg.max_shift_deg):
                continue
            clean[cid] = {
                'fit_ok': True,
                'x_deg': (a['x_deg'] + b['x_deg']) / 2, 'y_deg': (a['y_deg'] + b['y_deg']) / 2,
                'sigma_x': (a['sigma_x'] + b['sigma_x']) / 2, 'sigma_y': (a['sigma_y'] + b['sigma_y']) / 2,
                'diameter': (a['diameter'] + b['diameter']) / 2, 'snr': (a['snr'] + b['snr']) / 2,
                'shift_deg': dist, 'depth': a['depth'], 'polarity': a['polarity'],
                'lag_ms': (a['lag_ms'] + b['lag_ms']) / 2,
                'rfm1_x_deg': a['x_deg'], 'rfm1_y_deg': a['y_deg'],
                'rfm2_x_deg': b['x_deg'], 'rfm2_y_deg': b['y_deg'], 'source': 'consistent'}
        sh = [r['shift_deg'] for r in clean.values()]
        if sh:
            log('  RFM1-RFM2 shift: median={:.2f} max={:.2f} deg'.format(np.median(sh), max(sh)))
    else:
        raise ValueError('Unknown RF mode {!r}'.format(mode))
    log('RF_MODE={} -> {} units'.format(mode, len(clean)))
    np.save(os.path.join(cfg.save_dir, 'rf_results_clean.npy'), clean, allow_pickle=True)
    log('Saved rf_results_clean.npy')
    if clean:
        az = [r['x_deg'] for r in clean.values()]
        el = [r['y_deg'] for r in clean.values()]
        dm = [r['diameter'] for r in clean.values()]
        log('  Azimuth {:.1f}+/-{:.1f} | Elevation {:.1f}+/-{:.1f} | Diameter {:.1f}+/-{:.1f} deg'.format(
            np.median(az), np.std(az), np.median(el), np.std(el), np.median(dm), np.std(dm)))
    return clean

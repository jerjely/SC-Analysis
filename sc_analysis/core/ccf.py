"""Cell C1 - cross-correlograms during the spontaneous segment."""
import os

import numpy as np
import pandas as pd


def compute_ccf_fast(st_a, st_b, max_lag_s=0.050, bin_s=0.001):
    bins = np.arange(-max_lag_s, max_lag_s + bin_s, bin_s)
    lo = np.searchsorted(st_b, st_a - max_lag_s, side='left')
    hi = np.searchsorted(st_b, st_a + max_lag_s, side='right')
    cnt = hi - lo
    tot = int(cnt.sum()) if len(st_a) else 0
    if tot == 0:
        diffs = np.array([])
    else:
        # vectorised gather of every (a, b) pair within +-max_lag
        start = np.repeat(lo, cnt)
        offs = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        diffs = st_b[start + offs] - np.repeat(st_a, cnt)
    counts, _ = np.histogram(diffs, bins=bins)
    return (bins[:-1] + bin_s / 2) * 1000, counts.astype(np.float64)


def ccf_significance(counts, lags_ms, outer=50, inner=10, peak_win_ms=10, sig_z=5.0):
    bm = (((lags_ms >= -outer) & (lags_ms <= -inner)) | ((lags_ms >= inner) & (lags_ms <= outer)))
    if bm.sum() < 5:
        return False, 0.0, np.nan, 0.0
    mu = counts[bm].mean(); sd = counts[bm].std() + 1e-6
    pm = np.abs(lags_ms) <= peak_win_ms
    if pm.sum() == 0:
        return False, 0.0, np.nan, 0.0
    i = np.argmax(counts[pm])
    pv = counts[pm][i]; pl = lags_ms[pm][i]
    z = (pv - mu) / sd
    return z >= sig_z, z, pl, pv


def run_ccf(sess, log=print, progress=None, cancel=None):
    cfg = sess.cfg
    seg = cfg.ccf_segment
    b = sess.boundaries[seg]
    s0, s1 = b['start_s'], b['end_s']
    dur = s1 - s0
    log('{}: {:.1f}s -> {:.1f}s ({:.1f}s)'.format(seg, s0, s1, dur))
    tr = {}
    for cid in sess.good_ids:
        st = sess.spike_trains[cid]
        tr[cid] = st[(st >= s0) & (st < s1)] - s0
    fr = {c: len(tr[c]) / dur for c in tr}
    active = [c for c in sess.good_ids if fr[c] >= float(cfg.ccf_min_fr)]
    log('Units >= {} Hz in {}: {}/{}'.format(cfg.ccf_min_fr, seg, len(active), len(sess.good_ids)))
    ml, bs, z = float(cfg.ccf_max_lag_s), float(cfg.ccf_bin_s), float(cfg.ccf_sig_z)
    bins = np.arange(-ml, ml + bs, bs)
    lags_ms = (bins[:-1] + bs / 2) * 1000
    rfc = sess.rf_clean
    d = sess.unit_depths
    sig, all_z, n_pairs, all_pairs = [], [], 0, []
    for i, a in enumerate(active):
        if cancel is not None and cancel():
            raise RuntimeError('Cancelled')
        sa = tr[a]
        if len(sa) < 10:
            continue
        for j in range(i + 1, len(active)):
            bb = active[j]
            sb = tr[bb]
            if len(sb) < 10:
                continue
            _, c = compute_ccf_fast(sa, sb, ml, bs)
            ok, zz, pl, pv = ccf_significance(c, lags_ms, peak_win_ms=float(cfg.ccf_peak_win_ms), sig_z=z)
            all_z.append(zz); n_pairs += 1
            all_pairs.append((a, bb, zz))
            if ok:
                typ = 'synchronous' if abs(pl) < 5 else ('a->b' if pl > 0 else 'b->a')
                sig.append({'cid_a': a, 'cid_b': bb, 'z_score': zz, 'peak_lag_ms': pl, 'peak_val': pv,
                            'type': typ, 'depth_a': d.get(a, np.nan), 'depth_b': d.get(bb, np.nan),
                            'depth_diff': abs(d.get(a, np.nan) - d.get(bb, np.nan)),
                            'rf_x_a': rfc.get(a, {}).get('x_deg', np.nan),
                            'rf_x_b': rfc.get(bb, {}).get('x_deg', np.nan), 'counts': c})
        if progress:
            progress(i + 1, len(active))
    all_z = np.array(all_z)
    log('Tested: {:,} | Significant: {} ({:.1f}%)'.format(
        n_pairs, len(sig), 100 * len(sig) / max(n_pairs, 1)))
    log('  synchronous: {} | directional: {}'.format(
        sum(1 for p in sig if p['type'] == 'synchronous'), sum(1 for p in sig if p['type'] != 'synchronous')))
    out = os.path.join(cfg.save_dir, 'ccf_{}_{:.0f}ms_peak{:.0f}ms'.format(
        seg.lower(), ml * 1000, float(cfg.ccf_peak_win_ms)))
    os.makedirs(out, exist_ok=True)
    pd.DataFrame([{k: v for k, v in p.items() if k != 'counts'} for p in sig]).to_csv(
        os.path.join(out, 'ccf_significant_pairs.csv'), index=False)
    np.save(os.path.join(out, 'ccf_sig_pairs.npy'), sig, allow_pickle=True)
    np.save(os.path.join(out, 'ccf_all_z.npy'), all_z)
    log('Saved to {}'.format(out))
    r = {'sig_pairs': sig, 'all_z': all_z, 'n_pairs': n_pairs, 'lags_ms': lags_ms, 'bin_s': bs,
         'sig_z': z, 'dir': out}
    sess.results['ccf'] = r
    return r


def load_ccf(sess):
    cfg = sess.cfg
    out = os.path.join(cfg.save_dir, 'ccf_{}_{:.0f}ms_peak{:.0f}ms'.format(
        cfg.ccf_segment.lower(), float(cfg.ccf_max_lag_s) * 1000, float(cfg.ccf_peak_win_ms)))
    p = os.path.join(out, 'ccf_sig_pairs.npy')
    if not os.path.exists(p):
        return None
    sig = list(np.load(p, allow_pickle=True))
    all_z = np.load(os.path.join(out, 'ccf_all_z.npy'))
    ml, bs = float(cfg.ccf_max_lag_s), float(cfg.ccf_bin_s)
    bins = np.arange(-ml, ml + bs, bs)
    r = {'sig_pairs': sig, 'all_z': all_z, 'n_pairs': len(all_z), 'lags_ms': (bins[:-1] + bs / 2) * 1000,
         'bin_s': bs, 'sig_z': float(cfg.ccf_sig_z), 'dir': out}
    sess.results['ccf'] = r
    return r

"""
Data side of the interactive peak pickers (cells P1 and P2): PSTH prep,
RF-edge timing, result compilation and safe JSON/CSV saving.  The GUI widget
only draws and collects clicks.
"""
import json
import os
import shutil

import numpy as np
import pandas as pd

from .psth import trial_counts, smoothed_trials, raster


def _conv(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError('Not serializable: {}'.format(type(o)))


def load_picks(path, log=print):
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            c = f.read().strip()
        return json.loads(c) if c else {}
    except json.JSONDecodeError as e:
        log('Corrupted JSON ({}) - backed up, starting fresh'.format(e))
        shutil.copy(path, path + '.corrupted')
        return {}


def save_picks(all_picks, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(all_picks, f, indent=2, default=_conv)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# blink / loom picker
# ---------------------------------------------------------------------------
def blink_loom_conditions(sess):
    avail = set(sess.conditions('P1'))
    return [tuple(c) for c in sess.cfg.blink_loom_conditions if c[0] in avail]


def blink_loom_data(sess, cid, pre_s=0.5):
    bin_s = float(sess.cfg.bin_s)
    st = sess.spike_trains[cid]
    data = {}
    for cond, label, col in blink_loom_conditions(sess):
        on, off = sess.onsets('P1', cond)
        dur = float(np.nanmean(off - on))
        post = dur + 1.0
        sig = float(sess.cfg.sigma_blink) if 'blink' in cond else float(sess.cfg.sigma_loom)
        bins = np.arange(-pre_s, post + bin_s, bin_s)
        bc = bins[:-1] + bin_s / 2
        sm = smoothed_trials(trial_counts(st, on, bins), bin_s, sig)
        n = len(on)
        data[cond] = {'mean_fr': sm.mean(axis=0), 'sem_fr': sm.std(axis=0) / np.sqrt(max(n, 1)),
                      'bc': bc, 'stim_dur': dur, 'n': n, 'post_s': post, 'label': label, 'col': col,
                      'raster': raster(st, on, pre_s, post)}
    return data


def compile_blink_loom(sess, cid, data, picks):
    res = {'cid': int(cid), 'depth': int(np.nan_to_num(sess.unit_depths.get(cid, 0)))}
    conds = [c for c, _, _ in blink_loom_conditions(sess)]
    for cond in conds:
        d = data[cond]
        clicked = sorted(picks.get(cond, []))
        dur_ms = d['stim_dur'] * 1000
        bc_ms = d['bc'] * 1000
        res[cond + '_n_picks'] = len(clicked)
        for k, x in enumerate(clicked, start=1):
            i = int(np.argmin(np.abs(bc_ms - x)))
            res['{}_p{}_ms'.format(cond, k)] = x
            res['{}_p{}_fr'.format(cond, k)] = float(d['mean_fr'][i])
            res['{}_p{}_lat_on'.format(cond, k)] = x
            res['{}_p{}_lat_off'.format(cond, k)] = x - dur_ms
        on_pk = [x for x in clicked if x - dur_ms < -100]
        off_pk = [x for x in clicked if x - dur_ms >= -100]
        res[cond + '_class'] = ('ON-OFF' if on_pk and off_pk else 'ON' if on_pk else
                                'OFF' if off_pk else 'NONE')
    cl = [res[c + '_class'] for c in conds]
    if 'ON-OFF' in cl or ('ON' in cl and 'OFF' in cl):
        ov = 'ON-OFF'
    elif 'ON' in cl:
        ov = 'ON'
    elif 'OFF' in cl:
        ov = 'OFF'
    else:
        ov = 'NONE'
    res['overall_class'] = ov
    return res


def write_blink_loom_csv(sess, all_picks):
    if not all_picks:
        return None
    df = pd.DataFrame(list(all_picks.values()))
    p = sess.p('blink_loom_picks.csv')
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# sweep picker
# ---------------------------------------------------------------------------
def rf_left_edge_timing(rf_x_deg, rf_diam, spd, sweep_start_deg, sweep_end_deg, circle_diam_deg):
    rf_left = float(rf_x_deg) - float(rf_diam) / 2
    r = circle_diam_deg / 2
    sd = np.sign(sweep_end_deg - sweep_start_deg)
    s = sd * float(spd)
    if sd > 0:
        lead = (rf_left - sweep_start_deg - r) / s
        trail = (rf_left - sweep_start_deg + r) / s
    else:
        lead = (rf_left - sweep_start_deg + r) / s
        trail = (rf_left - sweep_start_deg - r) / s
    return {'rf_left': rf_left, 'leading_ms': max(0.0, lead * 1000), 'trailing_ms': max(0.0, trail * 1000),
            'leading_raw_ms': lead * 1000, 'clamped': lead * 1000 < 0}


def sweep_cell_list(sess):
    rfc = sess.rf_clean
    cd = float(sess.cfg.circle_diam_deg)
    out = []
    for u in sess.sweep_cells:
        cid = u['cid']
        if cid in rfc:
            rf = rfc[cid]
            out.append({'cid': cid, 'rf_x_deg': float(rf['x_deg']), 'rf_diam': float(rf['diameter']),
                        'depth': float(rf.get('depth', u['depth'])), 'base_fr': float(u['base_fr']),
                        'peak_t': float(u['peak_t'])})
        else:
            out.append({'cid': cid, 'rf_x_deg': float(u['rf_pos_deg']), 'rf_diam': cd * 2,
                        'depth': float(u['depth']), 'base_fr': float(u['base_fr']),
                        'peak_t': float(u['peak_t'])})
    out.sort(key=lambda x: x['peak_t'])
    return out


def sweep_bins(sess, pre=0.5, post=11.0):
    b = float(sess.cfg.bin_s)
    bins = np.arange(-pre, post + b, b)
    return bins, bins[:-1] + b / 2


def sweep_data(sess, u, pre=0.5, post=11.0):
    g = sess.sweep_geometry
    bin_s = float(sess.cfg.bin_s)
    bins, bc = sweep_bins(sess, pre, post)
    st = sess.spike_trains[u['cid']]
    sps = sess.cfg.sigma_per_speed
    data = {}
    for cond, spd in sess.speed_dps.items():
        on, off = sess.onsets('P1', cond)
        dur = float(np.mean(off - on))
        tim = rf_left_edge_timing(u['rf_x_deg'], u['rf_diam'], spd, g['sweep_start_deg'],
                                  g['sweep_end_deg'], g['circle_diam_deg'])
        sm = smoothed_trials(trial_counts(st, on, bins), bin_s, float(sps.get(cond, 2.0)))
        n = len(on)
        data[cond] = {'mean_fr': sm.mean(axis=0), 'sem_fr': sm.std(axis=0) / np.sqrt(max(n, 1)),
                      'dur': dur, 'n': n, 'spd': spd, 'bc': bc, **tim}
    return data


def compile_sweep(sess, u, data, picks):
    rfc = sess.rf_clean
    first = next(iter(data.values()))
    res = {'cid': int(u['cid']), 'rf_x_deg': u['rf_x_deg'], 'rf_left_deg': first['rf_left'],
           'rf_diam': u['rf_diam'], 'depth': u['depth'], 'base_fr': u['base_fr'],
           'rf_source': rfc.get(u['cid'], {}).get('source', 'sweep_estimate')}
    for cond, d in data.items():
        spd = d['spd']
        clicked = sorted(picks.get(cond, []))
        t_ms = d['bc'] * 1000
        res['n_picks_{}dps'.format(spd)] = len(clicked)
        res['leading_ms_{}dps'.format(spd)] = d['leading_ms']
        res['trailing_ms_{}dps'.format(spd)] = d['trailing_ms']
        res['clamped_{}dps'.format(spd)] = d['clamped']
        for k, x in enumerate(clicked, start=1):
            i = int(np.argmin(np.abs(t_ms - x)))
            res['p{}_ms_{}dps'.format(k, spd)] = x
            res['p{}_fr_{}dps'.format(k, spd)] = float(d['mean_fr'][i])
            res['p{}_lat_leading_{}dps'.format(k, spd)] = x - d['leading_ms']
            res['p{}_lat_trailing_{}dps'.format(k, spd)] = x - d['trailing_ms']
    return res


def write_sweep_csv(sess, all_picks):
    if not all_picks:
        return None
    df = pd.DataFrame(list(all_picks.values())).sort_values('rf_left_deg').reset_index(drop=True)
    p = sess.p('interactive_picks.csv')
    df.to_csv(p, index=False)
    return p

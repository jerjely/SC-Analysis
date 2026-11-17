"""
Synthetic SC recording session (same file layout as a real session) used by
the built-in self-test (``sc-analysis --selftest``) and for demos/screenshots.

  <root>/NPX/<prefix>_<SEG>_g0/  SpikeGLX-like meta + OneBox (.obx) files
  <root>/psychopy/<prefix>_<SEG>/ PsychoPy CSVs (+ eye/screen AVIs + DLC h5)
  <root>/frames/sparse_object_frames.npy
  <root>/analysis/ks4/sorter_output/  Kilosort-like output
  <root>/analysis/recording.dat        concatenated binary (first rec_seconds only)

Units are driven by the actual stimuli (RF sparse noise, sweeps, looms,
blinks, B1 battery, gratings) so every analysis has something to find.
"""
import json
import os

import numpy as np
import pandas as pd


def _video_writer(cv2, path, fps, size):
    """Absolute path + codec fallback (macOS OpenCV builds lack some codecs)."""
    path = os.path.abspath(path)
    for code in ('MJPG', 'XVID', 'mp4v'):
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*code), fps, size)
        if w.isOpened():
            return w
        w.release()
    raise RuntimeError('no working video codec for ' + path)


def make_synthetic(ROOT, PREFIX='2026.01.01_DEMO_SC', seed=1, n_units=36, n_frames=1000,
                   rec_seconds=40.0, log=print):
    """Write a synthetic session under ROOT; returns a dict for SessionConfig."""
    print = log  # noqa: A001
    ROOT = os.path.abspath(ROOT)
    rng = np.random.default_rng(seed)
    FS = 30000.0
    FS_OBX = 2000.0
    DT = 0.001                      # rate model resolution
    SCR_W, SCR_H = 160, 90          # sparse-noise frames (screen px in config)
    SCREEN_W_DEG = 2 * np.degrees(np.arctan(42.5 / 30.0))
    SCREEN_H_DEG = 2 * np.degrees(np.arctan(42.5 * SCR_H / SCR_W / 30.0))
    DRIFT = 1.00002                 # psychopy clock vs imec

    segs = {}           # name -> dict(start, dur)
    t_cursor = 0.0


    def add_seg(name, dur):
        nonlocal t_cursor
        segs[name] = {'start': t_cursor, 'dur': dur}
        t_cursor += dur


    # ---------------------------------------------------------------- P1 protocol
    p1_events = []      # (t_rel, type, dur, azi, extra)
    t = 2.0
    for rep in range(4):
        for typ, d in [('BLINK_WHITE', 0.5), ('BLINK_BLACK', 0.5)]:
            p1_events.append((t, typ, d, 0.0, None)); t += d + 1.5
        for typ in ['LOOM_BLACK', 'LOOM_WHITE']:
            p1_events.append((t, typ, 2.0, 0.0, None)); t += 3.5
            p1_events.append((t, typ + '_OFFSET', 2.0, 40.0, None)); t += 3.5   # offset routine (no TTL)
        if rep < 3:
            for spd, d in [(10, 10.0), (20, 5.0), (25, 4.0), (30, 3.333), (40, 2.5), (50, 2.0)]:
                p1_events.append((t, 'SMALL_MOVE', d, 0.0, spd)); t += d + 1.0
        p1_events.append((t, 'LARGE_MOVE', 4.0, 0.0, 25)); t += 5.0
    P1_DUR = t + 3.0
    add_seg('P1', P1_DUR)

    # ---------------------------------------------------------------- RFM
    N_FR = int(n_frames)
    FRAME_DT = 0.1
    frames = np.full((N_FR, SCR_H, SCR_W), 127, np.uint8)
    for i in range(N_FR):
        for _ in range(12):
            x0 = rng.integers(0, SCR_W - 10); y0 = rng.integers(0, SCR_H - 10)
            frames[i, y0:y0 + 10, x0:x0 + 10] = 255 if rng.random() < 0.5 else 0
    RFM_DUR = 1.0 + N_FR * FRAME_DT + 2.0
    add_seg('RFM1', RFM_DUR)
    add_seg('RFM2', RFM_DUR)

    # ---------------------------------------------------------------- B1
    from .core.sync import build_expected_rising_times
    B1_SCALE = 1.03
    b1_exp, b1_lab = build_expected_rising_times()
    B1_T0 = 3.0
    b1_on = B1_T0 + b1_exp * B1_SCALE
    B1_DUR = b1_on[-1] + 8 * B1_SCALE + 6.0
    add_seg('B1', B1_DUR)

    # ---------------------------------------------------------------- gratings
    g_trials = []
    t = 2.0
    for rep in range(2):
        for deg in [2.0, 5.0]:
            for ph in [1, -1]:
                g_trials.append((t, deg, ph)); t += 13.0
    G_DUR = t + 2.0
    add_seg('G1', G_DUR)
    add_seg('G2', G_DUR)
    add_seg('S1', 90.0)
    T_TOTAL = t_cursor
    print('total duration {:.1f}s'.format(T_TOTAL), {k: round(v['dur'], 1) for k, v in segs.items()})

    # ---------------------------------------------------------------- units
    N_UNITS = int(n_units)
    units = []
    for u in range(N_UNITS):
        depth = rng.uniform(100, 2000)
        rfx = -20 + depth / 2000 * 50 + rng.normal(0, 3)          # retinotopy
        rfy = rng.normal(0, 8)
        units.append({'depth': depth, 'rfx': rfx, 'rfy': rfy, 'sig': rng.uniform(4, 7),
                      'pol': 1 if rng.random() < 0.5 else -1, 'base': rng.uniform(1, 6),
                      'gain': rng.uniform(25, 60), 'pref_dir': rng.choice([0, 90, 180, 270]),
                      'ds': rng.random() < 0.3, 'col': rng.uniform(-1, 1), 'lat': 0.04 + depth / 2000 * 0.06,
                      'good': rng.random() < 0.85})

    nt = int(T_TOTAL / DT) + 1
    tt = np.arange(nt) * DT


    def gauss_rf(u, x, y):
        return np.exp(-((x - u['rfx']) ** 2 + (y - u['rfy']) ** 2) / (2 * u['sig'] ** 2))


    lum = np.zeros(nt)          # full-field luminance drive (-1..1)
    loc_drive = np.zeros((N_UNITS, nt))
    ds_gain = np.ones((N_UNITS, nt))
    colour = np.zeros(nt)       # +1 green, -1 blue
    deg_x = (np.arange(SCR_W) + 0.5) * SCREEN_W_DEG / SCR_W - SCREEN_W_DEG / 2
    deg_y = (np.arange(SCR_H) + 0.5) * SCREEN_H_DEG / SCR_H - SCREEN_H_DEG / 2
    X, Y = np.meshgrid(deg_x, deg_y)
    rf_masks = [gauss_rf(u, X, Y) for u in units]


    def idx(a, b):
        return slice(int(a / DT), int(b / DT))


    def put(arr, a, vals, add=True):
        i0 = int(a / DT)
        n = min(len(vals), arr.shape[-1] - i0)
        if add:
            arr[..., i0:i0 + n] += vals[:n]
        else:
            arr[..., i0:i0 + n] = vals[:n]


    # P1 drive
    s0 = segs['P1']['start']
    for (tr, typ, d, azi, spd) in p1_events:
        a = s0 + tr
        if typ == 'BLINK_WHITE':
            lum[idx(a, a + d)] = 1.0
        elif typ == 'BLINK_BLACK':
            lum[idx(a, a + d)] = -1.0
        elif typ.startswith('LOOM'):
            sl = idx(a + 0.5, a + d)
            n = sl.stop - sl.start
            r = np.linspace(1, 30, n)
            a_az = azi if 'OFFSET' not in typ else azi - 20.0
            sign = -1 if 'BLACK' in typ else 1
            for k, u in enumerate(units):
                dist = np.hypot(u['rfx'] - a_az, u['rfy'])
                loc_drive[k, sl] += sign * np.clip((r - dist + u['sig']) / u['sig'], 0, 1)
        elif typ in ('SMALL_MOVE', 'LARGE_MOVE'):
            sl = idx(a, a + d)
            n = sl.stop - sl.start
            x = (15 - 50) + np.arange(n) * DT * spd
            w = 2.5 if typ == 'SMALL_MOVE' else 10
            for k, u in enumerate(units):
                loc_drive[k, sl] += -np.exp(-(x - u['rfx']) ** 2 / (2 * (u['sig'] + w) ** 2)) * \
                    np.exp(-u['rfy'] ** 2 / (2 * (u['sig'] + w) ** 2)) * 1.5

    # RFM drive
    for seg in ['RFM1', 'RFM2']:
        a = segs[seg]['start'] + 1.0
        for i in range(N_FR):
            fr = (frames[i].astype(float) - 127) / 128
            sl = idx(a + i * FRAME_DT, a + (i + 1) * FRAME_DT)
            for k, u in enumerate(units):
                loc_drive[k, sl] += (fr * rf_masks[k]).sum() / rf_masks[k].sum() * 3

    # B1 drive
    b0 = segs['B1']['start']
    for t_on, lab in zip(b1_on, b1_lab):
        a = b0 + t_on
        p = lab.split('_')
        if 'step' in p[1]:
            lum[idx(a, a + 1.0 * B1_SCALE)] = 1.0 if p[2] == 'ON' else -1.0
        elif p[1] == 'freq':
            n = int(8 * B1_SCALE / DT); tl = np.arange(n) * DT / B1_SCALE
            put(lum, a, np.sin(2 * np.pi * (0.5 + 0.5 * tl) * tl), add=False)
        elif p[1] == 'contr':
            n = int(8 * B1_SCALE / DT); tl = np.arange(n) * DT / B1_SCALE
            put(lum, a, tl / 8 * np.sin(2 * np.pi * 2 * tl), add=False)
        elif p[1].startswith('bar'):
            d_ = int(p[2].replace('dir', ''))
            n = int(3.85 * B1_SCALE / DT)
            tl = np.arange(n) * DT / B1_SCALE
            proj = tl * 40.0 - 77.0
            for k, u in enumerate(units):
                pr_u = u['rfx'] * np.cos(np.radians(d_)) + (-u['rfy']) * np.sin(np.radians(d_))
                g = 1.0
                if u['ds']:
                    g = 1.0 if d_ == u['pref_dir'] else 0.15
                put(loc_drive[k], a, -1.5 * g * np.exp(-(proj - pr_u) ** 2 / (2 * (u['sig'] + 2) ** 2)))
        elif p[1].startswith('colour'):
            c = int(p[2].replace('c', ''))
            colour[idx(a, a + 1.0 * B1_SCALE)] = -1 if c == 0 else 1
            lum[idx(a, a + 1.0 * B1_SCALE)] = 0.6

    # gratings
    for seg in ['G1', 'G2']:
        g0 = segs[seg]['start']
        for (tr, deg, ph) in g_trials:
            a = g0 + tr
            sl = idx(a, a + 10.0)
            n = sl.stop - sl.start
            tf = {2.0: 1.5, 5.0: 0.6}[deg]
            lum[sl] += 0.5 * np.sin(2 * np.pi * tf * np.arange(n) * DT * ph)

    # rates -> spikes
    spike_times, spike_clusters = [], []
    lag_rf = 0.075
    for k, u in enumerate(units):
        L = int(u['lat'] / DT)
        drv_l = np.roll(lum, L) * u['pol']
        trans = np.clip(np.diff(np.r_[0, drv_l]) * 30, 0, None)
        sustained = np.clip(drv_l, 0, None)
        loc = np.roll(loc_drive[k], int(lag_rf / DT)) * u['pol']
        col = np.roll(colour, L) * u['col']
        rate = u['base'] + u['gain'] * (0.5 * sustained + trans + np.clip(loc, 0, None) + 0.5 * np.clip(col, 0, None))
        # spontaneous synchrony in S1 between neighbours
        n = rng.poisson(rate * DT)
        st = np.repeat(tt, n) + rng.uniform(0, DT, n.sum())
        spike_times.append(st); spike_clusters.append(np.full(len(st), k))
    # shared synchronous events for unit pairs (CCF)
    s1a, s1b = segs['S1']['start'], segs['S1']['start'] + segs['S1']['dur']
    for a_, b_ in [(0, 1), (2, 3), (4, 7)]:
        ev = rng.uniform(s1a, s1b, 600)
        spike_times[a_] = np.r_[spike_times[a_], ev]
        spike_times[b_] = np.r_[spike_times[b_], ev + rng.normal(0.002, 0.0005, len(ev))]
        spike_clusters[a_] = np.full(len(spike_times[a_]), a_)
        spike_clusters[b_] = np.full(len(spike_times[b_]), b_)
    st_all = np.concatenate(spike_times); sc_all = np.concatenate(spike_clusters)
    o = np.argsort(st_all); st_all, sc_all = st_all[o], sc_all[o]
    print('spikes', len(st_all))

    # ---------------------------------------------------------------- write NPX folders
    npx = os.path.join(ROOT, 'NPX')
    for i, (seg, info) in enumerate(segs.items()):
        folder = os.path.join(npx, '{}_{}_g0'.format(PREFIX, seg))
        os.makedirs(folder, exist_ok=True)
        base = os.path.join(folder, '{}_{}_g0_t0'.format(PREFIX, seg))
        with open(base + '.imec0.ap.meta', 'w') as f:
            f.write('imSampRate={}\nfileTimeSecs={:.6f}\nfileCreateTime=2026-01-01T10:{:02d}:00\nnSavedChans=385\n'.format(
                FS, info['dur'], i))
        n = int(info['dur'] * FS_OBX)
        obx = np.zeros((n, 14), np.int16)
        tl = np.arange(n) / FS_OBX
        V = 32768 / 5.0
        pd_amp = np.full(n, 0.05)
        if seg == 'P1':
            for (tr, typ, d, azi, spd) in p1_events:
                if 'OFFSET' in typ:
                    pd_amp[(tl >= tr + 0.03) & (tl < tr + d)] = 0.4
                    continue
                obx[(tl >= tr) & (tl < tr + d), 4] = int(3.3 * V)
                pd_amp[(tl >= tr + 0.03 + rng.uniform(0, 0.02)) & (tl < tr + d)] = 0.4
            obx[:, 0] = ((1.0 + pd_amp * np.sin(2 * np.pi * 60 * tl)) * V).astype(np.int16)
        elif seg.startswith('RFM'):
            lvl = np.zeros(n)
            lvl[tl >= 0.0] = 0
            edges = [0.2] + [1.0 + j * FRAME_DT for j in range(N_FR)]
            state = 0
            for j, e in enumerate(edges):
                state = 1 - state
                lvl[tl >= e] = state
            obx[:, 4] = (lvl * 3.3 * V).astype(np.int16)
        elif seg == 'B1':
            ph = np.zeros(n)
            for t_on in list(b1_on) + [b1_on[-1] + 6.0]:   # + trailing extra edge
                ph[(tl >= t_on) & (tl < t_on + 0.1)] = 1
            obx[:, 0] = ((0.4 + 1.2 * ph) * V).astype(np.int16)
        elif seg.startswith('G'):
            for (tr, deg, phs) in g_trials:
                obx[(tl >= tr) & (tl < tr + 10.0), 4] = int(3.3 * V)
        obx[:, 13] = (((tl % 1.0) < 0.5) * 3.3 * V).astype(np.int16)
        obx.tofile(base + '.obx0.obx.bin')
        with open(base + '.obx0.obx.meta', 'w') as f:
            f.write('obSampRate={}\nfileTimeSecs={:.6f}\nnSavedChans=14\n'.format(FS_OBX, info['dur']))

    # ---------------------------------------------------------------- PsychoPy
    pp = os.path.join(ROOT, 'psychopy')
    for seg in segs:
        os.makedirs(os.path.join(pp, '{}_{}'.format(PREFIX, seg)), exist_ok=True)
    rows = []
    for (tr, typ, d, azi, spd) in p1_events:
        py_t = (segs['P1']['start'] + tr) * DRIFT + 12.345
        if 'OFFSET' in typ:
            base = 'black' if 'BLACK' in typ else 'white'
            r = {'stimType': np.nan, 'ttl_time': np.nan, 'stim_duration': np.nan, 'azi': azi}
            r['looming_{}_circle_2.started'.format(base)] = py_t
            r['looming_{}_circle_2.stopped'.format(base)] = py_t + d if rng.random() > 0.2 else np.nan
            rows.append(r)
        else:
            rows.append({'stimType': typ, 'ttl_time': py_t, 'stim_duration': d, 'azi': azi})
    pd.DataFrame(rows).to_csv(os.path.join(pp, PREFIX + '_P1', 'P1_project_stimulus_2026.csv'), index=False)
    for seg in ['G1', 'G2']:
        rows = []
        for (tr, deg, ph) in g_trials:
            py_t = (segs[seg]['start'] + tr) * DRIFT + 3.0
            rows.append({'ttl_time': py_t, 'grating.stopped': py_t + 10.0, 'degrees': deg, 'phase': ph * 0.5})
        pd.DataFrame(rows).to_csv(os.path.join(pp, '{}_{}'.format(PREFIX, seg), 'grating_2026.csv'), index=False)

    # eye + screen videos with DLC for G2 and P1
    try:
        import cv2
        for seg in ['G2', 'P1']:
            folder = os.path.join(pp, '{}_{}'.format(PREFIX, seg))
            dur = segs[seg]['dur']
            fps = 30.0
            nfr = int(dur * fps)
            sw = _video_writer(cv2, os.path.join(folder, '{}_{}_screen.avi'.format(PREFIX, seg)),
                                 fps, (64, 36))
            pw = _video_writer(cv2, os.path.join(folder, '{}_{}.avi'.format(PREFIX, seg)),
                                 fps, (32, 24))
            s0_ = segs[seg]['start']
            x_eye = np.zeros(nfr)
            for i in range(nfr):
                tg = s0_ + i / fps
                v = int(np.clip(125 + 100 * lum[min(int(tg / DT), nt - 1)], 0, 255))
                img = np.full((36, 64, 3), v, np.uint8)
                if seg == 'G2':
                    img = (img + (np.sin(np.arange(64) / 3 + i * 0.3) * 30)[None, :, None]).clip(0, 255).astype(np.uint8)
                sw.write(img)
                pw.write(np.full((24, 32, 3), 80, np.uint8))
                x_eye[i] = 5 * ((i / fps * 0.8) % 1.0) if abs(lum[min(int(tg / DT), nt - 1)]) > 0.2 else 0
            sw.release(); pw.release()
            cols = pd.MultiIndex.from_product([['DLC_resnet50'], ['Dorsal', 'Ventral', 'Nasal', 'Temporal'],
                                               ['x', 'y', 'likelihood']], names=['scorer', 'bodyparts', 'coords'])
            dat = np.zeros((nfr, len(cols)))
            for j, (bp, dx, dy) in enumerate([('Dorsal', 0, -10), ('Ventral', 0, 10), ('Nasal', -12, 0), ('Temporal', 12, 0)]):
                dat[:, 3 * j] = 100 + x_eye + dx + rng.normal(0, 0.2, nfr)
                dat[:, 3 * j + 1] = 80 + dy + rng.normal(0, 0.2, nfr)
                dat[:, 3 * j + 2] = 0.95
            pd.DataFrame(dat, columns=cols).to_hdf(os.path.join(folder, '{}_{}DLC.h5'.format(PREFIX, seg)), key='df')
        print('videos written')
    except Exception as e:
        print('video step skipped:', e)

    # ---------------------------------------------------------------- frames
    os.makedirs(os.path.join(ROOT, 'frames'), exist_ok=True)
    np.save(os.path.join(ROOT, 'frames', 'sparse_object_frames.npy'), frames)

    # ---------------------------------------------------------------- Kilosort output
    ana = os.path.join(ROOT, 'analysis')
    so = os.path.join(ana, 'ks4', 'sorter_output')
    os.makedirs(so, exist_ok=True)
    NCH = 64
    chpos = np.zeros((NCH, 2))
    for c in range(NCH):
        sh = c // 16
        chpos[c] = [sh * 250 + (c % 2) * 32, (c % 16) * 130 + 15]
    np.save(os.path.join(so, 'channel_positions.npy'), chpos)
    np.save(os.path.join(so, 'channel_map.npy'), np.arange(NCH))
    templates = np.zeros((N_UNITS, 61, NCH), np.float32)
    best = []
    for k, u in enumerate(units):
        cand = np.argsort(np.abs(chpos[:, 1] - u['depth']) + (np.arange(NCH) // 16 != k % 4) * 1e4)
        b = cand[0]; best.append(b)
        w = -np.exp(-((np.arange(61) - 20) / 3.0) ** 2) + 0.3 * np.exp(-((np.arange(61) - 30) / 6.0) ** 2)
        for c in range(NCH):
            dd = np.hypot(*(chpos[c] - chpos[b]))
            templates[k, :, c] = w * np.exp(-dd / 60.0)
    np.save(os.path.join(so, 'templates.npy'), templates)
    np.save(os.path.join(so, 'spike_times.npy'), (st_all * FS).astype(np.uint64))
    np.save(os.path.join(so, 'spike_clusters.npy'), sc_all.astype(np.int32))
    np.save(os.path.join(so, 'spike_templates.npy'), sc_all.astype(np.int32))
    np.save(os.path.join(so, 'amplitudes.npy'), rng.uniform(10, 30, len(st_all)).astype(np.float32))
    pd.DataFrame({'cluster_id': np.arange(N_UNITS), 'group': ['good' if u['good'] else 'mua' for u in units]}).to_csv(
        os.path.join(so, 'cluster_group.tsv'), sep='\t', index=False)
    with open(os.path.join(so, 'params.py'), 'w') as f:
        f.write("dat_path = '/content/drive/MyDrive/recording.dat'\nn_channels_dat = {}\ndtype = 'int16'\n"
                "offset = 0\nsample_rate = {}\nhp_filtered = False\n".format(NCH, FS))
    # concatenated binary: only the first 40 s get data (rest sparse zeros)
    rec = os.path.join(ana, 'recording.dat')
    mm = np.memmap(rec, dtype=np.int16, mode='w+', shape=(int(min(rec_seconds, T_TOTAL) * FS), NCH))
    n40 = int(min(rec_seconds, T_TOTAL) * FS)
    for s in range(0, n40, int(FS)):
        blk = rng.normal(0, 12, (int(FS), NCH)).astype(np.float32)
        m = (st_all >= s / FS) & (st_all < (s + FS) / FS)
        for t_, c_ in zip(st_all[m], sc_all[m]):
            i0 = int(t_ * FS) - s - 20
            if 0 <= i0 < FS - 61:
                blk[i0:i0 + 61] += templates[c_] * 120
        mm[s:s + int(FS)] = blk.astype(np.int16)
    mm.flush(); del mm

    with open(os.path.join(ROOT, 'truth.json'), 'w') as f:
        json.dump({'units': [{k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (np.integer, bool, np.bool_)) else v)
                              for k, v in u.items()} for u in units], 'segs': segs, 'b1_scale': B1_SCALE}, f, indent=1)
    print('done ->', ROOT)
    date, rest = PREFIX.split('_', 1)
    line, region = rest.rsplit('_', 1)
    return {'recording_date': date, 'mouse_line': line, 'recording_region': region,
            'npx_root': os.path.join(ROOT, 'NPX'), 'save_dir': os.path.join(ROOT, 'analysis'),
            'sorter_dir': os.path.join(ROOT, 'analysis', 'ks4', 'sorter_output'),
            'psychopy_root': os.path.join(ROOT, 'psychopy'),
            'frames_path': os.path.join(ROOT, 'frames', 'sparse_object_frames.npy'),
            'recording_bin': os.path.join(ROOT, 'analysis', 'recording.dat'),
            'screen_px_w': 160, 'screen_px_h': 90, 'sta_ds': 1, 'pupil_crop': [0, 64, 0, 36]}

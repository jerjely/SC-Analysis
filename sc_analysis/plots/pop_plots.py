"""Population heatmaps (cells 37, 40, 42, 72, 87)."""
import numpy as np

from ..core import baden, stimuli
from ..core.pickers import rf_left_edge_timing
from ..core.psth import population_matrix, zscore_rows


def _imshow_depth(ax, mat_z, bc, depths, vmin=-4, vmax=4):
    ext = [bc[0] * 1000, bc[-1] * 1000, depths[0], depths[-1]] if len(depths) else None
    return ax.imshow(mat_z, aspect='auto', origin='lower', extent=ext, cmap='RdBu_r', vmin=vmin, vmax=vmax)


def condition_heatmaps(fig, sess, conditions=('P1:large_sweep',), depth_min=0, depth_max=3000,
                       pre_s=0.5, post_extra_s=1.0, sigma=1.5, pd_corrected=False, zmax=4.0):
    """Cells 37 / 40: z-scored population PSTH sorted by depth, one panel per condition."""
    conditions = list(conditions) or []
    if not conditions:
        raise ValueError('Select one or more conditions')
    cids, depths = sess.units_in_depth(depth_min, depth_max)
    if not cids:
        raise ValueError('No units in depth range')
    n = len(conditions)
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1] * n + [0.04])
    b = float(sess.cfg.bin_s)
    txt = []
    im = None
    for k, sc in enumerate(conditions):
        seg, cond = sc.split(':', 1)
        on, off = sess.onsets(seg, cond, pd_corrected=pd_corrected)
        ok = np.isfinite(off)
        dur = float(np.mean(off[ok] - on[ok]))
        bins = np.arange(-pre_s, dur + post_extra_s + b, b)
        bc = bins[:-1] + b / 2
        mz = zscore_rows(population_matrix(sess.spike_trains, cids, on, bins, sigma), (bc >= -pre_s) & (bc < 0))
        ax = fig.add_subplot(gs[0, k])
        im = _imshow_depth(ax, mz, bc, depths, -zmax, zmax)
        ax.axvline(0, color='white', ls='--', lw=1.5)
        ax.axvline(dur * 1000, color='yellow', ls='--', lw=1.5)
        ax.set_xlim(-pre_s * 1000, (dur + post_extra_s) * 1000)
        ax.set_xlabel('Time from onset (ms)')
        if k == 0:
            ax.set_ylabel('Depth (µm)')
        ax.set_title('{}\n{} trials'.format(sc, len(on)), fontsize=9)
        txt.append('{}: {} trials, dur {:.2f}s'.format(sc, len(on), dur))
    fig.colorbar(im, cax=fig.add_subplot(gs[0, -1]), label='z-score')
    fig.suptitle('{} units | depth {}-{}µm{}'.format(len(cids), depth_min, depth_max,
                                                      ' | PD-corrected' if pd_corrected else ''), fontweight='bold')
    name = '_'.join(c.split(':')[1] for c in conditions[:3]) + '_heatmap.png'
    return {'filename': name, 'text': '\n'.join(txt), 'height': 8}


def b1_routines(sess):
    try:
        df = sess.df_b1
        out = []
        for r in ['freq_chirp', 'contr_chirp', 'step', 'bar', 'colour']:
            if r in set(df['routine']):
                out.append(r + ':')
                for c in sorted(df.loc[df['routine'] == r, 'condition'].dropna().astype(str).unique()):
                    if c and c != 'nan':
                        out.append(r + ':' + c)
        return out
    except Exception:
        return []


def b1_heatmap(fig, sess, routine='freq_chirp:', depth_min=0, depth_max=3000, sigma=1.5, pre_s=0.5):
    """Cell 42."""
    T = baden.get_timing(sess)
    r, c = routine.split(':', 1)
    on = T.get(r, c)
    dur = T.nominal[r] * T.scale
    post = dur + 1.0
    b = float(sess.cfg.bin_s)
    bins = np.arange(-pre_s, post + b, b); bc = bins[:-1] + b / 2
    cids, depths = sess.units_in_depth(depth_min, depth_max)
    mz = zscore_rows(population_matrix(sess.spike_trains, cids, on, bins, sigma), (bc >= -pre_s) & (bc < 0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.04])
    ax = fig.add_subplot(gs[0, 0])
    im = _imshow_depth(ax, mz, bc, depths)
    ax.axvline(0, color='white', ls='--', lw=1.5); ax.axvline(dur * 1000, color='yellow', ls='--', lw=1.5, alpha=0.7)
    ax.set_xlim(-pre_s * 1000, post * 1000); ax.set_xlabel('Time from onset (ms)'); ax.set_ylabel('Depth (µm)')
    ax.set_title('B1 {} {!r} | {} trials | {} units | {}-{}µm\n(yellow = estimated offset, B1 ran {:.2f}x nominal)'.format(
        r, c, len(on), len(cids), depth_min, depth_max, T.scale))
    fig.colorbar(im, cax=fig.add_subplot(gs[0, 1]), label='z-score')
    return {'filename': 'B1_{}_{}_heatmap.png'.format(r, c), 'height': 8,
            'text': 'B1 scale factor {:.4f}\n{} trials, est. dur {:.2f}s'.format(T.scale, len(on), dur)}


def grating_heatmap(fig, sess, size_deg=5.0, direction='pos', sigma=3.0):
    from . import unit_plots
    unit_plots._check_dir(sess, direction)
    """Cell 72: G1 vs G2, units shallow -> deep."""
    from .unit_plots import _size
    size_deg = _size(size_deg)
    gsegs = [s for s in sess.order if s.startswith('G') and s[1:].isdigit()][:2]
    dur, pre, post = 10.0, 1.0, 11.0
    b = float(sess.cfg.bin_s)
    bins = np.arange(-pre, post + b, b); bc = bins[:-1] + b / 2
    base = (bc >= -pre) & (bc < 0)
    cids = list(sess.good_ids_by_depth)
    gs = fig.add_gridspec(1, len(gsegs) + 1, width_ratios=[1] * len(gsegs) + [0.04])
    cols = ['steelblue', 'darkorange']
    im = None
    for k, seg in enumerate(gsegs):
        df = stimuli.load_grating_final(sess.cfg.seg_psychopy_folder(seg), size_deg, direction)
        on = stimuli.align_grating_onsets(sess.cfg, df, seg, sess.seg_onsets, sess.boundaries, log=lambda *a: None)
        mz = zscore_rows(population_matrix(sess.spike_trains, cids, on, bins, sigma), base)
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(mz, aspect='auto', origin='lower', extent=[bc[0] * 1000, bc[-1] * 1000, 0, len(cids)],
                       cmap='RdBu_r', vmin=-4, vmax=4)
        ax.axvline(0, color='white', ls='--', lw=1.2)
        ax.set_xlabel('Time from onset (ms)'); ax.set_ylabel('Unit (shallow -> deep)')
        ax.set_xlim(-pre * 1000, post * 1000)
        ax.set_title('{} | {}deg | n={}'.format(seg, size_deg, len(on)), color=cols[k % 2], fontweight='bold')
    fig.colorbar(im, cax=fig.add_subplot(gs[0, -1]), label='z-score')
    return {'filename': 'grating_{}deg_{}_heatmap.png'.format(size_deg, direction), 'height': 9}


def sweep_heatmaps(fig, sess, depth_min=0, depth_max=3000, pre_s=0.5):
    """Cell 87: one panel per speed with leading/trailing RF-edge lines."""
    cids, depths = sess.units_in_depth(depth_min, depth_max)
    sp = sess.speed_dps
    if not sp:
        raise ValueError('No small-sweep conditions in P1')
    g = sess.sweep_geometry
    sps = sess.cfg.sigma_per_speed
    gs = fig.add_gridspec(1, len(sp) + 1, width_ratios=[1] * len(sp) + [0.04])
    b = float(sess.cfg.bin_s)
    rfc = sess.rf_clean
    im = None
    for k, (cond, spd) in enumerate(sp.items()):
        on, off = sess.onsets('P1', cond)
        dur = float(np.mean(off - on)); post = dur + 1.0
        bins = np.arange(-pre_s, post + b, b); bc = bins[:-1] + b / 2
        mz = zscore_rows(population_matrix(sess.spike_trains, cids, on, bins, float(sps.get(cond, 2.0))),
                         (bc >= -pre_s) & (bc < 0))
        ax = fig.add_subplot(gs[0, k])
        im = _imshow_depth(ax, mz, bc, depths)
        ax.axvline(0, color='white', ls='--', lw=1.2); ax.axvline(dur * 1000, color='yellow', ls='--', lw=1.2)
        L, Tr, D = [], [], []
        for cid, dep in zip(cids, depths):
            if cid not in rfc:
                continue
            t = rf_left_edge_timing(rfc[cid]['x_deg'], rfc[cid]['diameter'], spd, g['sweep_start_deg'],
                                    g['sweep_end_deg'], g['circle_diam_deg'])
            L.append(t['leading_ms']); Tr.append(t['trailing_ms']); D.append(dep)
        if D:
            ax.plot(L, D, '-', color='lime', lw=1.2, alpha=0.8, label='leading edge')
            ax.plot(Tr, D, '--', color='yellow', lw=1.0, alpha=0.7, label='trailing edge')
        ax.set_xlim(-pre_s * 1000, post * 1000)
        ax.set_xlabel('Time (ms)')
        if k == 0:
            ax.set_ylabel('Depth (µm)')
            if D:
                ax.legend(fontsize=7, loc='upper left')
        ax.set_title('{} dps\n{} trials'.format(spd, len(on)), color=sess.speed_colors.get(cond), fontweight='bold',
                     fontsize=9)
    fig.colorbar(im, cax=fig.add_subplot(gs[0, -1]), label='z-score')
    fig.suptitle('Sweep PSTH heatmaps by speed | {} units | green = leading edge @ RF-left, '
                 'yellow = trailing edge'.format(len(cids)), fontweight='bold')
    return {'filename': 'sweep_heatmaps_by_speed.png', 'height': 8.5}

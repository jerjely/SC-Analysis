"""Session / stimulus overview figures (cells 16, 20, 33 + PD delays)."""
import numpy as np


SEG_COLORS = {'P': '#1f77b4', 'R': '#2ca02c', 'B': '#9467bd', 'G': '#ff7f0e', 'S': '#7f7f7f'}


def sequence_text(sess):
    lines = ['STIMULUS SEQUENCE', '=' * 60]
    b = sess.boundaries
    on = sess.cond_onsets
    off = sess.cond_offsets
    for seg in sess.order:
        bb = b.get(seg, {})
        dur = bb.get('end_s', 0) - bb.get('start_s', 0)
        lines.append('\n{}  [{:.1f}s -> {:.1f}s  ({:.1f} min)]'.format(seg, bb.get('start_s', 0), bb.get('end_s', 0), dur / 60))
        if seg.startswith('S'):
            lines.append('  (spontaneous - no stimuli)'); continue
        if seg.startswith('RFM'):
            ft = sess.rfm_data.get(seg, {}).get('frame_times', [])
            n = int(sess.cfg.n_frames_rfm)
            lines.append('  RFM frames: {}  ({} complete repeats{})'.format(
                len(ft), len(ft) // n, ' + {} partial'.format(len(ft) % n) if len(ft) % n else ''))
            continue
        if seg == sess.cfg.b1_segment:
            try:
                df = sess.df_b1
                for r, g in df.groupby('routine'):
                    lines.append('  {:32s}  x{:3d}'.format(r, len(g)))
            except Exception:
                lines.append('  (B1 not labelled yet)')
            continue
        conds = on.get(seg, {})
        if not conds:
            lines.append('  (no conditions parsed)'); continue
        summ = sorted([(float(np.min(v)), k, len(v)) for k, v in conds.items() if '_azi' not in k and len(v)])
        for first, name, n in summ:
            d = np.nanmean(np.asarray(off[seg][name]) - np.asarray(conds[name]))
            lines.append('  {:32s}  x{:3d}  first={:.1f}s  dur={:.3f}s'.format(name, n, first, d))
            for ak in sorted(k for k in conds if k.startswith(name + '_azi')):
                lines.append('    └ {:30s}  x{:3d}'.format(ak, len(conds[ak])))
    return '\n'.join(lines)


def timeline(fig, sess):
    b = sess.boundaries
    ax = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    for i, seg in enumerate(sess.order):
        bb = b[seg]
        c = SEG_COLORS.get(seg[0], '#888')
        ax.axvspan(bb['start_s'], bb['end_s'], color=c, alpha=0.25)
        ax.text((bb['start_s'] + bb['end_s']) / 2, 0.92, seg, ha='center', va='top', fontweight='bold', color=c,
                transform=ax.get_xaxis_transform())
    try:
        flat = sess.stim_onsets_flat
        ax.vlines(flat, 0.1, 0.7, color='k', lw=0.3, alpha=0.5)
    except Exception:
        pass
    ax.set_yticks([]); ax.set_xlabel('Concatenated time (s)')
    ax.set_title('Segments and TTL onsets ({} onsets)'.format(len(sess.stim_onsets_flat)))
    ax.set_xlim(0, max(bb['end_s'] for bb in b.values()))
    # P1 conditions raster
    try:
        conds = [c for c in sess.conditions('P1') if '_azi' not in c]
        for i, c in enumerate(conds):
            on = sess.cond_onsets['P1'][c]
            ax2.vlines(on, i, i + 0.8, lw=1)
        ax2.set_yticks(np.arange(len(conds)) + 0.4); ax2.set_yticklabels(conds, fontsize=7)
        ax2.set_xlabel('time (s)'); ax2.set_title('P1 condition onsets')
        bb = b['P1']; ax2.set_xlim(bb['start_s'], bb['end_s'])
    except Exception:
        ax2.axis('off')
    fig.tight_layout()
    try:
        txt = sequence_text(sess)
    except Exception as e:
        txt = str(e)
    return {'filename': 'session_timeline.png', 'text': txt, 'height': 8}


def pd_delays(fig, sess):
    pdo = sess.pd_onsets
    if pdo is None:
        raise ValueError('Run the PD correction step first')
    seg = sess.cfg.pd_segment
    raw = sess.seg_onsets[seg]
    d = (pdo - raw) * 1000
    v = d[np.isfinite(d)]
    a0, a1 = fig.subplots(1, 2)
    a0.hist(v, bins=40, color='steelblue', edgecolor='white')
    a0.set_xlabel('TTL → screen delay (ms)'); a0.set_ylabel('# onsets'); a0.set_title('{} photodiode delays'.format(seg))
    a1.plot(raw, d, '.', ms=3)
    a1.set_xlabel('onset time (s)'); a1.set_ylabel('delay (ms)'); a1.set_title('Delay over the segment')
    fig.tight_layout()
    return {'filename': '{}_pd_delays.png'.format(seg),
            'text': 'detected {}/{}\nmean {:.1f} median {:.1f} std {:.1f}\nmin {:.1f} max {:.1f} ms'.format(
                len(v), len(d), v.mean(), np.median(v), v.std(), v.min(), v.max()) if len(v) else 'none detected'}


def unit_overview(fig, sess):
    """Depth vs firing rate of all units + good-unit counts."""
    sp = sess.spikes
    lab = sp['labels_df'].set_index('cluster_id')['group']
    tmax = sp['t_max'] or 1
    ax0, ax1 = fig.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]})
    for g, c in [('good', 'tab:green'), ('mua', 'tab:orange'), ('noise', 'tab:gray')]:
        ids = [u for u in sp['trains_all'] if lab.get(u, 'unsorted') == g]
        ax0.scatter([len(sp['trains_all'][u]) / tmax for u in ids], [sp['unit_depths'].get(u, np.nan) for u in ids],
                    s=12, c=c, alpha=0.7, label='{} ({})'.format(g, len(ids)))
    ax0.set_xscale('log'); ax0.set_xlabel('mean FR (Hz)'); ax0.set_ylabel('depth (µm)'); ax0.legend(); ax0.grid(alpha=0.2)
    ax0.set_title('Units ({}): {}'.format(sp['label_src'], len(sp['trains_all'])))
    d = [sess.unit_depths[u] for u in sess.good_ids if np.isfinite(sess.unit_depths.get(u, np.nan))]
    ax1.hist(d, bins=30, orientation='horizontal', color='tab:green')
    ax1.set_xlabel('# good units'); ax1.set_title('Good units by depth')
    fig.tight_layout()
    return {'filename': 'unit_overview.png', 'text': '{} good units | depth {:.0f}-{:.0f} µm | {:.1f} s total'.format(
        len(sess.good_ids), min(d) if d else 0, max(d) if d else 0, tmax)}

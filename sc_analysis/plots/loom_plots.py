"""Loom analyses from the blink/loom picker CSV (cells L1-L5, R1)."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats import linregress

DEFAULT_COLORS = ['blue', 'red', 'black', 'gray', 'goldenrod', 'darkorange', 'green', 'purple']


def _df(sess):
    df = sess.picks_df('blink_loom')
    if df is None or not len(df):
        raise ValueError('No blink/loom picks yet - use the Blink/loom picker tab')
    return df


def picked_conditions(sess, only_loom=True):
    try:
        df = sess.picks_df('blink_loom')
        if df is None:
            return []
        c = [col[:-6] for col in df.columns if col.endswith('_class') and col != 'overall_class']
        return [x for x in c if ('loom' in x or not only_loom)]
    except Exception:
        return []


def _sel(df, cond):
    cc, pc = cond + '_class', cond + '_p1_ms'
    if cc not in df.columns or pc not in df.columns:
        return None
    m = df[cc].isin(['ON', 'ON-OFF']) & df[pc].notna()
    s = df[m]
    lat = s[cond + '_p1_lat_on'].values.astype(float)
    dep = s['depth'].values.astype(float)
    fr = s[cond + '_p1_fr'].values.astype(float)
    cid = s['cid'].values.astype(int)
    v = np.isfinite(lat) & np.isfinite(dep) & np.isfinite(fr)
    return cid[v], lat[v], dep[v], fr[v]


def _size_fn(vals, smin=30, smax=400):
    lo, hi = (np.min(vals), np.max(vals)) if len(vals) else (0, 1)
    return lambda f: smin + (f - lo) / (hi - lo) * (smax - smin) if hi > lo else 80.0, lo, hi


def _size_legend(ax, fn, lo, hi, inset=True):
    tgt = ax.inset_axes([0.68, 0.02, 0.30, 0.25]) if inset else ax
    for f in (10, 30, 60, 100):
        if lo <= f <= hi:
            tgt.scatter([], [], s=fn(f), color='gray', edgecolors='black', linewidths=0.6, label='{} Hz'.format(f))
    if tgt.get_legend_handles_labels()[0]:
        tgt.legend(title='Peak FR', fontsize=7, title_fontsize=7, loc='center' if inset else 'lower right')
    if inset:
        tgt.axis('off')


def latency_vs_depth_single(fig, sess, condition='black_loom_azi+0'):
    """Cell L1."""
    r = _sel(_df(sess), condition)
    if r is None or len(r[0]) == 0:
        raise ValueError('{}: no ON/ON-OFF cells'.format(condition))
    cid, lat, dep, fr = r
    fn, lo, hi = _size_fn(fr, 20, 300)
    ax = fig.add_subplot(111)
    sc = ax.scatter(lat, dep, s=[fn(f) for f in fr], c=dep, cmap='viridis', edgecolors='black', linewidths=0.6,
                    zorder=5, alpha=0.85)
    fig.colorbar(sc, ax=ax, label='Depth (µm)')
    _size_legend(ax, fn, lo, hi, inset=False)
    ax.invert_yaxis(); ax.grid(alpha=0.3)
    ax.set_xlabel('Onset latency (ms)'); ax.set_ylabel('Depth (µm)')
    ax.set_title('{} onset latency vs depth\nn={} ON/ON-OFF cells (size = peak FR)'.format(condition, len(lat)),
                 fontweight='bold')
    return {'filename': '{}_onset_latency_vs_depth.png'.format(condition),
            'text': 'n={}\nLatency {:.0f} ± {:.0f} ms\nPeak FR {:.0f} ± {:.0f} Hz\nDepth {:.0f} ± {:.0f} µm'.format(
                len(lat), lat.mean(), lat.std(), fr.mean(), fr.std(), dep.mean(), dep.std())}


def latency_on_rf_map(fig, sess, condition='black_loom_azi+0'):
    """Cell L2."""
    r = _sel(_df(sess), condition)
    if r is None:
        raise ValueError('{}: columns not found'.format(condition))
    rfc = sess.rf_clean
    keep = [i for i, c in enumerate(r[0]) if c in rfc and np.isfinite(rfc[c].get('x_deg', np.nan))
            and np.isfinite(rfc[c].get('y_deg', np.nan))]
    if not keep:
        raise ValueError('{}: no cells with RF + loom response'.format(condition))
    cid = r[0][keep]; lat = r[1][keep]; fr = r[3][keep]
    x = np.array([rfc[c]['x_deg'] for c in cid]); y = np.array([rfc[c]['y_deg'] for c in cid])
    fn, lo, hi = _size_fn(fr)
    ax = fig.add_subplot(111)
    sc = ax.scatter(x, y, s=[fn(f) for f in fr], c=lat, cmap='viridis', edgecolors='black', linewidths=0.6, zorder=5,
                    alpha=0.85)
    fig.colorbar(sc, ax=ax, label='Onset latency (ms)')
    _size_legend(ax, fn, lo, hi, inset=False)
    ax.axhline(0, color='gray', lw=0.8, alpha=0.5); ax.axvline(0, color='gray', lw=0.8, alpha=0.5)
    ax.set_xlabel('Azimuth (deg)'); ax.set_ylabel('Elevation (deg)'); ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.2)
    ax.set_title('{} onset latency on RF map\nn={} (colour = latency, size = peak FR)'.format(condition, len(cid)),
                 fontweight='bold')
    return {'filename': '{}_latency_on_rfmap.png'.format(condition),
            'text': 'n={}\nLatency {:.0f} ± {:.0f} ms\nRF azimuth {:.0f} to {:.0f} deg'.format(
                len(cid), lat.mean(), lat.std(), x.min(), x.max())}


def latency_vs_depth_multi(fig, sess, conditions=(), hulls=False, rf_bars=False, hull_clip=2.5, min_hull=8):
    """Cells L3 / L4 / R1: several loom conditions on one depth plot,
    optional convex hulls (grouped by azimuth suffix) and RF-size bars."""
    df = _df(sess)
    conds = list(conditions) or picked_conditions(sess)
    data = {c: _sel(df, c) for c in conds}
    data = {c: d for c, d in data.items() if d is not None and len(d[0])}
    if not data:
        raise ValueError('No valid ON/ON-OFF loom responses for the selected conditions')
    allfr = np.concatenate([d[3] for d in data.values()])
    fn, lo, hi = _size_fn(allfr)
    ax = fig.add_subplot(111)
    txt = []
    if rf_bars:
        rfc = sess.rf_clean
        az = [r['x_deg'] for r in rfc.values() if np.isfinite(r.get('depth', np.nan)) and np.isfinite(r.get('x_deg', np.nan))]
        dp = [r['depth'] for r in rfc.values() if np.isfinite(r.get('depth', np.nan)) and np.isfinite(r.get('x_deg', np.nan))]
        if len(az) >= 3:
            sl = linregress(dp, az)[0]
            um_per_deg = 1.0 / abs(sl) if sl else np.nan
            txt.append('Retinotopic scale: {:.1f} deg/mm ({:.1f} µm/deg)'.format(abs(sl) * 1000, um_per_deg))
            for k, (c, (cid, lat, dep, fr)) in enumerate(data.items()):
                col = DEFAULT_COLORS[k % len(DEFAULT_COLORS)]
                for i, u in enumerate(cid):
                    rf = rfc.get(u)
                    if rf and rf.get('fit_ok') and np.isfinite(rf.get('diameter', np.nan)):
                        ax.barh(dep[i], width=20, left=lat[i] - 10, height=rf['diameter'] * um_per_deg, color=col,
                                alpha=0.16, linewidth=0, zorder=1)
    if hulls:
        groups = {}
        for c, d in data.items():
            g = c.split('_azi')[-1] if '_azi' in c else c
            groups.setdefault(g, []).append(d)
        hc = ['purple', 'gold', 'teal', 'olive']
        for k, (g, ds) in enumerate(groups.items()):
            x = np.concatenate([d[1] for d in ds]); y = np.concatenate([d[2] for d in ds])
            if len(x) >= min_hull:
                xl, xh = np.percentile(x, [hull_clip, 100 - hull_clip]); yl, yh = np.percentile(y, [hull_clip, 100 - hull_clip])
                m = (x >= xl) & (x <= xh) & (y >= yl) & (y <= yh)
                x, y = x[m], y[m]
            if len(x) < min_hull:
                txt.append('azi {}: not enough points for hull'.format(g)); continue
            P = np.column_stack([x, y])
            try:
                h = ConvexHull(P); poly = P[h.vertices]
                ax.fill(poly[:, 0], poly[:, 1], color=hc[k % 4], alpha=0.15, zorder=1, label='azi {} hull'.format(g))
                ax.plot(np.r_[poly[:, 0], poly[0, 0]], np.r_[poly[:, 1], poly[0, 1]], color=hc[k % 4], lw=2.5, zorder=2)
            except Exception as e:
                txt.append('hull failed: {}'.format(e))
    for k, (c, (cid, lat, dep, fr)) in enumerate(data.items()):
        col = DEFAULT_COLORS[k % len(DEFAULT_COLORS)]
        ax.scatter(lat, dep, s=[fn(f) for f in fr], color=col, edgecolors='black', linewidths=0.6, zorder=5,
                   alpha=0.85, label='{} (n={})'.format(c, len(lat)))
        txt.append('{}: n={} lat={:.0f}±{:.0f}ms depth={:.0f}±{:.0f}µm peak={:.0f}±{:.0f}Hz'.format(
            c, len(lat), lat.mean(), lat.std(), dep.mean(), dep.std(), fr.mean(), fr.std()))
    _size_legend(ax, fn, lo, hi)
    ax.invert_yaxis(); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='upper right')
    ax.set_xlabel('Onset latency (ms)'); ax.set_ylabel('Depth (µm)')
    ax.set_title('Loom onset latency vs depth\nsize = peak FR{}{}'.format(
        ' | hulls by azimuth' if hulls else '', ' | bar height = RF size (µm)' if rf_bars else ''), fontweight='bold')
    name = 'loom_onset_latency_vs_depth{}{}.png'.format('_hulls' if hulls else '', '_RFbars' if rf_bars else '')
    return {'filename': name, 'text': '\n'.join(txt), 'height': 8}


def loom_bars(fig, sess, conditions=()):
    """Cell L5."""
    df = _df(sess)
    conds = list(conditions) or picked_conditions(sess)
    axes = fig.subplots(1, 2)
    rng = np.random.default_rng(1)
    for ax, metric, ylab, title in zip(axes, ['p1_lat_on', 'p1_fr'], ['Onset latency (ms)', 'Peak FR (Hz)'],
                                       ['Onset latency', 'Peak firing rate']):
        means, sems, labs, cols, vals = [], [], [], [], []
        for k, c in enumerate(conds):
            if c + '_class' not in df.columns:
                continue
            m = df[c + '_class'].isin(['ON', 'ON-OFF']) & df[c + '_p1_ms'].notna()
            v = df.loc[m, c + '_' + metric].values.astype(float)
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            means.append(v.mean()); sems.append(v.std() / np.sqrt(len(v)))
            labs.append('{}\n(n={})'.format(c.replace('_loom_', '\n'), len(v)))
            cols.append(DEFAULT_COLORS[k % len(DEFAULT_COLORS)]); vals.append(v)
        x = np.arange(len(means))
        ax.bar(x, means, yerr=sems, color=cols, edgecolor='black', lw=0.8, capsize=6, width=0.55,
               error_kw=dict(elinewidth=1.5, ecolor='black'))
        for i, v in enumerate(vals):
            ax.scatter(np.full(len(v), x[i]) + rng.normal(0, 0.06, len(v)), v, color='black', s=15, alpha=0.4, zorder=5)
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8); ax.set_ylabel(ylab); ax.set_title(title)
        ax.set_ylim(0); ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.suptitle('Loom responses — ON/ON-OFF cells (mean ± SEM, points = cells)', fontweight='bold')
    fig.tight_layout()
    return {'filename': 'loom_barplots.png'}


def classification_summary(fig, sess):
    """Picker classification counts (printed at the end of cell P1)."""
    df = _df(sess)
    conds = picked_conditions(sess, only_loom=False)
    classes = ['ON', 'OFF', 'ON-OFF', 'NONE']
    ax = fig.add_subplot(111)
    w = 0.8 / (len(conds) + 1)
    cols = ['#4caf50', '#2196f3', '#9c27b0', '#9e9e9e']
    allc = ['overall'] + conds
    for j, cl in enumerate(classes):
        vals = [(df['overall_class'] == cl).sum()] + [(df[c + '_class'] == cl).sum() for c in conds]
        ax.bar(np.arange(len(allc)) + j * 0.2 - 0.3, vals, width=0.2, color=cols[j], label=cl)
    ax.set_xticks(np.arange(len(allc))); ax.set_xticklabels(allc, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('# cells'); ax.legend(); ax.set_title('Blink / loom response classes (n={} cells)'.format(len(df)))
    fig.tight_layout()
    lines = ['Overall:'] + ['  {:6s}: {:3d} ({:.0f}%)'.format(c, (df['overall_class'] == c).sum(),
                                                              100 * (df['overall_class'] == c).mean()) for c in classes]
    return {'filename': 'blink_loom_classes.png', 'text': '\n'.join(lines)}

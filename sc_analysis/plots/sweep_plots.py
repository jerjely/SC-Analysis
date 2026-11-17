"""Sweep analyses from the sweep-picker CSV (cells S2-S7)."""
import json
import os

import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats import linregress

from ..core import pickers
from ..core.psth import trial_counts, smoothed_trials

SPEEDS = [10, 20, 25, 30, 40, 50]


def _df(sess):
    df = sess.picks_df('sweep')
    if df is None or not len(df):
        raise ValueError('No sweep picks yet - use the Sweep picker tab')
    return df


def _speeds(df):
    return [s for s in SPEEDS if 'p1_lat_leading_{}dps'.format(s) in df.columns]


def _points(df, excluded):
    out = {}
    for spd in _speeds(df):
        lc, fc, nc = 'p1_lat_leading_{}dps'.format(spd), 'p1_fr_{}dps'.format(spd), 'n_picks_{}dps'.format(spd)
        m = (df[nc].fillna(0) >= 1) & (~df['cid'].isin(excluded))
        out[spd] = df[m][['cid', 'depth', lc, fc]].dropna().rename(columns={lc: 'lat', fc: 'fr'})
    return out


def latency_vs_depth(fig, sess, use_exclusions=True, size_scale=5.0):
    """Cells S2 + S3: click a dot to exclude that cell (all speeds); right-click restores the last one.
    Exclusions are saved to sweep_excluded_cids.json + interactive_picks_clean.csv."""
    df = _df(sess)
    state = {'excl': sess.sweep_excluded() if use_exclusions else set(), 'hist': []}
    ax = fig.add_subplot(111)
    cols = {s: sess.speed_colors.get('small_sweep_{}dps'.format(s), 'gray') for s in SPEEDS}

    def draw():
        ax.cla()
        pts = _points(df, state['excl'])
        handles = []
        for spd, sub in pts.items():
            c = cols[spd]
            P = np.column_stack([sub['lat'].values, sub['depth'].values])
            if len(P) >= 3:
                try:
                    h = ConvexHull(P); v = P[h.vertices]; v = np.vstack([v, v[0]])
                    ax.plot(v[:, 0], v[:, 1], '-', color=c, lw=1.5, alpha=0.6)
                    ax.fill(v[:, 0], v[:, 1], color=c, alpha=0.08)
                except Exception:
                    pass
            handles.append(ax.scatter(sub['lat'], sub['depth'], s=np.clip(sub['fr'] * size_scale, 10, 800), c=c,
                                      alpha=0.75, edgecolors='white', linewidths=0.5, zorder=5,
                                      label='{} dps (n={})'.format(spd, len(sub))))
        ax.axvline(0, color='gray', ls='--', lw=0.8, alpha=0.5)
        ax.invert_yaxis()
        ax.set_xlabel('Onset latency re. leading edge @ RF-left (ms)'); ax.set_ylabel('Depth (µm)')
        ax.set_title('Sweep onset latency vs depth  [excluded: {}]\nleft-click = exclude cell | right-click = '
                     'restore last'.format(len(state['excl'])), fontweight='bold')
        ax.grid(alpha=0.2)
        l1 = ax.legend(handles=handles, title='Speed', fontsize=8, loc='upper right')
        ax.add_artist(l1)
        sz = [ax.scatter([], [], s=f * size_scale, c='gray', alpha=0.7, edgecolors='white', label='{} Hz'.format(f))
              for f in (10, 30, 60, 100)]
        ax.legend(handles=sz, title='Peak FR', fontsize=8, loc='lower right')
        state['pts'] = pts
        fig.canvas.draw_idle()

    def save():
        with open(sess.p('sweep_excluded_cids.json'), 'w') as f:
            json.dump(sorted(int(c) for c in state['excl']), f)
        df[~df['cid'].isin(state['excl'])].to_csv(sess.p('interactive_picks_clean.csv'), index=False)

    def on_click(ev):
        if ev.inaxes is not ax or ev.xdata is None:
            return
        if ev.button == 3:
            if state['hist']:
                state['excl'].discard(state['hist'].pop()); save(); draw()
            return
        if ev.button != 1 or fig.canvas.toolbar is not None and fig.canvas.toolbar.mode:
            return
        ex, ey = ax.transData.transform((ev.xdata, ev.ydata))
        best, bd = None, np.inf
        for sub in state['pts'].values():
            for _, r in sub.iterrows():
                px, py = ax.transData.transform((r['lat'], r['depth']))
                d = np.hypot(px - ex, py - ey)
                if d < bd:
                    bd, best = d, int(r['cid'])
        if best is not None and bd < 20:
            state['excl'].add(best); state['hist'].append(best); save(); draw()
    draw()
    return {'events': [('button_press_event', on_click)], 'filename': 'sweep_latency_vs_depth.png',
            'text': '{} cells in picks CSV'.format(len(df))}


def depth_vs_arrival(fig, sess, condition='small_sweep_25dps', depth_min=0, depth_max=3000):
    """Cell S4."""
    df = _df(sess)
    spd = sess.cfg.speed_dps[condition]
    lc, nc = 'leading_ms_{}dps'.format(spd), 'n_picks_{}dps'.format(spd)
    sub = df[df[nc].fillna(0) >= 1][['cid', 'depth', lc, 'rf_left_deg', 'rf_x_deg']].dropna()
    sub = sub[(sub['depth'] >= depth_min) & (sub['depth'] <= depth_max)]
    if len(sub) < 3:
        raise ValueError('Not enough picked cells for {}'.format(condition))
    sl, ic, r, p, _ = linregress(sub[lc].values, sub['depth'].values)
    ax = fig.add_subplot(111)
    sc = ax.scatter(sub[lc], sub['depth'], c=sub['rf_left_deg'], cmap='plasma', s=60, alpha=0.75,
                    edgecolors='white', linewidths=0.5, zorder=5)
    xf = np.linspace(sub[lc].min(), sub[lc].max(), 200)
    ax.plot(xf, sl * xf + ic, 'k--', lw=1.5, alpha=0.7, label='r={:.2f} slope={:.2f} µm/ms\np={:.3f}'.format(r, sl, p))
    fig.colorbar(sc, ax=ax, label='RF left edge (deg)')
    ax.invert_yaxis(); ax.legend(fontsize=9, loc='upper right'); ax.grid(alpha=0.2)
    ax.set_xlabel('Time for leading edge to reach RF-left (ms) [{}]'.format(condition)); ax.set_ylabel('Depth (µm)')
    ax.set_title('Depth vs stimulus arrival time at RF | n={} | R²={:.2f}'.format(len(sub), r ** 2), fontweight='bold')
    return {'filename': 'depth_vs_arrival_time_{}.png'.format(condition),
            'text': 'Slope: {:.3f} µm/ms = {:.1f} µm/deg (retinotopic gradient)'.format(sl, sl * spd)}


def population_summary(fig, sess, use_clean=True):
    """Cell S5."""
    df = _df(sess)
    if use_clean:
        df = df[~df['cid'].isin(sess.sweep_excluded())]
    sp = _speeds(df)
    a0, a1, a2 = fig.subplots(1, 3)
    rng = np.random.default_rng(0)
    lat_pop, fr_pop = [], []
    for spd in sp:
        m = df['n_picks_{}dps'.format(spd)].fillna(0) >= 1
        l = df.loc[m, 'p1_lat_leading_{}dps'.format(spd)].values.astype(float)
        f = df.loc[m, 'p1_fr_{}dps'.format(spd)].values.astype(float)
        lat_pop.append(l); fr_pop.append(f)
        c = sess.speed_colors.get('small_sweep_{}dps'.format(spd), 'gray')
        a0.scatter(np.full(len(l), spd) + rng.normal(0, 0.3, len(l)), l, alpha=0.3, s=15, color=c)
        a1.scatter(np.full(len(f), spd) + rng.normal(0, 0.3, len(f)), f, alpha=0.3, s=15, color=c)
    ml = [np.nanmean(l) if len(l) else np.nan for l in lat_pop]
    sl = [np.nanstd(l) / np.sqrt(len(l)) if len(l) > 1 else np.nan for l in lat_pop]
    mf = [np.nanmean(f) if len(f) else np.nan for f in fr_pop]
    sf = [np.nanstd(f) / np.sqrt(len(f)) if len(f) > 1 else np.nan for f in fr_pop]
    a0.errorbar(sp, ml, yerr=sl, fmt='ko-', lw=2, ms=8, capsize=4, zorder=10, label='mean±sem')
    a0.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.5)
    a0.set_xlabel('Speed (dps)'); a0.set_ylabel('Latency re. leading edge @ RF-left (ms)')
    a0.set_title('Latency to peak vs speed'); a0.set_xticks(sp); a0.legend(fontsize=9)
    a1.errorbar(sp, mf, yerr=sf, fmt='ko-', lw=2, ms=8, capsize=4, zorder=10, label='mean±sem')
    a1.set_xlabel('Speed (dps)'); a1.set_ylabel('Peak FR (Hz)'); a1.set_title('Peak firing rate vs speed')
    a1.set_xticks(sp); a1.legend(fontsize=9)
    n = [len(l) for l in lat_pop]
    a2.bar(sp, n, color=[sess.speed_colors.get('small_sweep_{}dps'.format(s), 'gray') for s in sp],
           edgecolor='black', lw=0.8, width=3)
    for s, k in zip(sp, n):
        a2.text(s, k + 0.5, str(k), ha='center', fontsize=9)
    a2.set_xlabel('Speed (dps)'); a2.set_ylabel('N cells with pick'); a2.set_title('Coverage per speed'); a2.set_xticks(sp)
    fig.suptitle('Population sweep analysis | n={} cells'.format(len(df)), fontweight='bold')
    fig.tight_layout()
    txt = '\n'.join('{:3d} dps: {:.0f} ± {:.0f} ms (n={})'.format(s, m, e, k) for s, m, e, k in zip(sp, ml, sl, n))
    return {'filename': 'sweep_population_summary.png', 'text': 'Mean latency vs speed:\n' + txt}


def picked_cells(sess):
    try:
        df = sess.picks_df('sweep')
        if df is None:
            return []
        d = sess.unit_depths
        return [(int(c), 'u{}  {:.0f}µm'.format(int(c), d.get(int(c), np.nan))) for c in df['cid']]
    except Exception:
        return []


def _cell_psths(sess, row):
    bins, bc = pickers.sweep_bins(sess)
    st = sess.spike_trains[int(row['cid'])]
    out = {}
    for cond, spd in sess.speed_dps.items():
        on, off = sess.onsets('P1', cond)
        sm = smoothed_trials(trial_counts(st, on, bins), float(sess.cfg.bin_s),
                             float(sess.cfg.sigma_per_speed.get(cond, 2.0)))
        out[cond] = (spd, sm.mean(0), sm.std(0) / np.sqrt(max(len(on), 1)), float(np.mean(off - on)), len(on))
    return bc, out


def cell_picks(fig, sess, unit=None):
    """Cell S7: one cell's picks at every speed + latency/peak summaries."""
    df = _df(sess)
    if unit not in set(df['cid'].astype(int)):
        raise ValueError('Unit {} not in picks yet'.format(unit))
    row = df[df['cid'].astype(int) == unit].iloc[0]
    bc, ps = _cell_psths(sess, row)
    n = len(ps)
    gs = fig.add_gridspec(n + 2, 2, height_ratios=[1] * n + [0.15, 2.5], hspace=0.6, wspace=0.35)
    axes = [fig.add_subplot(gs[i, :]) for i in range(n)]
    al, ap = fig.add_subplot(gs[n + 1, 0]), fig.add_subplot(gs[n + 1, 1])
    fig.suptitle('Unit {} | RF centre={:.1f}° RF left={:.1f}° diam={:.1f}° base={:.1f}Hz {:.0f}µm'.format(
        unit, row['rf_x_deg'], row['rf_left_deg'], row['rf_diam'], row['base_fr'], row['depth']), fontweight='bold')
    ll, lt, pf, vs, vc = [], [], [], [], []
    t = bc * 1000
    for ax, (cond, (spd, m, s, dur, ntr)) in zip(axes, ps.items()):
        col = sess.speed_colors.get(cond, 'k')
        lead = float(row.get('leading_ms_{}dps'.format(spd), 0) or 0)
        trail = float(row.get('trailing_ms_{}dps'.format(spd), 0) or 0)
        npk = int(np.nan_to_num(row.get('n_picks_{}dps'.format(spd), 0)))
        ax.fill_between(t, np.clip(m - s, 0, None), m + s, alpha=0.25, color=col)
        ax.plot(t, m, color=col, lw=2)
        ax.axvspan(0, dur * 1000, alpha=0.06, color='gray'); ax.axvline(0, color='k', ls='--', lw=1)
        ax.axvline(dur * 1000, color='gray', ls='--', lw=1)
        ax.axvline(lead, color=col, lw=2.5, alpha=0.9); ax.axvline(trail, color=col, ls=':', lw=2.5, alpha=0.9)
        for k in range(1, npk + 1):
            pk = row.get('p{}_ms_{}dps'.format(k, spd), np.nan)
            fr = row.get('p{}_fr_{}dps'.format(k, spd), np.nan)
            l1 = row.get('p{}_lat_leading_{}dps'.format(k, spd), np.nan)
            l2 = row.get('p{}_lat_trailing_{}dps'.format(k, spd), np.nan)
            if not np.isfinite(pk):
                continue
            ax.plot(pk, fr, 'v', color='k', ms=10, mec='white', mew=1, zorder=15)
            y = fr * 1.10
            ax.annotate('', xy=(pk, y), xytext=(lead, y), arrowprops=dict(arrowstyle='<->', color='k', lw=1))
            ax.text((lead + pk) / 2, y * 1.03, '→L:{:+.0f}ms →T:{:+.0f}ms'.format(l1, l2), ha='center', va='bottom',
                    fontsize=6.5)
            if k == 1:
                ll.append(l1); lt.append(l2); pf.append(fr); vs.append(spd); vc.append(col)
        ax.set_xlim(-500, max(dur * 1000 + 500, trail + 800))
        ax.set_ylim(0, max(float((m + s).max()) * 1.3, 5))
        ax.set_ylabel('FR (Hz)', fontsize=8)
        ax.set_title('{} dps n={} lead={:.0f}ms trail={:.0f}ms | {}'.format(
            spd, ntr, lead, trail, '{} pick(s)'.format(npk) if npk else 'no pick'), fontsize=8, color=col)
    axes[-1].set_xlabel('Time from sweep onset (ms)')
    if vs:
        al.plot(vs, ll, 'o-', color='k', lw=1.5, ms=8, label='peak − leading edge')
        al.plot(vs, lt, 's--', color='gray', lw=1.5, ms=8, label='peak − trailing edge')
        ap.scatter(vs, pf, s=80, c=vc, edgecolors='k', zorder=5); ap.plot(vs, pf, 'k-', lw=1.5, alpha=0.6)
    al.axhline(0, color='k', ls='--', lw=0.8, alpha=0.5)
    al.set_xlabel('Speed (dps)'); al.set_ylabel('Latency re. RF-left (ms)'); al.set_title('Latency vs speed'); al.legend(fontsize=7)
    ap.axhline(float(row['base_fr']), color='gray', ls=':', label='base={:.1f}Hz'.format(row['base_fr']))
    ap.set_xlabel('Speed (dps)'); ap.set_ylabel('Peak FR (Hz)'); ap.set_title('Peak FR vs speed'); ap.legend(fontsize=7)
    return {'filename': 'sweep_cell_u{}.png'.format(unit), 'height': 2.6 * n + 5}


def export_all_cell_plots(sess, log=print, progress=None):
    """Cell S6: one PNG per picked cell into SAVE_DIR/sweep_cell_plots."""
    import matplotlib
    from matplotlib.figure import Figure
    df = _df(sess)
    out = sess.p('sweep_cell_plots')
    os.makedirs(out, exist_ok=True)
    for i, (_, row) in enumerate(df.iterrows()):
        fig = Figure(figsize=(14, 2.6 * len(sess.speed_dps) + 5))
        cell_picks(fig, sess, unit=int(row['cid']))
        fn = 'unit{:04d}_RFleft{:+.0f}deg_{}um.png'.format(int(row['cid']), row['rf_left_deg'], int(row['depth']))
        fig.savefig(os.path.join(out, fn), dpi=120, bbox_inches='tight', facecolor='white')
        if progress:
            progress(i + 1, len(df))
    log('Done - {} figures saved to {}'.format(len(df), out))
    return out

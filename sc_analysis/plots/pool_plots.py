"""Figures for the multi-recording classification ('state' = PooledState)."""
import numpy as np
import matplotlib
from matplotlib import cm

from ..core import pooling


def _need(state):
    if state.R is None:
        raise ValueError('Run the pooled classification first (left panel)')
    return state.P, state.R


def _colors(k):
    cmap = matplotlib.colormaps['tab20' if k > 10 else 'tab10']
    return [cmap(i % cmap.N) for i in range(k)]


def cluster_choices(state):
    if state.R is None:
        return []
    df = state.R['df']
    return [(k, 'cluster {}  (n={})'.format(k, int((df['cluster'] == k).sum()))) for k in range(state.R['best_k'])]


def bic(fig, state):
    P, R = _need(state)
    ax = fig.add_subplot(111)
    if R['bics']:
        ax.plot(R['ks'], R['bics'], 'o-')
        ax.axvline(R['best_k'], color='r', ls='--', label='best k = {}'.format(R['best_k']))
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'k fixed at {}'.format(R['best_k']), ha='center', transform=ax.transAxes)
    ax.set_xlabel('number of clusters'); ax.set_ylabel('BIC')
    ax.set_title('Model order — pooled QI-pass units (n={}) from {} recordings'.format(
        int(R['df']['is_qi_pass'].sum()), R['df']['recording'].nunique()))
    return {'filename': 'pooled_bic.png'}


def traces(fig, state, clusters='', qi_only=False):
    """Mean ± SEM of each feature segment per cluster (Baden-style)."""
    P, R = _need(state)
    df, X = R['df'], P['X']
    show = [int(x) for x in str(clusters).replace(',', ' ').split() if x.isdigit()] or list(range(R['best_k']))
    cols = _colors(R['best_k'])
    sb = pooling.segment_bounds(P['sizes'])
    gs = fig.add_gridspec(len(show), 5, width_ratios=[4, 4, 1.6, 1.6, 1.6], hspace=0.15, wspace=0.12,
                          left=0.07, right=0.99, top=0.95, bottom=0.04)
    for r, k in enumerate(show):
        m = (df['cluster'] == k).values
        if qi_only:
            m &= df['is_qi_pass'].values
        for c, (a, b, name) in enumerate(sb):
            ax = fig.add_subplot(gs[r, c])
            if m.sum():
                seg = X[m, a:b]
                mu, se = seg.mean(0), seg.std(0) / np.sqrt(m.sum())
                ax.fill_between(np.arange(b - a), mu - se, mu + se, color=cols[k], alpha=0.3, lw=0)
                ax.plot(mu, color=cols[k], lw=1.3)
            for ev, frac in pooling.SEG_EVENTS[name]:
                ax.axvline(frac * (b - a), color='0.6', lw=0.7, ls='--')
            ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(-2.2, 2.8)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            if r == 0:
                ax.set_title(name, fontsize=9)
            if c == 0:
                g = df[(df['cluster'] == k)]
                ax.set_ylabel('C{}\nn={}\n{} rec'.format(k, len(g), g['recording'].nunique()), rotation=0,
                              ha='right', va='center', fontsize=8, color=cols[k])
    return {'filename': 'pooled_cluster_traces.png', 'height': max(4, 0.9 * len(show) + 1)}


def composition(fig, state, normalise=True):
    """Which recordings / mouse lines each cluster is made of."""
    P, R = _need(state)
    df = R['df']
    a0, a1 = fig.subplots(1, 2)
    for ax, key in [(a0, 'recording'), (a1, 'mouse_line')]:
        tab = df.pivot_table(index='cluster', columns=key, values='unit', aggfunc='count', fill_value=0)
        tab = tab.reindex(range(R['best_k']), fill_value=0)
        if normalise and key == 'mouse_line':
            # share of each line's units falling in each cluster
            tab = tab / tab.sum(axis=0)
            ylabel = 'fraction of the line\'s units'
        else:
            ylabel = '# units'
        bottom = np.zeros(len(tab))
        cmap = matplotlib.colormaps['tab20']
        for i, c in enumerate(tab.columns):
            ax.bar(tab.index, tab[c].values, bottom=bottom, color=cmap(i % 20), label=str(c), width=0.8)
            bottom += tab[c].values
        ax.set_xlabel('cluster'); ax.set_ylabel(ylabel); ax.set_xticks(range(R['best_k']))
        ax.legend(fontsize=7, loc='upper right')
        ax.set_title('Clusters by {}'.format(key.replace('_', ' ')))
    fig.tight_layout()
    t = df.pivot_table(index='cluster', columns='recording', values='unit', aggfunc='count', fill_value=0)
    return {'filename': 'pooled_composition.png', 'text': t.to_string()}


def feature_matrix(fig, state):
    P, R = _need(state)
    df, X = R['df'], P['X']
    o = np.lexsort((df['depths'].values, df['recording'].values, df['cluster'].values))
    ax = fig.add_subplot(111)
    ax.imshow(X[o], aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3, interpolation='nearest')
    for a, b, n in pooling.segment_bounds(P['sizes']):
        ax.axvline(a, color='k', lw=0.8)
        ax.text((a + b) / 2, -2, n, ha='center', va='bottom', fontsize=8)
    lab = df['cluster'].values[o]
    for k in range(R['best_k']):
        idx = np.nonzero(lab == k)[0]
        if len(idx):
            ax.axhline(idx[-1] + 0.5, color='k', lw=0.6)
            ax.text(X.shape[1] + 5, idx.mean(), 'C{}'.format(k), va='center', fontsize=8)
    ax.set_xlabel('feature bin'); ax.set_ylabel('unit (cluster, recording, depth)')
    return {'filename': 'pooled_feature_matrix.png', 'height': 9}


def profile_table(fig, state):
    P, R = _need(state)
    t = pooling.profile(P, R)
    ax = fig.add_subplot(111); ax.axis('off')
    cells = [[('{:.2f}'.format(v) if isinstance(v, float) else str(v)) for v in r] for r in t.reset_index().values]
    tb = ax.table(cellText=cells, colLabels=['cluster'] + list(t.columns), loc='center', cellLoc='center')
    tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.3)
    return {'filename': 'pooled_cluster_profile.png',
            'text': t.to_string(float_format=lambda x: '{:.2f}'.format(x)),
            'height': max(4, 0.3 * len(t) + 1)}


def cluster_detail(fig, state, cluster=0):
    """One cluster: every unit's features, direction tuning, depth, RF size, recordings."""
    P, R = _need(state)
    df, X = R['df'], P['X']
    k = int(cluster)
    m = (df['cluster'] == k).values
    g = df[m]
    col = _colors(R['best_k'])[k]
    gs = fig.add_gridspec(3, 4, height_ratios=[1.3, 1, 1], hspace=0.75, wspace=0.45,
                          left=0.08, right=0.97, top=0.92, bottom=0.06)
    ax = fig.add_subplot(gs[0, :])
    o = np.argsort(g['depths'].values)
    ax.imshow(X[m][o], aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3, interpolation='nearest')
    sb = pooling.segment_bounds(P['sizes'])
    for a, b, n in sb:
        ax.axvline(a, color='k', lw=0.8)
    ax.set_xticks([(a + b) / 2 for a, b, _ in sb]); ax.set_xticklabels([n for _, _, n in sb], fontsize=8)
    ax.set_ylabel('unit (by depth)')
    ax.set_title('Cluster {} — {} units ({} QI-pass, {} RF-only) from {} recordings'.format(
        k, int(m.sum()), int(g['is_qi_pass'].sum()), int((~g['is_qi_pass']).sum()), g['recording'].nunique()),
        fontweight='bold')
    bd = P['bar_dir'][m]
    dirs = P['bar_directions']
    axb = fig.add_subplot(gs[1, :2])
    for j, d in enumerate(dirs):
        if np.isfinite(bd[:, j]).any():
            axb.plot(np.nanmean(bd[:, j], axis=0), label='{}°'.format(d))
    axb.axvline(bd.shape[2] * 0.4, color='0.6', ls='--', lw=0.7)
    axb.set_title('bar response by direction (RF-aligned, z)', fontsize=9)
    axb.legend(fontsize=7, ncol=2, loc='upper left', frameon=False)
    axp = fig.add_subplot(gs[1, 2], projection='polar')
    ds = g[np.isfinite(g['pref_dir'])]
    if len(ds):
        axp.scatter(np.deg2rad(ds['pref_dir']), ds['dsi'], s=12, color=col, alpha=0.7)
    axp.set_title('pref. direction vs DSI', fontsize=9, pad=14); axp.tick_params(labelsize=6)
    axs = fig.add_subplot(gs[1, 3])
    sp = P['speed_shape'][m & (df['speed_frac'].values == 1.0)]
    if len(sp):
        axs.errorbar([10, 20, 25, 30, 40, 50], np.nanmean(sp, 0), yerr=np.nanstd(sp, 0) / np.sqrt(len(sp)),
                     color=col, marker='o')
    axs.set_title('speed tuning (n={})'.format(len(sp)), fontsize=9); axs.set_xlabel('dps')
    a1 = fig.add_subplot(gs[2, 0])
    a1.hist(g['depths'].dropna(), bins=20, color=col, orientation='horizontal'); a1.invert_yaxis()
    a1.set_ylabel('depth (µm)'); a1.set_title('depth', fontsize=9)
    a2 = fig.add_subplot(gs[2, 1])
    a2.hist(g['rf_diameter'].dropna(), bins=15, color=col); a2.set_title('RF diameter (deg)', fontsize=9)
    a3 = fig.add_subplot(gs[2, 2])
    c = g[g['color_reliable']]
    a3.hist(c['color_opponency'].dropna(), bins=15, range=(-1, 1), color=col)
    a3.set_title('colour opponency (n={})'.format(len(c)), fontsize=9)
    a4 = fig.add_subplot(gs[2, 3])
    vc = g['recording'].value_counts()
    a4.barh(range(len(vc)), vc.values, color=col); a4.set_yticks(range(len(vc)))
    a4.set_yticklabels([s[:22] for s in vc.index], fontsize=6); a4.set_title('units per recording', fontsize=9)
    return {'filename': 'pooled_cluster{}.png'.format(k), 'height': 9,
            'text': g.assign(depth=g['depths'].round(0), QI=g['is_qi_pass'].map({True: 'y', False: 'RF'}),
                             p=g['cluster_prob'].round(2))[['recording', 'unit', 'depth', 'QI', 'p']].to_string(index=False)}

"""Baden / clustering figures (cells 8-9, 44, 52-55, 60, 63-64)."""
import numpy as np

from ..core import baden, sync


def b1_diagnostic(fig, sess):
    """Cells 8-9: real photodiode intervals vs protocol reference timeline."""
    cfg = sess.cfg
    real = sess.seg_onsets.get(cfg.b1_segment, np.array([]))
    if len(real) < 3:
        raise ValueError('No {} onsets'.format(cfg.b1_segment))
    exp_t, exp_l = sync.b1_expected(cfg)
    rd, ed = np.diff(real), np.diff(exp_t)
    a0, a1 = fig.subplots(2, 1)
    cmap = {'step': 'tab:blue', 'freq': 'tab:orange', 'contr': 'tab:green', 'bar': 'tab:red', 'colour': 'tab:purple'}
    for i, l in enumerate(exp_l[:-1]):
        r = l.split('_')[1]
        key = ('step' if 'step' in r else 'freq' if 'freq' in r else 'contr' if 'contr' in r
               else 'bar' if 'bar' in r else 'colour')
        a0.scatter(i, ed[i], color=cmap[key], marker='x', s=40)
    a0.plot(range(len(rd)), rd, 'k.-', ms=4, alpha=0.7, label='real (photodiode)')
    for k, c in cmap.items():
        a0.scatter([], [], color=c, marker='x', label=k)
    a0.legend(fontsize=8, loc='upper right'); a0.set_ylabel('inter-edge interval (s)')
    a0.set_title('Real photodiode intervals (line) vs expected (x)')
    rn, en = real - real[0], exp_t - exp_t[0]
    n = min(len(rn), len(en))
    a1.plot(en[:n], rn[:n], 'o-', ms=3)
    a1.plot([0, en[:n].max()], [0, en[:n].max()], 'r--', alpha=0.4, label='y = x')
    a1.set_xlabel('expected time (s)'); a1.set_ylabel('real time (s)'); a1.legend()
    a1.set_title('Real vs expected — slope = actual scale factor')
    fig.tight_layout()
    txt = '{} real edges vs {} expected'.format(len(real), len(exp_t))
    if n > 2:
        s, i = np.polyfit(en[:n], rn[:n], 1)
        txt += '\nFitted scale factor: {:.4f}\nIntercept: {:.4f} s'.format(s, i)
    return {'filename': 'B1_vs_protocol_reference.png', 'text': txt}


def qi_hist(fig, sess):
    qi = sess.results.get('qi')
    if not qi:
        raise ValueError('Compute QI first (button on the left)')
    v = np.array([x for x in qi.values() if np.isfinite(x)])
    thr = float(sess.cfg.qi_threshold)
    ax = fig.add_subplot(111)
    ax.hist(v, bins=40, color='steelblue', edgecolor='white')
    ax.axvline(thr, color='r', ls='--', label='threshold {}'.format(thr))
    ax.set_xlabel('QI'); ax.set_ylabel('# units'); ax.legend()
    ax.set_title('Quality index (max over chirp / contrast / step / bar)')
    pc = np.percentile(v, [10, 25, 50, 75, 90, 95]) if len(v) else []
    return {'filename': 'qi_histogram.png', 'text': 'NaN: {}\npercentiles 10/25/50/75/90/95:\n{}\npassing: {}'.format(
        len(qi) - len(v), np.round(pc, 3), int((v >= thr).sum()))}


def bic_curve(fig, sess):
    C = sess.results.get('clustering')
    if C is None:
        raise ValueError('Run clustering first')
    ax = fig.add_subplot(111)
    ax.plot(C['ks'], C['bics'], marker='o')
    ax.axvline(C['best_k'], color='red', ls='--')
    ax.set_xlabel('n_clusters'); ax.set_ylabel('BIC'); ax.set_title('Model order selection (QI-pass units)')
    return {'filename': 'clustering_bic.png', 'text': 'best k = {}'.format(C['best_k'])}


def cluster_traces(fig, sess, clusters=''):
    """Cell 55: mean ± SEM feature segments per cluster."""
    F, C = sess.results.get('features'), sess.results.get('clustering')
    if C is None:
        raise ValueError('Run clustering first')
    T = baden.get_timing(sess)
    X, lab = F['X'], C['labels']
    sb = T.segment_bounds()
    if sb[-1][1] != X.shape[1]:
        raise ValueError('Segment bin counts {} do not match X {}'.format(sb[-1][1], X.shape[1]))
    ie = T.internal_edges()
    show = [int(x) for x in str(clusters).replace(',', ' ').split() if x.isdigit()] or list(range(C['best_k']))
    axes = fig.subplots(5, 1)
    for ax, (a, b, name) in zip(axes, sb):
        for k in show:
            m = lab == k
            if m.sum() == 0:
                continue
            mt = X[m, a:b].mean(axis=0)
            se = X[m, a:b].std(axis=0) / np.sqrt(m.sum())
            ax.plot(mt, label='cluster {} (n={})'.format(k, m.sum()))
            ax.fill_between(range(b - a), mt - se, mt + se, alpha=0.2)
        for en, ep in ie.get(name, []):
            ax.axvline(ep, color='0.4', ls='--', lw=0.8)
            ax.text(ep, ax.get_ylim()[1], en, fontsize=7, color='0.3', ha='center', va='bottom')
        ax.set_title(name); ax.set_ylabel('z-score')
    axes[0].legend(fontsize=7, ncol=max(1, len(show) // 12 + 1), bbox_to_anchor=(1.01, 1.0), loc='upper left')
    fig.subplots_adjust(left=0.07, right=0.80, hspace=0.45, top=0.97, bottom=0.03)
    return {'filename': 'cluster_mean_traces.png', 'height': 13}


def cluster_heat(fig, sess):
    """Extra: every unit's feature vector, sorted by cluster."""
    F, C = sess.results.get('features'), sess.results.get('clustering')
    if C is None:
        raise ValueError('Run clustering first')
    X, lab = F['X'], C['labels']
    o = np.lexsort((np.array([sess.unit_depths.get(c, 0) for c in F['valid_units']]), lab))
    ax = fig.add_subplot(111)
    ax.imshow(X[o], aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3, interpolation='nearest')
    T = baden.get_timing(sess)
    for a, b, n in T.segment_bounds():
        ax.axvline(a, color='k', lw=0.8)
        ax.text((a + b) / 2, -1, n.split(' ')[0], ha='center', va='bottom', fontsize=8)
    ls = lab[o]
    for k in range(C['best_k']):
        idx = np.nonzero(ls == k)[0]
        if len(idx):
            ax.axhline(idx[-1] + 0.5, color='k', lw=0.6)
            ax.text(X.shape[1] + 2, idx.mean(), str(k), va='center', fontsize=8)
    ax.set_xlabel('feature bin'); ax.set_ylabel('unit (sorted by cluster, depth)')
    return {'filename': 'cluster_feature_matrix.png', 'height': 9}


def dsos_hist(fig, sess, version='pooled', dsi_thresh=0.3, osi_thresh=0.3, p_thresh=0.05, filtered=True):
    """Cells 60/63/64."""
    key = 'dsos_v2' if version == 'pooled' else 'dsos_v1'
    res = sess.results.get(key + ('_filtered' if filtered else ''))
    if res is None:
        raise ValueError('Compute DS/OS ({}) first'.format(version))
    v = list(res.values())
    dsi = np.array([r['dsi'] for r in v]); osi = np.array([r['osi'] for r in v])
    sd = np.array([r['p_dsi'] < p_thresh for r in v]); so = np.array([r['p_osi'] < p_thresh for r in v])
    gs = fig.add_gridspec(2, 4, height_ratios=[1.2, 1])
    a0 = fig.add_subplot(gs[0, :2]); a1 = fig.add_subplot(gs[0, 2:])
    a0.hist(dsi[~sd], bins=20, alpha=0.5, label='n.s.'); a0.hist(dsi[sd], bins=20, alpha=0.7, label='p<{}'.format(p_thresh), color='firebrick')
    a0.axvline(dsi_thresh, color='k', ls='--'); a0.set_xlabel('DSI'); a0.set_ylabel('# units'); a0.legend()
    a0.set_title('Direction selectivity (n={})'.format(len(v)))
    a1.hist(osi[~so], bins=20, alpha=0.5, label='n.s.'); a1.hist(osi[so], bins=20, alpha=0.7, label='p<{}'.format(p_thresh), color='steelblue')
    a1.axvline(osi_thresh, color='k', ls='--'); a1.set_xlabel('OSI'); a1.legend(); a1.set_title('Orientation selectivity')
    ex = [c for c, r in res.items() if r['p_dsi'] < p_thresh]
    ex = sorted(ex, key=lambda c: -res[c]['dsi'])[:4]
    dirs = list(sess.cfg.b1_bar_directions)
    for i, cid in enumerate(ex):
        ax = fig.add_subplot(gs[1, i], projection='polar')
        r = res[cid]
        th = np.deg2rad(dirs + [dirs[0]])
        ax.plot(th, np.clip(np.append(r['mean_resp'], r['mean_resp'][0]), 0, None), 'o-')
        ax.set_title('Unit {}\nDSI={:.2f}'.format(cid, r['dsi']), fontsize=9)
    fig.tight_layout()
    nd = int(((dsi > dsi_thresh) & sd).sum()); no = int(((osi > osi_thresh) & so).sum())
    return {'filename': 'dsi_osi_histograms_{}.png'.format(version), 'height': 8,
            'text': 'DS units (DSI>{}, p<{}): {} / {}\nOS units (OSI>{}, p<{}): {} / {}'.format(
                dsi_thresh, p_thresh, nd, len(v), osi_thresh, p_thresh, no, len(v))}


def colour_scatter(fig, sess):
    res = sess.results.get('colour')
    if res is None:
        raise ValueError('Compute colour opponency first')
    rel = {c: r for c, r in res.items() if r['reliable']}
    a0, a1 = fig.subplots(1, 2)
    b = np.array([r['blue_resp'] for r in rel.values()]); g = np.array([r['green_resp'] for r in rel.values()])
    sig = np.array([r['p_opponency'] < 0.05 for r in rel.values()])
    a0.scatter(b[~sig], g[~sig], s=15, alpha=0.5, label='n.s.')
    a0.scatter(b[sig], g[sig], s=20, color='firebrick', label='p<0.05')
    mx = max(b.max() if len(b) else 1, g.max() if len(g) else 1)
    a0.plot([0, mx], [0, mx], 'k--', lw=0.8); a0.set_xlabel('blue response (Hz)'); a0.set_ylabel('green response (Hz)')
    a0.legend(); a0.set_title('Colour responses (reliable units, n={})'.format(len(rel)))
    a1.hist([r['opponency'] for r in rel.values()], bins=25, color='gray', edgecolor='white')
    a1.set_xlabel('opponency  (green − blue)/(green + blue)'); a1.set_ylabel('# units')
    fig.tight_layout()
    return {'filename': 'colour_opponency.png'}

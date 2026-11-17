"""CCF figures (cells C2-C5)."""
import numpy as np
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d

from ..core import ccf as ccfmod

TYPE_COLORS = {'synchronous': 'red', 'a->b': 'steelblue', 'b->a': 'darkorange'}
TH = np.linspace(0, 2 * np.pi, 200)


def _ccf(sess):
    r = sess.results.get('ccf') or ccfmod.load_ccf(sess)
    if r is None:
        raise ValueError('Compute CCFs first (button on the left)')
    return r


def _legend(ax, bg='RF contours'):
    ax.legend(handles=[Line2D([0], [0], color='red', lw=2, label='synchronous (<5ms)'),
                       Line2D([0], [0], color='steelblue', lw=2, label='a->b directional'),
                       Line2D([0], [0], color='darkorange', lw=2, label='b->a directional'),
                       Line2D([0], [0], color='lightgray', lw=2, label=bg)], fontsize=8, loc='lower right')


def pairs_on_rf_map(fig, sess, consistent_only=False):
    """Cell C2 (both RFM maps) / C3 (consistent cells only, RFM2 positions)."""
    r = _ccf(sess)
    sig = r['sig_pairs']
    rfc = sess.rf_clean
    sets = [('RFM2', sess.rf_rfm2)] if consistent_only else [('RFM1', sess.rf_rfm1), ('RFM2', sess.rf_rfm2)]
    axes = fig.subplots(1, len(sets), squeeze=False)[0]
    txt = []
    for ax, (name, rd) in zip(axes, sets):
        for cid in rfc:
            rf = rd.get(cid, {})
            if not rf.get('fit_ok') or not np.isfinite(rf.get('x_deg', np.nan)):
                continue
            ax.plot(rf['x_deg'] + rf['sigma_x'] * np.cos(TH), rf['y_deg'] + rf['sigma_y'] * np.sin(TH),
                    color='lightgray', lw=0.8, alpha=0.5, zorder=1)
        n = 0
        for p in sig:
            a, b = p['cid_a'], p['cid_b']
            if consistent_only and (a not in rfc or b not in rfc):
                continue
            ra, rb = rd.get(a, {}), rd.get(b, {})
            if not ra.get('fit_ok') or not rb.get('fit_ok'):
                continue
            if not (np.isfinite(ra.get('x_deg', np.nan)) and np.isfinite(rb.get('x_deg', np.nan))):
                continue
            c = TYPE_COLORS.get(p['type'], 'purple')
            ax.plot([ra['x_deg'], rb['x_deg']], [ra['y_deg'], rb['y_deg']], color=c, lw=0.9, alpha=0.45, zorder=2)
            ax.plot([ra['x_deg'], rb['x_deg']], [ra['y_deg'], rb['y_deg']], 'o' if consistent_only else '.',
                    color=c, ms=5 if consistent_only else 4, zorder=3, ls='none')
            n += 1
        ax.axhline(0, color='gray', lw=0.8, alpha=0.5); ax.axvline(0, color='gray', lw=0.8, alpha=0.5)
        ax.set_xlabel('Azimuth (deg)'); ax.set_ylabel('Elevation (deg)')
        ax.set_title('{} | correlated pairs shown: {}'.format(name, n)); ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=0.2)
        txt.append('{}: {} pairs plotted'.format(name, n))
    _legend(axes[-1])
    fig.suptitle('Correlated pairs (z≥{}) on RF map | n_sig={} / {:,} tested{}'.format(
        r['sig_z'], len(sig), r['n_pairs'], ' | consistent cells only' if consistent_only else ''), fontweight='bold')
    return {'filename': 'ccf_pairs_on_rf_map{}.png'.format('_RFM2_consistent' if consistent_only else ''),
            'text': '\n'.join(txt), 'height': 7}


def _zcounts(counts, lags):
    bm = ((lags >= -50) & (lags <= -10)) | ((lags >= 10) & (lags <= 50))
    return (counts - counts[bm].mean()) / (counts[bm].std() + 1e-6)


def ccf_examples(fig, sess, n_examples=4):
    """Cell C4."""
    r = _ccf(sess)
    lags = r['lags_ms']
    types = ['synchronous', 'a->b', 'b->a']
    axes = fig.subplots(len(types), n_examples, squeeze=False)
    rfc = sess.rf_clean
    for row, t in enumerate(types):
        ex = sorted([p for p in r['sig_pairs'] if p['type'] == t], key=lambda p: -p['z_score'])[:n_examples]
        for col in range(n_examples):
            ax = axes[row, col]
            if col >= len(ex):
                ax.axis('off'); continue
            p = ex[col]
            z = _zcounts(np.asarray(p['counts'], float), lags)
            c = TYPE_COLORS[t]
            ax.bar(lags, z, width=r['bin_s'] * 1000, color=c, alpha=0.3)
            ax.plot(lags, gaussian_filter1d(z, 1.5), color=c, lw=1.5)
            ax.axvline(0, color='k', lw=1, ls='--', alpha=0.5); ax.axhline(0, color='gray', lw=0.5)
            ax.axhline(r['sig_z'], color='red', lw=0.8, ls=':'); ax.axvspan(-5, 5, alpha=0.05, color='k')
            ra, rb = rfc.get(p['cid_a'], {}), rfc.get(p['cid_b'], {})
            rs = ('RF dist={:.1f}deg'.format(np.hypot(ra['x_deg'] - rb['x_deg'], ra['y_deg'] - rb['y_deg']))
                  if ra and rb else 'no RF')
            ax.set_title('u{} — u{}\nz={:.1f} | lag={:.1f}ms\nΔdepth={:.0f}µm | {}'.format(
                p['cid_a'], p['cid_b'], p['z_score'], p['peak_lag_ms'], p.get('depth_diff', np.nan), rs), fontsize=7)
            ax.set_xlim(-50, 50); ax.tick_params(labelsize=6)
            if col == 0:
                ax.set_ylabel('{}\nz-score'.format(t), color=c, fontweight='bold', fontsize=8)
    fig.suptitle('CCF examples by type | top {} by z-score | {}'.format(n_examples, sess.cfg.ccf_segment),
                 fontweight='bold')
    fig.tight_layout()
    return {'filename': 'ccf_examples_by_type.png', 'height': 9}


def z_vs_distance(fig, sess):
    """Cell C5."""
    r = _ccf(sess)
    rfc = sess.rf_clean
    rows = []
    for p in r['sig_pairs']:
        ra, rb = rfc.get(p['cid_a'], {}), rfc.get(p['cid_b'], {})
        d = (np.hypot(ra['x_deg'] - rb['x_deg'], ra['y_deg'] - rb['y_deg'])
             if ra and rb and np.isfinite(ra.get('x_deg', np.nan)) and np.isfinite(rb.get('x_deg', np.nan)) else np.nan)
        rows.append((p['z_score'], p['type'], p.get('depth_diff', np.nan), d))
    a0, a1 = fig.subplots(1, 2)
    for t, c in TYPE_COLORS.items():
        pts = [(d, z) for z, ty, dd, d in rows if ty == t and np.isfinite(d)]
        if pts:
            a0.scatter(*zip(*pts), color=c, s=15, alpha=0.5, label='{} (n={})'.format(t, len(pts)))
        pts = [(dd, z) for z, ty, dd, d in rows if ty == t and np.isfinite(dd)]
        if pts:
            a1.scatter(*zip(*pts), color=c, s=15, alpha=0.5, label='{} (n={})'.format(t, len(pts)))
    for ax, xl, tt in [(a0, 'RF distance (deg)', 'Correlation strength vs RF distance'),
                       (a1, 'Depth difference (µm)', 'Correlation strength vs depth difference')]:
        ax.axhline(r['sig_z'], color='gray', ls='--', lw=1, label='threshold z={}'.format(r['sig_z']))
        ax.set_xlabel(xl); ax.set_ylabel('CCF z-score'); ax.set_title(tt); ax.legend(fontsize=8); ax.grid(alpha=0.2)
    a0.set_ylim(r['sig_z'] - 1, max(10, max([x[0] for x in rows] or [10]) + 1))
    fig.suptitle('Significant correlated pairs (n={}) | {}'.format(len(rows), sess.cfg.ccf_segment), fontweight='bold')
    fig.tight_layout()
    return {'filename': 'ccf_zscore_vs_distance.png'}


def far_pair(fig, sess, rf_dist_min=5.0, rank=0):
    """Cell C5b: highest-z pair whose RFs are at least rf_dist_min apart."""
    r = _ccf(sess)
    rfc = sess.rf_clean
    lags = r['lags_ms']
    far = []
    for p in r['sig_pairs']:
        ra, rb = rfc.get(p['cid_a'], {}), rfc.get(p['cid_b'], {})
        if not (ra and rb):
            continue
        d = np.hypot(ra['x_deg'] - rb['x_deg'], ra['y_deg'] - rb['y_deg'])
        if np.isfinite(d) and d >= rf_dist_min:
            far.append((p, d))
    far.sort(key=lambda x: -x[0]['z_score'])
    if not far:
        raise ValueError('No pairs with RF distance >= {} deg'.format(rf_dist_min))
    p, d = far[min(rank, len(far) - 1)]
    z = _zcounts(np.asarray(p['counts'], float), lags)
    c = TYPE_COLORS[p['type']]
    a0, a1 = fig.subplots(1, 2)
    a0.bar(lags, z, width=r['bin_s'] * 1000, color=c, alpha=0.3); a0.plot(lags, gaussian_filter1d(z, 1.5), color=c, lw=2)
    a0.axvline(0, color='k', lw=1, ls='--', alpha=0.5); a0.axhline(r['sig_z'], color='red', lw=0.8, ls=':')
    a0.axvspan(-5, 5, alpha=0.05, color='k'); a0.set_xlim(-50, 50); a0.set_xlabel('Lag (ms)'); a0.set_ylabel('z-score')
    a0.set_title('u{} — u{}\nz={:.1f} | lag={:.1f}ms | {}\nRF dist={:.1f}deg | Δdepth={:.0f}µm'.format(
        p['cid_a'], p['cid_b'], p['z_score'], p['peak_lag_ms'], p['type'], d, p['depth_diff']))
    rd = sess.rf_rfm2
    for cid in rfc:
        rf = rd.get(cid, {})
        if rf.get('fit_ok'):
            a1.plot(rf['x_deg'] + rf['sigma_x'] * np.cos(TH), rf['y_deg'] + rf['sigma_y'] * np.sin(TH), color='lightgray',
                    lw=0.6, alpha=0.4)
    for cid, mk in [(p['cid_a'], 'o'), (p['cid_b'], 's')]:
        rf = rd.get(cid, {})
        if rf.get('fit_ok'):
            a1.plot(rf['x_deg'] + rf['sigma_x'] * np.cos(TH), rf['y_deg'] + rf['sigma_y'] * np.sin(TH), color=c, lw=2)
            a1.plot(rf['x_deg'], rf['y_deg'], mk, color=c, ms=10, label='u{}'.format(cid))
    ra, rb = rfc[p['cid_a']], rfc[p['cid_b']]
    a1.plot([ra['x_deg'], rb['x_deg']], [ra['y_deg'], rb['y_deg']], color=c, lw=1.5, ls='--', alpha=0.7)
    a1.set_aspect('equal', adjustable='box'); a1.legend(fontsize=8); a1.grid(alpha=0.2)
    a1.set_xlabel('Azimuth (deg)'); a1.set_ylabel('Elevation (deg)'); a1.set_title('RF positions (RFM2)')
    fig.tight_layout()
    txt = 'Pairs with RF dist >= {}: {}\n'.format(rf_dist_min, len(far)) + '\n'.join(
        'u{}-u{}: z={:.1f} dist={:.1f} lag={:.1f} {}'.format(q['cid_a'], q['cid_b'], q['z_score'], dd,
                                                           q['peak_lag_ms'], q['type']) for q, dd in far[:10])
    return {'filename': 'ccf_example_far_pair_u{}_u{}.png'.format(p['cid_a'], p['cid_b']), 'text': txt}


def ccf_pair(fig, sess, unit=None, unit_b=None, max_lag_ms=50.0, bin_ms=1.0, segment='S1'):
    """Extra: CCF of any two units in a chosen segment."""
    b = sess.boundaries[segment]
    a = sess.spike_trains[unit]; bb = sess.spike_trains[unit_b]
    a = a[(a >= b['start_s']) & (a < b['end_s'])]; bb = bb[(bb >= b['start_s']) & (bb < b['end_s'])]
    lags, c = ccfmod.compute_ccf_fast(a, bb, max_lag_ms / 1000, bin_ms / 1000)
    ok, z, pl, pv = ccfmod.ccf_significance(c, lags, outer=max_lag_ms, sig_z=float(sess.cfg.ccf_sig_z))
    ax = fig.add_subplot(111)
    ax.bar(lags, c, width=bin_ms, color='gray')
    ax.axvline(0, color='k', ls='--', lw=1)
    ax.set_xlabel('Lag of u{} re. u{} (ms)'.format(unit_b, unit)); ax.set_ylabel('count')
    ax.set_title('CCF u{} × u{} during {} | peak z={:.1f} at {:.1f} ms{}'.format(
        unit, unit_b, segment, z, pl, ' (significant)' if ok else ''))
    return {'filename': 'ccf_u{}_u{}_{}.png'.format(unit, unit_b, segment)}

"""RF / STA figures (cells 1B, 29, 30, 31, 75, 77-79)."""
import numpy as np
from matplotlib import cm, colors as mcolors
from scipy.ndimage import gaussian_filter
from scipy.stats import linregress

from ..core import rf as rfmod

THETA = np.linspace(0, 2 * np.pi, 100)


def _rf_dict(sess, which):
    return {'RFM1': sess.rf_rfm1, 'RFM2': sess.rf_rfm2, 'combined': sess.rf_combined,
            'clean': sess.rf_clean}[which]


def _sta_dict(sess, which):
    return {'RFM1': sess.sta_rfm1, 'RFM2': sess.sta_rfm2}.get(which, sess.sta_rfm1)


def rf_mode_maps(fig, sess, min_snr=6.0, max_diam=30.0):
    """Cell 1B overview: RF ellipses per RFM config, coloured by depth."""
    sets = [(list(sess.cfg.rfm_good)[0], sess.rf_rfm1)]
    if len(sess.cfg.rfm_good) > 1:
        sets.append((sess.cfg.rfm_good[1], sess.rf_rfm2))
    if sess.rf_combined:
        sets.append(('combined', sess.rf_combined))
    sets.append(('clean ({})'.format(sess.cfg.rf_mode), sess.rf_clean))
    axes = fig.subplots(1, len(sets), squeeze=False)[0]
    norm = mcolors.Normalize(0, 3000)
    lines = []
    for ax, (name, d) in zip(axes, sets):
        n = 0
        fits = []
        for cid, r in d.items():
            if not r.get('fit_ok'):
                continue
            x, y, sx, sy = (r.get(k, np.nan) for k in ('x_deg', 'y_deg', 'sigma_x', 'sigma_y'))
            if not np.all(np.isfinite([x, y, sx, sy])):
                continue
            if r.get('snr', 0) < min_snr or r.get('diameter', 999) > max_diam:
                continue
            dep = r.get('depth', np.nan)
            c = cm.viridis(norm(dep)) if np.isfinite(dep) else 'steelblue'
            ax.plot(x + sx * np.cos(THETA), y + sy * np.sin(THETA), color=c, lw=0.9, alpha=0.5)
            ax.plot(x, y, '.', color=c, ms=3, alpha=0.8)
            n += 1
            if np.isfinite(dep):
                fits.append((dep, x))
        t = '{} | n={}'.format(name, n)
        if len(fits) >= 3:
            sl, ic, rr, p, _ = linregress(*zip(*fits))
            t += ' | {:.1f} deg/mm | R²={:.2f}'.format(sl * 1000, rr ** 2)
        ax.set_title(t, fontsize=9, fontweight='bold')
        ax.axhline(0, color='gray', lw=0.8, alpha=0.5); ax.axvline(0, color='gray', lw=0.8, alpha=0.5)
        ax.set_xlabel('Azimuth (deg)'); ax.set_ylabel('Elevation (deg)')
        ax.set_aspect('equal', adjustable='box'); ax.grid(alpha=0.2)
        lines.append(t)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), ax=axes[-1], label='Depth (µm)', fraction=0.03)
    fig.suptitle('RF maps — colour = depth | RF_MODE = {!r}'.format(sess.cfg.rf_mode), fontweight='bold')
    return {'filename': 'rf_maps_mode_selector.png', 'text': '\n'.join(lines), 'height': 5.5}


def rf_depth_filtered(fig, sess, depth_min=0, depth_max=2000, snr_min=6.0):
    """Cell 29."""
    sets = [(sess.rf_rfm1, 'RFM1'), (sess.rf_rfm2, 'RFM2')]
    if sess.rf_combined:
        sets.append((sess.rf_combined, 'combined'))
    axes = fig.subplots(1, len(sets), squeeze=False)[0]
    norm = mcolors.Normalize(0, 3000)
    d = sess.unit_depths
    out = []
    for ax, (rd, lab) in zip(axes, sets):
        u = {c: r for c, r in rd.items() if r.get('fit_ok') and np.isfinite(r.get('x_deg', np.nan))
             and depth_min <= d.get(c, -1) <= depth_max and r.get('snr', 0) >= snr_min}
        for c, r in u.items():
            col = cm.viridis(norm(d.get(c, 0)))
            ax.plot(r['x_deg'], r['y_deg'], '.', color=col, ms=4)
            ax.plot(r['x_deg'] + r['sigma_x'] * np.cos(THETA), r['y_deg'] + r['sigma_y'] * np.sin(THETA),
                    '-', color=col, lw=0.8, alpha=0.7)
        ax.axhline(0, color='k', lw=0.5, alpha=0.3); ax.axvline(0, color='k', lw=0.5, alpha=0.3)
        ax.set_xlabel('Azimuth (deg)'); ax.set_ylabel('Elevation (deg)')
        ax.set_title('{} | n={} | {}-{}µm | SNR≥{}'.format(lab, len(u), depth_min, depth_max, snr_min), fontsize=9)
        ax.set_xlim(-70, 70); ax.set_ylim(-60, 60)
        out.append('{}: {}'.format(lab, len(u)))
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), ax=axes[-1], label='Depth (µm)')
    fig.suptitle('RF maps — depth {}–{}µm | SNR≥{}'.format(depth_min, depth_max, snr_min), fontweight='bold')
    return {'filename': 'rf_maps_depth_{}_{}_snr{}.png'.format(depth_min, depth_max, snr_min),
            'text': '\n'.join(out), 'height': 5.5}


def _units_for_viewer(sess, rf_use, sta_use, depth_min, depth_max, snr_min):
    d = sess.unit_depths
    units = []
    for cid, r in rf_use.items():
        if not r.get('fit_ok') or not np.isfinite(r.get('x_deg', np.nan)):
            continue
        dep = d.get(cid, np.nan)
        if not np.isfinite(dep) or not (depth_min <= dep <= depth_max) or r.get('snr', 0) < snr_min:
            continue
        sta = sta_use.get(cid)
        if sta is None:
            continue
        units.append((cid, r, dep, sta))
    return units


def _draw_sta(ax, sta, r, geom, ds, lags, title_prefix='', cax=None, fig=None):
    li = rfmod.peak_lag_idx(sta)
    frame = gaussian_filter(sta[li], sigma=1.5)
    vmax = np.abs(frame).max() or 1
    im = ax.imshow(frame, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='upper',
                   extent=geom.extent(ds), aspect='equal')
    if r is not None and r.get('fit_ok') and np.isfinite(r.get('x_deg', np.nan)):
        ax.plot(r['x_deg'], r['y_deg'], 'k+', ms=10, markeredgewidth=2)
        ax.plot(r['x_deg'] + r['sigma_x'] * np.cos(THETA), r['y_deg'] + r['sigma_y'] * np.sin(THETA),
                'k-', lw=1.5, alpha=0.9)
        ax.set_xlim(r['x_deg'] - 25, r['x_deg'] + 25)
        ax.set_ylim(r['y_deg'] + 20, r['y_deg'] - 20)
    ax.axhline(0, color='gray', lw=0.4, alpha=0.4); ax.axvline(0, color='gray', lw=0.4, alpha=0.4)
    ax.set_xlabel('Az (deg)', fontsize=8); ax.set_ylabel('El (deg)', fontsize=8)
    if fig is not None:
        if cax is not None:
            fig.colorbar(im, cax=cax)
        else:
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    return li


def rf_interactive(fig, sess, depth_min=0, depth_max=2500, snr_min=6.0, rf_set='clean', sta_set='RFM1'):
    """Cell 30: click a contour to see the STA."""
    geom = sess.geom
    ds = int(sess.cfg.sta_ds)
    lags = list(sess.cfg.sta_lags_s)
    units = _units_for_viewer(sess, _rf_dict(sess, rf_set), _sta_dict(sess, sta_set),
                              depth_min, depth_max, snr_min)
    ax_map = fig.add_axes([0.05, 0.08, 0.40, 0.82])
    ax_sta = fig.add_axes([0.53, 0.08, 0.36, 0.82])
    ax_cb = fig.add_axes([0.91, 0.08, 0.02, 0.82])
    norm = mcolors.Normalize(0, 3000)
    arts = []
    for cid, r, dep, sta in units:
        c = cm.viridis(norm(dep))
        dot, = ax_map.plot(r['x_deg'], -r['y_deg'], '.', color=c, ms=5, picker=8)
        ell, = ax_map.plot(r['x_deg'] + r['sigma_x'] * np.cos(THETA), -r['y_deg'] + r['sigma_y'] * np.sin(THETA),
                           '-', color=c, lw=1.0, alpha=0.7, picker=5)
        arts.append((dot, ell, cid, r, dep, sta))
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax_map, label='Depth (µm)', fraction=0.03, pad=0.02)
    ax_map.axhline(0, color='k', lw=0.5, alpha=0.3); ax_map.axvline(0, color='k', lw=0.5, alpha=0.3)
    ax_map.set_xlim(-geom.width_deg / 2, geom.width_deg / 2)
    ax_map.set_ylim(-geom.height_deg / 2, geom.height_deg / 2)
    ax_map.set_xlabel('Azimuth (deg)'); ax_map.set_ylabel('Elevation (deg)')
    ax_map.set_title('{} RFs — click a contour\n{}-{}µm | SNR≥{} | n={}'.format(
        rf_set, depth_min, depth_max, snr_min, len(units)), fontsize=10)
    ax_sta.set_title('Click a unit to see its {} STA'.format(sta_set)); ax_sta.axis('off'); ax_cb.axis('off')
    sel = [None]

    def on_pick(event):
        hit = None
        for a in arts:
            if event.artist is a[0] or event.artist is a[1]:
                hit = a; break
        if hit is None:
            return
        dot, ell, cid, r, dep, sta = hit
        if sel[0] is not None:
            sel[0][0].set_markersize(5); sel[0][1].set_linewidth(1.0); sel[0][1].set_alpha(0.7)
        dot.set_markersize(10); ell.set_linewidth(2.5); ell.set_alpha(1.0)
        sel[0] = hit
        ax_sta.cla(); ax_cb.cla()
        ax_sta.axis('on')
        li = _draw_sta(ax_sta, sta, r, geom, ds, lags, cax=ax_cb, fig=fig)
        ax_sta.set_title('u{} | {} STA | lag={:.0f}ms\nRF: ({:.1f}°, {:.1f}°) σ=({:.1f}°, {:.1f}°) SNR={:.1f}\n'
                         'depth={:.0f}µm'.format(cid, sta_set, lags[li] * 1000, r['x_deg'], r['y_deg'],
                                                 r['sigma_x'], r['sigma_y'], r['snr'], dep), fontsize=9)
        fig.canvas.draw_idle()
    return {'events': [('pick_event', on_pick)], 'filename': 'rf_interactive.png',
            'text': '{} units shown. Click any contour.'.format(len(units))}


def rf_examples(fig, sess, depth_min=0, depth_max=2500, snr_min=6.0, n_examples=4):
    """Cell 31: RF-vs-depth map + example STAs spread across depth."""
    geom = sess.geom
    ds = int(sess.cfg.sta_ds)
    lags = list(sess.cfg.sta_lags_s)
    units = _units_for_viewer(sess, sess.rf_clean, sess.sta_rfm1, depth_min, depth_max, snr_min)
    if not units:
        raise ValueError('No units pass the filters')
    deps = np.array([u[2] for u in units])
    edges = np.linspace(deps.min(), deps.max(), n_examples + 1)
    ex = []
    for i in range(n_examples):
        inb = [u for u in units if edges[i] <= u[2] <= edges[i + 1]]
        if inb:
            ex.append(max(inb, key=lambda u: u[1].get('snr', 0)))
    ax_map = fig.add_axes([0.05, 0.08, 0.38, 0.82])
    ncol = 2
    nrow = int(np.ceil(len(ex) / 2.0)) or 1
    gs = fig.add_gridspec(nrow, ncol, left=0.50, right=0.98, top=0.88, bottom=0.08, wspace=0.35, hspace=0.6)
    norm = mcolors.Normalize(0, 3000)
    exid = {u[0]: i + 1 for i, u in enumerate(ex)}
    for cid, r, dep, sta in units:
        c = cm.viridis(norm(dep))
        isx = cid in exid
        ax_map.plot(r['x_deg'], -r['y_deg'], '.', color=c, ms=9 if isx else 5, zorder=5 if isx else 1)
        ax_map.plot(r['x_deg'] + r['sigma_x'] * np.cos(THETA), -r['y_deg'] + r['sigma_y'] * np.sin(THETA), '-',
                    color='k' if isx else c, lw=2.2 if isx else 1.0, alpha=1.0 if isx else 0.7, zorder=5 if isx else 1)
        if isx:
            ax_map.annotate(str(exid[cid]), (r['x_deg'], -r['y_deg']), xytext=(4, 4),
                            textcoords='offset points', fontsize=9, fontweight='bold')
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax_map, label='Depth (µm)', fraction=0.03, pad=0.02)
    ax_map.set_xlim(-geom.width_deg / 2, geom.width_deg / 2)
    ax_map.set_ylim(-geom.height_deg / 2, geom.height_deg / 2)
    ax_map.axhline(0, color='k', lw=0.5, alpha=0.3); ax_map.axvline(0, color='k', lw=0.5, alpha=0.3)
    ax_map.set_xlabel('Azimuth (deg)'); ax_map.set_ylabel('Elevation (deg)')
    ax_map.set_title('Consistent RFs vs depth | numbered = examples', fontsize=10)
    for i, (cid, r, dep, sta) in enumerate(ex):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        li = _draw_sta(ax, sta, r, geom, ds, lags, fig=fig)
        ax.set_title('{}. u{} | lag={:.0f}ms\n({:.1f}°, {:.1f}°) SNR={:.1f} | {:.0f}µm'.format(
            i + 1, cid, lags[li] * 1000, r['x_deg'], r['y_deg'], r['snr'], dep), fontsize=8)
    fig.suptitle('Receptive fields vs depth, with example single-unit STAs', fontweight='bold')
    return {'filename': 'rf_vs_depth_with_examples.png', 'height': 7.5,
            'text': '\n'.join('u{} {:.0f}µm SNR={:.1f}'.format(u[0], u[2], u[1]['snr']) for u in ex)}


def sta_browser(fig, sess, n_show=12, start=0):
    """Cell 75: top consistent units, RFM1 vs RFM2 STA side by side."""
    geom = sess.geom
    ds = int(sess.cfg.sta_ds)
    lags = list(sess.cfg.sta_lags_s)
    s1, s2 = sess.sta_rfm1, sess.sta_rfm2

    def snr(s):
        return rfmod.sta_snr(s) if s is not None else 0
    ranked = sorted(sess.rf_clean.keys(), key=lambda c: -(snr(s1.get(c)) + snr(s2.get(c))) / 2)
    show = ranked[start:start + n_show]
    if not show:
        raise ValueError('No clean RF units')
    axes = fig.subplots(len(show), 2, squeeze=False)
    names = list(sess.cfg.rfm_good) + ['RFM2']
    for row, cid in enumerate(show):
        rf = sess.rf_clean[cid]
        for col, (sd, rd, lab) in enumerate([(s1, sess.rf_rfm1, names[0]), (s2, sess.rf_rfm2, names[1])]):
            ax = axes[row, col]
            sta = sd.get(cid)
            if sta is None:
                ax.text(0.5, 0.5, 'no STA', ha='center', transform=ax.transAxes); ax.axis('off'); continue
            r = rd.get(cid, {})
            li = _draw_sta(ax, sta, r if r.get('fit_ok') else None, geom, ds, lags, fig=fig)
            ax.set_xlim(rf['x_deg'] - 25, rf['x_deg'] + 25); ax.set_ylim(rf['y_deg'] + 20, rf['y_deg'] - 20)
            ax.set_title('u{} {} | lag={:.0f}ms | SNR={:.1f}\n{:.0f}µm | shift={:.1f}°'.format(
                cid, lab, lags[li] * 1000, rfmod.sta_snr(sta), sess.unit_depths.get(cid, 0),
                rf.get('shift_deg', 0)), fontsize=7)
            ax.tick_params(labelsize=5)
    fig.suptitle('Clean RFs — ranked by STA SNR ({}–{} of {})'.format(
        start + 1, start + len(show), len(ranked)), fontweight='bold')
    fig.subplots_adjust(hspace=0.7, top=0.95, bottom=0.03)
    return {'filename': 'rf_maps_consistent_RFM1_vs_RFM2.png', 'height': 2.8 * len(show)}


def retinotopy_summary(fig, sess, depth_cutoff=3000):
    """Cell 77."""
    g = [(r['depth'], r['x_deg'], r['y_deg'], r['diameter']) for r in sess.rf_clean.values()
         if np.isfinite(r.get('depth', np.nan)) and r['depth'] <= depth_cutoff]
    if len(g) < 3:
        raise ValueError('Not enough RF fits: n={}'.format(len(g)))
    d, az, el, dm = (np.array(x) for x in zip(*g))
    sa, ia, ra, pa, _ = linregress(d, az)
    se, ie, re_, pe, _ = linregress(d, el)
    tot = np.hypot(sa * 1000, se * 1000)
    axes = fig.subplots(1, 3)
    sc = axes[0].scatter(az, el, c=d, cmap='viridis', s=8, alpha=0.6)
    axes[0].set_xlabel('Azimuth (deg)'); axes[0].set_ylabel('Elevation (deg)')
    axes[0].set_title('RF centres (n={})'.format(len(g)))
    fig.colorbar(sc, ax=axes[0], label='Depth (µm)')
    axes[1].scatter(d, az, c=d, cmap='viridis', s=8, alpha=0.4)
    xl = np.array([d.min(), d.max()])
    axes[1].plot(xl, sa * xl + ia, 'r-', lw=2, label='{:.1f}deg/mm R²={:.3f}'.format(sa * 1000, ra ** 2))
    axes[1].set_xlabel('Depth (µm)'); axes[1].set_ylabel('RF azimuth (deg)'); axes[1].legend()
    axes[1].set_title('SC retinotopic map')
    axes[2].hist(dm[np.isfinite(dm)], bins=25, color='steelblue', edgecolor='white')
    axes[2].axvline(np.nanmedian(dm), color='red', ls='--', label='median={:.1f}deg'.format(np.nanmedian(dm)))
    axes[2].set_xlabel('RF diameter (deg)'); axes[2].set_title('RF size distribution'); axes[2].legend()
    fig.suptitle('RF mapping | {:.1f}deg/mm total | az R²={:.3f} el R²={:.3f}'.format(tot, ra ** 2, re_ ** 2),
                 fontweight='bold')
    fig.tight_layout()
    txt = ('n={}\nAzimuth vs depth:   {:.2f}deg/mm | R2={:.3f} | p={:.2e}\n'
           'Elevation vs depth: {:.2f}deg/mm | R2={:.3f} | p={:.2e}\nTotal RF shift: {:.2f}deg/mm\n'
           'RF centre az={:.1f}+/-{:.1f} el={:.1f}+/-{:.1f} diam={:.1f}deg').format(
        len(g), sa * 1000, ra ** 2, pa, se * 1000, re_ ** 2, pe, tot, np.median(az), np.std(az),
        np.median(el), np.std(el), np.median(dm))
    return {'filename': 'retinotopic_map_summary.png', 'text': txt}


def rf_contours(fig, sess, depth_min=0, depth_max=3000, flip_y=False):
    """Cells 78/79."""
    geom = sess.geom
    ax = fig.add_subplot(111)
    norm = mcolors.Normalize(depth_min, depth_max)
    n = 0
    for cid, r in sess.rf_clean.items():
        dep = r.get('depth', np.nan)
        if not r.get('fit_ok') or not np.isfinite(dep) or not (depth_min <= dep <= depth_max):
            continue
        c = cm.viridis(norm(dep))
        ax.plot(r['x_deg'], r['y_deg'], '.', color=c, ms=3, alpha=0.8)
        ax.plot(r['x_deg'] + r['sigma_x'] * np.cos(THETA), r['y_deg'] + r['sigma_y'] * np.sin(THETA), '-',
                color=c, lw=0.8, alpha=0.6)
        n += 1
    ax.axhline(0, color='k', lw=0.5, alpha=0.3); ax.axvline(0, color='k', lw=0.5, alpha=0.3)
    if flip_y:
        ax.set_xlim(-geom.width_deg / 2, geom.width_deg / 2)
        ax.set_ylim(geom.height_deg / 2, -geom.height_deg / 2)
    ax.set_xlabel('Azimuth (deg)'); ax.set_ylabel('Elevation (deg)')
    ax.set_title('RF contours — {}–{}µm | n={} | RF_MODE={!r}'.format(depth_min, depth_max, n, sess.cfg.rf_mode),
                 fontweight='bold')
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax, label='Depth (µm)')
    return {'filename': 'rf_contours_{}_{}um.png'.format(depth_min, depth_max)}


def sta_lags(fig, sess, unit=None, sta_set='RFM1'):
    """Extra: all STA lags of one unit."""
    sd = _sta_dict(sess, sta_set)
    sta = sd.get(unit)
    if sta is None:
        raise ValueError('No {} STA for unit {}'.format(sta_set, unit))
    geom = sess.geom
    ds = int(sess.cfg.sta_ds)
    lags = list(sess.cfg.sta_lags_s)
    vmax = np.abs(sta).max() or 1
    axes = fig.subplots(2, int(np.ceil(len(lags) / 2.0)), squeeze=False).ravel()
    rd = sess.rf_rfm1 if sta_set == 'RFM1' else sess.rf_rfm2
    r = rd.get(unit, {})
    for i, lag in enumerate(lags):
        ax = axes[i]
        ax.imshow(gaussian_filter(sta[i], 1.0), cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='upper',
                  extent=geom.extent(ds), aspect='equal')
        if r.get('fit_ok'):
            ax.plot(r['x_deg'] + r['sigma_x'] * np.cos(THETA), r['y_deg'] + r['sigma_y'] * np.sin(THETA), 'k-', lw=1)
        ax.set_title('{:.0f} ms'.format(lag * 1000), fontsize=9)
        ax.tick_params(labelsize=6)
    for j in range(len(lags), len(axes)):
        axes[j].axis('off')
    fig.suptitle('u{} — {} STA, all lags | SNR={:.1f}'.format(unit, sta_set, rfmod.sta_snr(sta)), fontweight='bold')
    return {'filename': 'sta_lags_u{}_{}.png'.format(unit, sta_set)}

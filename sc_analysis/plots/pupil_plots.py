"""Pupil / OKN figures (cells PUP2-PUP5, 142-148, 153, 155-165)."""
import numpy as np
from matplotlib import cm
from scipy.ndimage import gaussian_filter1d

from ..core import pupil as pmod
from ..core.psth import compute_psth, trial_counts, raster


def _pd(sess, seg):
    d = pmod.get_pupil(sess, seg)
    if d is None:
        raise ValueError('No pupil data for {} - run the alignment for this segment first'.format(seg))
    return d


def pupil_segments(sess):
    try:
        return [s for s in sess.order if s.startswith('G') or s == 'P1']
    except Exception:
        return ['G1', 'G2', 'P1']


def sf_choices(sess, seg='G2'):
    try:
        return sorted(set(t['deg'] for t in pmod.grating_trials(sess, seg)))
    except Exception:
        return []


def all_sf(sess):
    out = set()
    for s in pupil_segments(sess):
        if s.startswith('G'):
            out |= set(sf_choices(sess, s))
    return sorted(out)


def alignment_check(fig, sess, segment='G2'):
    d = _pd(sess, segment)
    lum = d.get('lum')
    a0, a1 = fig.subplots(2, 1)
    b = sess.boundaries[segment]
    if lum is not None and d.get('slope') is not None and d.get('fps_screen'):
        t = d['slope'] * (np.arange(len(lum)) / d['fps_screen']) + d['intercept']
        a0.plot(t, lum, lw=0.5, color='steelblue')
        flat = sess.stim_onsets_flat
        on = flat[(flat >= b['start_s']) & (flat < b['end_s'])]
        for x in on[:40]:
            a0.axvline(x, color='green', lw=0.8, alpha=0.6)
        if len(on):
            a0.set_xlim(on[0] - 5, on[min(len(on) - 1, 6)] + 5)
        a0.set_title('Screen luminance in imec time (green = TTL onsets)')
    else:
        a0.text(0.5, 0.5, 'Luminance trace available after running the alignment in this session',
                ha='center', transform=a0.transAxes)
    a0.set_xlabel('imec time (s)'); a0.set_ylabel('luminance')
    pr = d.get('align_pairs')
    if pr is not None:
        st, it, res = pr
        a1.plot(it, res * 1000, 'o', ms=5, alpha=0.8)
        a1.axhline(0, color='k', ls='--', lw=0.8)
        a1.set_title('Alignment residuals | std={:.2f} ms'.format(res.std() * 1000))
        a1.set_xlabel('imec time (s)'); a1.set_ylabel('residual (ms)')
    else:
        t = d['pupil_imec_t']
        a1.plot(t, d['pupil_x_sm'], lw=0.5)
        a1.axvspan(b['start_s'], b['end_s'], color='gray', alpha=0.1)
        a1.set_title('Pupil x over the segment (grey = segment window)')
        a1.set_xlabel('imec time (s)'); a1.set_ylabel('pupil x (px)')
    fig.tight_layout()
    return {'filename': 'pupil_alignment_{}.png'.format(segment),
            'text': '{}: {} pupil frames, fps={:.3f}\nframe 0 -> {:.2f}s, last -> {:.2f}s\nsegment {:.2f}-{:.2f}s\n'
                    'camera offset {} ms'.format(segment, len(d['pupil_imec_t']), d['pupil_fps'],
                                                 d['pupil_imec_t'][0], d['pupil_imec_t'][-1], b['start_s'], b['end_s'],
                                                 '{:.1f}'.format(d['cam_offset_s'] * 1000) if d.get('cam_offset_s') is not None else '?')}


def per_sf_mean(fig, sess, segment='G2', pre_s=2.0, post_s=12.0, drift_s=10.0):
    """Cell PUP3."""
    d = _pd(sess, segment)
    tr = pmod.grating_trials(sess, segment)
    tc, ex, info = pmod.epoch(d, tr, pre_s, post_s, 'pupil_x_sm')
    _, ey, _ = pmod.epoch(d, tr, pre_s, post_s, 'pupil_y_sm')
    ex, ey = pmod.baseline_sub(tc, ex), pmod.baseline_sub(tc, ey)
    degs = sorted(set(e['deg'] for e in info))
    cols = cm.plasma(np.linspace(0, 1, max(len(degs), 1)))
    axes = fig.subplots(2, 1, sharex=True)
    for ax, E, yl in [(axes[0], ex, 'Pupil x (px, baseline-sub)'), (axes[1], ey, 'Pupil y (px, baseline-sub)')]:
        for deg, c in zip(degs, cols):
            idx = [i for i, e in enumerate(info) if e['deg'] == deg]
            m = E[idx].mean(0); s = E[idx].std(0) / np.sqrt(len(idx))
            ax.fill_between(tc, m - s, m + s, alpha=0.2, color=c); ax.plot(tc, m, color=c, lw=1.5, label='{} deg'.format(deg))
        ax.axvline(0, color='green', ls='--', lw=1.5, label='drift ON'); ax.axvline(drift_s, color='red', ls='--', lw=1.5, label='drift OFF')
        ax.axvspan(0, drift_s, alpha=0.05, color='gray'); ax.axhline(0, color='k', lw=0.5, ls=':')
        ax.set_ylabel(yl); ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7, ncol=2, loc='upper right')
    axes[0].set_title('{} | pupil position per grating size | n={} trials | fps={:.2f}'.format(
        segment, len(info), d['pupil_fps']))
    axes[1].set_xlabel('Time from drift onset (s)')
    fig.tight_layout()
    return {'filename': 'pupil_xy_per_sf_{}.png'.format(segment)}


def trial_browser(fig, sess, segment='G2', trial=0, pre_s=2.0, post_s=12.0, drift_s=10.0):
    """Cell PUP4."""
    d = _pd(sess, segment)
    tc, ex, info = pmod.epoch(d, pmod.grating_trials(sess, segment), pre_s, post_s, 'pupil_x_sm')
    _, ey, _ = pmod.epoch(d, pmod.grating_trials(sess, segment), pre_s, post_s, 'pupil_y_sm')
    if not len(info):
        raise ValueError('No trials')
    trial = int(np.clip(trial, 0, len(info) - 1))
    ex, ey = pmod.baseline_sub(tc, ex), pmod.baseline_sub(tc, ey)
    a0, a1 = fig.subplots(2, 1, sharex=True)
    for ax, y, c, yl in [(a0, ex[trial], 'steelblue', 'Pupil x (px)'), (a1, ey[trial], 'firebrick', 'Pupil y (px)')]:
        ax.plot(tc, y, lw=1, color=c)
        ax.axvline(0, color='green', ls='--', lw=1.5); ax.axvline(drift_s, color='red', ls='--', lw=1.5)
        ax.axvspan(0, drift_s, alpha=0.05, color='gray'); ax.axhline(0, color='k', lw=0.5, ls=':')
        ax.set_ylabel(yl); ax.grid(alpha=0.2)
    a1.set_xlabel('Time from drift onset (s)')
    e = info[trial]
    fig.suptitle('Trial {}/{} | {} | t={:.1f}s   (← → keys step trials)'.format(trial + 1, len(info), e['cond'], e['t_on']),
                 fontweight='bold')
    return {'filename': 'pupil_trial_{}_{}.png'.format(segment, trial + 1)}


def raster_pupil(fig, sess, segments=('G1', 'G2'), sf=4.0, trial=0, pre_s=2.0, post_s=12.0, bin_s=0.05,
                 drift_s=10.0):
    """Cells PUP5 / 153: all units by depth + pupil x overlay, G1 and G2 side by side."""
    segments = [s for s in (segments or []) if s] or ['G2']
    cids = [c for c in sess.good_ids_by_depth if np.isfinite(sess.unit_depths.get(c, np.nan))]
    deps = [sess.unit_depths[c] for c in cids]
    n = len(cids)
    bins = np.arange(-pre_s, post_s + bin_s, bin_s)
    axes = fig.subplots(1, 2 * len(segments), gridspec_kw={'width_ratios': [4, 1] * len(segments)}, squeeze=False)[0]
    maxn = 0
    for k, seg in enumerate(segments):
        am, ap = axes[2 * k], axes[2 * k + 1]
        trials = [t for t in pmod.grating_trials(sess, seg) if abs(t['deg'] - sf) < 1e-6]
        maxn = max(maxn, len(trials))
        if trial >= len(trials):
            am.text(0.5, 0.5, 'Only {} trials'.format(len(trials)), ha='center', transform=am.transAxes)
            am.set_title('{} — no trial {}'.format(seg, trial + 1)); continue
        t_on = trials[trial]['t_on']
        xs, ys = [], []
        for row, c in enumerate(cids):
            st = sess.spike_trains[c]
            s = st[(st >= t_on - pre_s) & (st <= t_on + post_s)] - t_on
            xs.append(s); ys.append(np.full(len(s), row, float))
        if xs:
            am.vlines(np.concatenate(xs), np.concatenate(ys) + 0.1, np.concatenate(ys) + 0.9, lw=0.4, color='k', alpha=0.6)
        d = pmod.get_pupil(sess, seg)
        if d is not None:
            t = d['pupil_imec_t']
            m = (t >= t_on - pre_s) & (t <= t_on + post_s)
            if m.sum() > 2:
                x = d['pupil_x_sm'][m]; xc = x - x.mean(); rg = np.abs(xc).max() + 1e-6
                am.plot(t[m] - t_on, xc / rg * n * 0.3 + n * 0.5, lw=1.5, color='cornflowerblue', alpha=0.85,
                        zorder=5, label='Pupil x')
        am.axvline(0, color='green', ls='--', lw=1.5, label='Drift ON'); am.axvline(drift_s, color='red', ls='--', lw=1.5, label='Drift OFF')
        am.axvspan(0, drift_s, alpha=0.05, color='gray')
        step = max(1, n // 8)
        rows = list(range(0, n, step))
        am.set_yticks([r + 0.5 for r in rows])
        am.set_yticklabels(['{:.0f}µm'.format(deps[r]) for r in rows] if k == 0 else [], fontsize=7)
        am.set_ylim(0, n); am.set_xlim(-pre_s, post_s); am.set_xlabel('Time from drift onset (s)')
        am.legend(fontsize=7, loc='upper right', ncol=3)
        am.set_title('{} | trial {}/{} | t={:.1f}s'.format(seg, trial + 1, len(trials), t_on), fontweight='bold')
        mat = np.zeros((n, len(bins) - 1))
        for row, c in enumerate(cids):
            mat[row] = trial_counts(sess.spike_trains[c], [t['t_on'] for t in trials], bins).sum(0)
        mat /= max(len(trials), 1) * bin_s
        ms = np.array([gaussian_filter1d(r, 2) for r in mat]) if n else mat
        rm = ms.max(1, keepdims=True); rm[rm == 0] = 1
        ap.imshow(ms / rm, aspect='auto', origin='lower', extent=[-pre_s, post_s, 0, n], cmap='hot', vmin=0, vmax=1)
        ap.axvline(0, color='cyan', ls='--', lw=1.2); ap.axvline(drift_s, color='yellow', ls='--', lw=1.2)
        ap.set_yticks([]); ap.set_xlabel('Time (s)'); ap.set_title('Mean PSTH', fontsize=8)
    fig.suptitle('SF = {} | trial {}/{} | {} units by depth   (← → keys step trials)'.format(sf, trial + 1, maxn, n),
                 fontweight='bold')
    fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.08, wspace=0.08)
    return {'filename': 'raster_pupil_sf{}_trial{}.png'.format(sf, trial + 1), 'height': 9}


def neural_osc_vs_sf(fig, sess, segment='G2', bin_s=0.033, fmin=0.2, fmax=6.0, n_top=3):
    """Cell 142."""
    tr = pmod.grating_trials(sess, segment)
    cids = [c for c in sess.good_ids_by_depth if np.isfinite(sess.unit_depths.get(c, np.nan))]
    res = pmod.neural_psd(sess, tr, cids, bin_s, fmin, fmax, n_top)
    return _osc_plot(fig, sess, res, 'Population FR', 'raster_osc_vs_sf_{}.png'.format(segment), fmin, fmax)


def okn_freq_vs_sf(fig, sess, segment='G2', fmin=0.1, fmax=12.0, n_top=3):
    """Cell 143."""
    d = _pd(sess, segment)
    res = pmod.okn_psd(d, pmod.grating_trials(sess, segment), fmin, fmax, n_top)
    return _osc_plot(fig, sess, res, 'Pupil x', 'pupil_okn_freq_vs_sf_{}.png'.format(segment), fmin, fmax)


def _osc_plot(fig, sess, res, what, fname, fmin, fmax):
    if not res:
        raise ValueError('No trials long enough for a PSD')
    gp = sess.cfg.grating_params
    degs = sorted(res)
    cols = cm.plasma(np.linspace(0, 1, len(degs)))
    a0, a1 = fig.subplots(1, 2)
    for i, (deg, c) in enumerate(zip(degs, cols)):
        r = res[deg]
        pn = r['psd'] / (r['psd'].max() + 1e-12)
        a0.plot(r['f'], pn + i * 0.8, lw=1.5, color=c, label='{}'.format(deg))
        a0.fill_between(r['f'], i * 0.8, pn + i * 0.8, alpha=0.15, color=c)
        for f, p in zip(r['freqs'], r['pows']):
            a0.plot(f, p / (r['psd'].max() + 1e-12) + i * 0.8, 'v', color=c, ms=7)
    a0.set_xlim(fmin, fmax); a0.set_xlabel('Frequency (Hz)'); a0.set_ylabel('norm. power + offset')
    a0.set_title('{} PSD per grating size (▼ peaks)'.format(what)); a0.legend(fontsize=7, ncol=2); a0.grid(alpha=0.2)
    ed = [d for d in sorted(gp) if fmin <= gp[d]['tf'] <= fmax]
    a1.plot(ed, [gp[d]['tf'] for d in ed], 'k--', lw=2, alpha=0.7, label='Expected TF')
    a1.plot(ed, [gp[d]['tf'] / 2 for d in ed], 'k:', lw=1.5, alpha=0.5, label='½ × TF')
    pc = ['royalblue', 'orange', 'green']
    for k in range(3):
        xs = [d for d in degs if k < len(res[d]['freqs'])]
        ys = [res[d]['freqs'][k] for d in xs]
        ss = [res[d]['pows'][k] / (res[d]['psd'].max() + 1e-12) * 300 + 30 for d in xs]
        if xs:
            a1.scatter(xs, ys, s=ss, color=pc[k], alpha=0.85, edgecolors='white', zorder=5, label='peak #{}'.format(k + 1))
            a1.plot(xs, ys, lw=1, color=pc[k], alpha=0.4)
    a1.set_xscale('log'); a1.set_yscale('log'); a1.grid(alpha=0.2, which='both'); a1.legend(fontsize=8)
    a1.set_xlabel('Grating size (deg)'); a1.set_ylabel('Oscillation frequency (Hz)')
    a1.set_title('{} oscillation frequency vs grating'.format(what))
    fig.tight_layout()
    lines = ['{:>6}  {:>8}  {:>8}  {:>8}'.format('size', 'ExpTF', 'peak1', 'peak2')]
    for d in degs:
        f = list(res[d]['freqs']) + [np.nan, np.nan]
        lines.append('{:>6.1f}  {:>8.3f}  {:>8.3f}  {:>8.3f}'.format(d, gp.get(d, {}).get('tf', np.nan), f[0], f[1]))
    return {'filename': fname, 'text': '\n'.join(lines)}


def okn_reset_rate(fig, sess, segment='G2', vel_thresh_std=3.0, min_sep_s=0.5):
    """Cell 146."""
    d = _pd(sess, segment)
    r = pmod.okn_reset_rates(d, pmod.grating_trials(sess, segment), vel_thresh_std, min_sep_s)
    gp = sess.cfg.grating_params
    degs = [x for x in sorted(r) if len(r[x])]
    ax = fig.add_subplot(111)
    lg = np.log10(degs)
    rng = np.random.default_rng(0)
    for deg in degs:
        ax.scatter(np.log10([deg] * len(r[deg])) + rng.uniform(-0.01, 0.01, len(r[deg])), r[deg], s=30, alpha=0.5,
                   color='steelblue')
    m = [np.mean(r[x]) for x in degs]; s = [np.std(r[x]) / np.sqrt(len(r[x])) for x in degs]
    ax.errorbar(lg, m, yerr=s, fmt='o-', ms=8, color='steelblue', lw=2, capsize=5, label='OKN reset rate (mean±SEM)')
    tf = [gp.get(x, {}).get('tf', np.nan) for x in degs]
    ax.plot(lg, tf, 'k--', lw=2, alpha=0.7, label='Expected TF'); ax.plot(lg, [t / 2 for t in tf], 'k:', lw=1.5, alpha=0.5, label='½ × TF')
    gm = np.mean([v for x in degs for v in r[x]]) if degs else np.nan
    ax.axhline(gm, color='red', ls='--', lw=1.5, alpha=0.7, label='grand mean = {:.3f}Hz'.format(gm))
    ax.set_xticks(lg); ax.set_xticklabels([str(x) for x in degs]); ax.legend(fontsize=9); ax.grid(alpha=0.2)
    ax.set_xlabel('Grating size (deg)'); ax.set_ylabel('OKN reset rate (Hz)')
    ax.set_title('OKN fast-phase reset rate vs grating', fontweight='bold')
    return {'filename': 'okn_reset_rate_vs_sf_{}.png'.format(segment),
            'text': '\n'.join('{:>5.1f}: {:.3f} ± {:.3f} Hz (n={})'.format(x, a, b, len(r[x])) for x, a, b in zip(degs, m, s))}


def slow_phase(fig, sess, segment='G2'):
    """Cells 147-148: slow-phase velocity and OKN gain."""
    d = _pd(sess, segment)
    v = pmod.slow_phase_velocity(d, pmod.grating_trials(sess, segment))
    gp = sess.cfg.grating_params
    degs = [x for x in sorted(v) if len(v[x])]
    if not degs:
        raise ValueError('No trials')
    cdp = float(sess.cfg.cam_deg_per_px)
    spd = [gp.get(x, {}).get('speed_dps', np.nan) for x in degs]
    a0, a1 = fig.subplots(1, 2)
    m = [np.mean(v[x]) for x in degs]; s = [np.std(v[x]) / np.sqrt(len(v[x])) for x in degs]
    a0.errorbar(degs, m, yerr=s, fmt='o-', ms=7, lw=2, color='steelblue', capsize=5)
    for x in degs:
        a0.scatter([x] * len(v[x]), v[x], s=20, alpha=0.4, color='steelblue')
    a0.axhline(np.mean(m), color='red', ls='--', label='grand mean {:.2f}px/s'.format(np.mean(m)))
    a0.set_xlabel('Grating size (deg)'); a0.set_ylabel('Slow-phase velocity (px/s)'); a0.legend(fontsize=8); a0.grid(alpha=0.2)
    a0.set_title('OKN slow-phase velocity vs grating')
    g = [np.mean(v[x]) * cdp / sp for x, sp in zip(degs, spd)]
    gs = [np.std(v[x]) * cdp / sp / np.sqrt(len(v[x])) for x, sp in zip(degs, spd)]
    a1.errorbar(degs, g, yerr=gs, fmt='o-', ms=7, lw=2, color='darkorange', capsize=5)
    a1.axhline(1.0, color='k', ls=':', lw=1)
    a1.set_xlabel('Grating size (deg)'); a1.set_ylabel('OKN gain (eye / grating speed)'); a1.grid(alpha=0.2)
    a1.set_title('OKN gain ({} deg/px)'.format(cdp))
    fig.tight_layout()
    return {'filename': 'okn_slow_phase_velocity_{}.png'.format(segment),
            'text': '\n'.join('{:>5.1f}: {:.2f} px/s  gain {:.3f}'.format(x, a, b) for x, a, b in zip(degs, m, g))}


def gaze_corrected_sweep(fig, sess, unit=None, bin_s=0.010, pre_s=0.5, post_s=4.0, smooth=2):
    """Cell 161: per-speed sweep PSTHs, raw vs gaze-corrected onsets."""
    d = _pd(sess, 'P1')
    b = sess.boundaries['P1']
    g = sess.sweep_geometry
    rf = sess.rf_clean.get(unit)
    if rf is None:
        raise ValueError('Unit {} has no clean RF'.format(unit))
    sp = sess.speed_dps
    bins = np.arange(-pre_s, post_s + bin_s, bin_s); bc = bins[:-1] + bin_s / 2; t = bc * 1000
    axes = fig.subplots(len(sp), 2, sharex=True, squeeze=False)
    st = sess.spike_trains[unit]
    for row, (cond, spd) in enumerate(sp.items()):
        on, off = sess.onsets('P1', cond)
        dur = float(np.mean(off - on))
        corr, gz = pmod.gaze_corrections(d, on, spd, float(sess.cfg.cam_deg_per_px), (b['start_s'], b['end_s']))
        col = sess.speed_colors.get(cond, 'k')
        rl = rf['x_deg'] - rf['diameter'] / 2
        lead = (rl - g['sweep_start_deg']) / spd * 1000
        trail = (rl + rf['diameter'] - g['sweep_start_deg']) / spd * 1000
        for k, ons in enumerate([on, on + corr]):
            ax = axes[row, k]
            c = trial_counts(st, ons, bins)
            m = gaussian_filter1d(c.mean(0) / bin_s, smooth)
            s = c.std(0) / np.sqrt(len(ons)) / bin_s
            ax.fill_between(t, np.clip(m - s, 0, None), m + s, alpha=0.25, color=col); ax.plot(t, m, lw=1.8, color=col)
            ax.axvline(0, color='k', ls='--', lw=1); ax.axvline(dur * 1000, color='gray', ls='--', lw=0.8)
            ax.axvspan(0, dur * 1000, alpha=0.05, color='gray')
            if 0 <= lead <= post_s * 1000:
                ax.axvline(lead, color=col, lw=2, alpha=0.8)
            if 0 <= trail <= post_s * 1000:
                ax.axvline(trail, color=col, ls=':', lw=1.5, alpha=0.7)
            ax.set_xlim(-pre_s * 1000, post_s * 1000); ax.set_ylim(bottom=0)
            ax.set_ylabel('{}dps\nFR (Hz)'.format(spd), fontsize=7)
            if row == 0:
                ax.set_title('Raw alignment' if k == 0 else 'Gaze-corrected', fontweight='bold')
    axes[-1, 0].set_xlabel('Time from onset (ms)'); axes[-1, 1].set_xlabel('Time from onset (ms)')
    fig.suptitle('Unit {} | RF={:.1f}° diam={:.1f}° {:.0f}µm   (← → keys step units)'.format(
        unit, rf['x_deg'], rf['diameter'], sess.unit_depths.get(unit, 0)), fontweight='bold')
    fig.subplots_adjust(hspace=0.45, wspace=0.15, top=0.92, bottom=0.06)
    return {'filename': 'gaze_corrected_sweep_u{}.png'.format(unit), 'height': 2.2 * len(sp) + 1}


def loom_pupil(fig, sess, condition='black_loom', trial=0, pre_s=0.5, post_s=3.0, show_raster=False):
    """Cells 163-165: x / y / diameter around loom onsets (one trial over the mean)."""
    d = _pd(sess, 'P1')
    on, off = sess.onsets('P1', condition)
    o = np.argsort(on)
    on, off = on[o], off[o]
    dur = float(np.mean(off - on))
    tr = [{'t_on': float(t), 'deg': 0, 'cond': condition} for t in on]
    out = {}
    for key in ('pupil_x_sm', 'pupil_y_sm', 'diam_sm'):
        tc, E, info = pmod.epoch(d, tr, pre_s, post_s, key)
        out[key] = (tc, pmod.baseline_sub(tc, E), info)
    tc, _, info = out['pupil_x_sm']
    if not info:
        raise ValueError('No pupil epochs for {}'.format(condition))
    trial = int(np.clip(trial, 0, len(info) - 1))
    azi = '—'
    for k, v in sess.cond_onsets['P1'].items():
        if k.startswith(condition + '_azi') and np.any(np.isclose(v, info[trial]['t_on'], atol=0.01)):
            azi = k.split('_')[-1]
    nrow = 4 if show_raster else 3
    axes = fig.subplots(nrow, 1, sharex=True)
    tm = tc * 1000
    for ax, key, yl, tt in zip(axes, ['pupil_x_sm', 'pupil_y_sm', 'diam_sm'], ['x (px)', 'y (px)', 'diameter (px)'],
                               ['Horizontal eye movement', 'Vertical eye movement', 'Pupil diameter']):
        _, E, _ = out[key]
        if not len(E):
            ax.text(0.5, 0.5, 'not available', transform=ax.transAxes, ha='center'); continue
        m, s = E.mean(0), E.std(0) / np.sqrt(len(E))
        ax.fill_between(tm, m - s, m + s, alpha=0.2, color='gray'); ax.plot(tm, m, lw=1.5, color='gray', label='mean all trials')
        ax.plot(tm, E[trial], lw=2, color='firebrick', label='trial {} ({})'.format(trial + 1, azi))
        ax.axvline(0, color='k', ls='--', lw=1.5); ax.axvline(dur * 1000, color='red', ls='--', lw=1.2)
        ax.axvspan(0, dur * 1000, alpha=0.07, color='firebrick'); ax.axhline(0, color='k', lw=0.5, ls=':')
        ax.set_ylabel(yl); ax.set_title(tt, fontsize=9, fontweight='bold'); ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.2)
    if show_raster:
        ax = axes[3]
        t0 = info[trial]['t_on']
        cids = [c for c in sess.good_ids_by_depth if np.isfinite(sess.unit_depths.get(c, np.nan))]
        for row, c in enumerate(cids):
            st = sess.spike_trains[c]
            s = (st[(st >= t0 - pre_s) & (st <= t0 + post_s)] - t0) * 1000
            ax.vlines(s, row, row + 0.8, lw=0.4, color='k', alpha=0.6)
        ax.set_ylabel('unit (by depth)'); ax.axvline(0, color='k', ls='--')
    axes[-1].set_xlabel('Time from loom onset (ms)')
    fig.suptitle('{} | trial {}/{} | t={:.1f}s | {}   (← → keys step trials)'.format(
        condition, trial + 1, len(info), info[trial]['t_on'], azi), fontweight='bold')
    fig.subplots_adjust(hspace=0.35, top=0.92)
    return {'filename': 'pupil_{}_trial{}.png'.format(condition, trial + 1), 'height': 3 * nrow}

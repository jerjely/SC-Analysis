"""Single-unit figures (cells 19, 34, 66, 68, 70)."""
import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..core import baden, stimuli
from ..core.psth import compute_psth, raster, draw_raster, compute_trial_psths_windowed


def _seg_cond(sess, segcond):
    seg, cond = segcond.split(':', 1)
    return seg, cond


def seg_conditions(sess):
    out = []
    try:
        for seg in sess.order:
            for c in sess.conditions(seg):
                if c == 'rfm_frames':
                    continue
                out.append('{}:{}'.format(seg, c))
    except Exception:
        pass
    return out


def unit_condition(fig, sess, unit=None, condition='P1:black_blink', pd_corrected=False,
                   pre_s=0.5, post_extra_s=1.0, sigma=1.5):
    """Cell 68 (+ optional photodiode-corrected onsets)."""
    seg, cond = _seg_cond(sess, condition)
    on, off = sess.onsets(seg, cond, pd_corrected=pd_corrected)
    ok = np.isfinite(off)
    on, off = on[ok], off[ok]
    dur = float(np.mean(off - on))
    post = dur + post_extra_s
    b = float(sess.cfg.bin_s)
    bins = np.arange(-pre_s, post + b, b)
    bc = bins[:-1] + b / 2
    st = sess.spike_trains[unit]
    m, s = compute_psth(st, on, bins, b)
    m = gaussian_filter1d(m, sigma=sigma)
    a0, a1 = fig.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    draw_raster(a0, raster(st, on, pre_s, post))
    for ax in (a0, a1):
        ax.axvspan(0, dur * 1000, alpha=0.07, color='gray')
        ax.axvline(0, color='k', ls='--', lw=1)
        ax.set_xlim(-pre_s * 1000, post * 1000)
    a0.set_ylabel('Trial')
    a0.set_ylim(-0.5, len(on))
    a0.set_title('Unit {} | {} | {} trials | {:.0f}µm{}'.format(
        unit, condition, len(on), sess.unit_depths.get(unit, 0), ' | PD-corrected' if pd_corrected else ''))
    a1.fill_between(bc * 1000, np.clip(m - s, 0, None), m + s, alpha=0.3, color='steelblue')
    a1.plot(bc * 1000, m, color='steelblue', lw=2)
    a1.set_ylabel('FR (Hz)'); a1.set_xlabel('Time from onset (ms)'); a1.set_ylim(0)
    fig.tight_layout()
    pk = int(np.argmax(m))
    return {'filename': 'unit{}_{}.png'.format(unit, cond),
            'text': 'trials={}  dur={:.3f}s\npeak {:.1f} Hz at {:.0f} ms\nbaseline {:.1f} Hz'.format(
                len(on), dur, m[pk], bc[pk] * 1000, m[bc < 0].mean() if (bc < 0).any() else np.nan)}


def ttl_vs_pd(fig, sess, units='', condition='small_sweep_30dps', pre_s=0.1, post_extra_s=0.5, sigma=1.5):
    """Cell 19: raster + PSTH with raw TTL vs photodiode-corrected onsets."""
    if sess.pd_onsets is None:
        raise ValueError('Run the PD correction step first')
    cids = [int(x) for x in str(units).replace(',', ' ').split() if x.strip().lstrip('u').isdigit()] \
        if units else list(sess.good_ids[:4])
    on_raw, off_raw = sess.onsets('P1', condition)
    on_pd, _ = sess.onsets('P1', condition, pd_corrected=True)
    ok = np.isfinite(off_raw)
    on_raw, off_raw, on_pd = on_raw[ok], off_raw[ok], on_pd[ok]
    dur = np.mean(off_raw - on_raw)
    post = dur + post_extra_s
    b = float(sess.cfg.bin_s)
    bins = np.arange(-pre_s, post + b, b)
    bc = bins[:-1] + b / 2
    axes = fig.subplots(len(cids), 4, squeeze=False)
    for row, cid in enumerate(cids):
        st = sess.spike_trains[cid]
        dep = sess.unit_depths.get(cid, np.nan)
        for k, (ons, lab, col) in enumerate([(on_raw, 'Raw TTL', '#F44336'), (on_pd, 'PD-corrected', '#2196F3')]):
            ar, ap = axes[row, 2 * k], axes[row, 2 * k + 1]
            draw_raster(ar, raster(st, ons, pre_s, post))
            m, s = compute_psth(st, ons, bins, b)
            ms = gaussian_filter1d(m, sigma=sigma)
            ap.fill_between(bc * 1000, np.clip(ms - s, 0, None), ms + s, alpha=0.3, color=col)
            ap.plot(bc * 1000, ms, color=col, lw=2)
            for ax in (ar, ap):
                ax.axvspan(0, dur * 1000, alpha=0.07, color='gray'); ax.axvline(0, color='k', ls='--', lw=1)
                ax.set_xlim(-pre_s * 1000, post * 1000); ax.tick_params(labelsize=6)
            ap.set_ylim(0)
            ar.set_title('u{} | {:.0f}µm\n{} raster'.format(cid, dep, lab), fontsize=8)
            ap.set_title('u{} | {:.0f}µm\n{} PSTH'.format(cid, dep, lab), fontsize=8)
    fig.suptitle('{} | Raw TTL vs PD-corrected'.format(condition), fontweight='bold')
    fig.subplots_adjust(hspace=0.6, wspace=0.35)
    d = (on_pd - on_raw) * 1000
    return {'filename': 'psth_raster_ttl_vs_pd_{}.png'.format(condition), 'height': 3.2 * len(cids),
            'text': 'n_trials={}  stim_dur={:.3f}s\nPD delay: mean {:.1f} ms, range {:.1f}-{:.1f} ms'.format(
                len(on_raw), dur, d.mean(), d.min(), d.max())}


def b1_sequence(fig, sess, unit=None, smooth_mult=4):
    """Cell 66: whole B1 repeat raster + PSTH with routine shading."""
    T = baden.get_timing(sess)
    df = T.df
    rep_on = df.groupby('repeat')['onset_s'].min().sort_index().values
    period = np.median(np.diff(rep_on)) if len(rep_on) > 1 else 120.0
    pre, post = 0.5, period + 0.5
    b = float(sess.cfg.bin_s)
    bins = np.arange(-pre, post + b, b)
    bc = bins[:-1] + b / 2
    st = sess.spike_trains[unit]
    m, s = compute_psth(st, rep_on, bins, b)
    m = gaussian_filter1d(m, sigma=float(sess.cfg.sigma) * smooth_mult)
    a0, a1 = fig.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    draw_raster(a0, raster(st, rep_on, pre, post), lw=0.5, alpha=0.8)
    rep0 = df[df['repeat'] == df['repeat'].min()]
    seen = set()
    for _, row in rep0.iterrows():
        t0 = row['onset_s'] - rep_on[0]
        dur = T.nominal[row['routine']] * T.scale
        c = baden.ROUTINE_COLORS.get(row['routine'], 'gray')
        lab = row['routine'] if row['routine'] not in seen else None
        seen.add(row['routine'])
        for ax in (a0, a1):
            ax.axvspan(t0 * 1000, (t0 + dur) * 1000, color=c, alpha=0.12, label=lab if ax is a0 else None)
    a0.axvline(0, color='k', ls='--', lw=1)
    a0.set_ylabel('Repeat')
    a0.set_title('Unit {} | B1 sequence | {} repeats | {:.0f}µm'.format(unit, len(rep_on),
                                                                         sess.unit_depths.get(unit, 0)))
    a0.legend(loc='upper right', fontsize=8, ncol=5)
    a1.fill_between(bc * 1000, np.clip(m - s, 0, None), m + s, alpha=0.3, color='steelblue')
    a1.plot(bc * 1000, m, color='steelblue', lw=1.5)
    a1.axvline(0, color='k', ls='--', lw=1)
    a1.set_ylabel('FR (Hz)'); a1.set_xlabel('Time from repeat start (ms)'); a1.set_ylim(0)
    for ax in (a0, a1):
        ax.set_xlim(-pre * 1000, post * 1000)
    fig.tight_layout()
    return {'filename': 'unit{}_B1_sequence.png'.format(unit)}


def baden_browser(fig, sess, unit=None, sigma=2.0):
    """Cell 34: chirp freq / contr / step / bar (RF-aligned) / colour."""
    T = baden.get_timing(sess)
    st = sess.spike_trains[unit]
    axes = fig.subplots(1, 5)
    B = T.bin
    F = sess.results.get('features')
    C = sess.results.get('clustering')
    in_set = F is not None and unit in F['valid_units']
    cstr = ''
    if in_set and C is not None:
        cstr = '  |  cluster {}'.format(C['labels'][F['valid_units'].index(unit)])
    fig.suptitle('Unit {}  |  {:.0f}µm{}{}'.format(
        unit, sess.unit_depths.get(unit, np.nan),
        ('  |  in clustering set' if in_set else ('  |  NOT in clustering set' if F is not None else '')),
        cstr), fontweight='bold')
    fr = gaussian_filter1d(compute_trial_psths_windowed(st, T.chirp_freq, T.freq_pre, T.freq_post, B).mean(0), sigma)
    cr = gaussian_filter1d(compute_trial_psths_windowed(st, T.chirp_contr, T.contr_pre, T.contr_post, B).mean(0), sigma)
    axes[0].plot(fr, 'k', lw=1); axes[0].axvline(T.freq_pre / B, color='0.5', ls='--', lw=0.7)
    axes[0].set_title('chirp: freq', fontsize=9); axes[0].set_ylabel('FR (Hz)')
    axes[1].plot(cr, 'k', lw=1); axes[1].axvline(T.contr_pre / B, color='0.5', ls='--', lw=0.7)
    axes[1].set_title('chirp: contr', fontsize=9)
    sr = gaussian_filter1d(compute_trial_psths_windowed(st, T.step_on, T.step_pre, T.step_post, B).mean(0), sigma)
    axes[2].plot(sr, 'k', lw=1)
    for x in (T.step_pre, T.step_pre + T.step_dur, T.step_pre + T.step_dur + T.step_gap * T.scale):
        axes[2].axvline(x / B, color='0.5', ls='--', lw=0.7)
    axes[2].set_title('step', fontsize=9)
    rf = sess.rf_clean.get(unit)
    for d in T.dirs:
        if rf and rf.get('fit_ok'):
            o = T.bar_by_dir[d] + T.bar_cross(rf['x_deg'], rf['y_deg'], d) * T.scale
        else:
            o = T.bar_by_dir[d]
        br = gaussian_filter1d(compute_trial_psths_windowed(st, o, T.bar_align_pre, T.bar_align_post, B).mean(0), sigma)
        axes[3].plot(br, color=baden.BAR_DIR_COLORS.get(d), lw=1, label='{}°'.format(d))
    axes[3].axvline(T.bar_align_pre / B, color='0.5', ls='--', lw=0.7)
    axes[3].set_title('bar' + (' (RF-aligned)' if rf and rf.get('fit_ok') else ' (sweep onset, no RF)'), fontsize=9)
    axes[3].legend(fontsize=6, ncol=2)
    colr = gaussian_filter1d(compute_trial_psths_windowed(st, T.blue, T.color_pre, T.color_post, B).mean(0), sigma)
    g0 = int(round((T.color_pre + T.color_dur + T.color_gap * T.scale) / B))
    axes[4].plot(range(g0), colr[:g0], color='tab:blue', lw=1)
    axes[4].plot(range(g0, len(colr)), colr[g0:], color='tab:green', lw=1)
    axes[4].axvline(int(round(T.color_pre / B)), color='0.5', ls='--', lw=0.7); axes[4].axvline(g0, color='0.5', ls='--', lw=0.7)
    axes[4].set_title('colour', fontsize=9)
    for ax in axes:
        ax.set_xlabel('bin ({:.0f} ms)'.format(B * 1000))
    fig.subplots_adjust(left=0.05, right=0.98, top=0.82, bottom=0.15, wspace=0.35)
    qi = sess.results.get('qi', {}).get(unit)
    return {'filename': 'unit{}_baden.png'.format(unit), 'height': 3.6,
            'text': 'QI = {}'.format('{:.3f}'.format(qi) if qi is not None else '(compute QI in Clustering tab)')}


def grating_sizes(sess):
    """Sizes from the parsed condition names (grating_5.0deg), else the CSV."""
    sizes = set()
    try:
        for seg in sess.order:
            if seg.startswith('G'):
                for k in sess.cond_onsets.get(seg, {}):
                    if k.startswith('grating_') and k.endswith('deg'):
                        sizes.add(float(k[len('grating_'):-3]))
    except Exception:
        pass
    if not sizes:
        try:
            seg = [s for s in sess.order if s.startswith('G')][0]
            sizes = set(stimuli.grating_sizes(sess.cfg, seg))
        except Exception:
            pass
    return sorted(sizes)


def grating_directions(sess):
    """Drift directions that actually have trials (drift_pos / drift_neg)."""
    dirs = []
    try:
        for d in ('pos', 'neg'):
            if any(len(sess.cond_onsets.get(seg, {}).get('drift_' + d, [])) > 0
                   for seg in sess.order if seg.startswith('G')):
                dirs.append(d)
    except Exception:
        pass
    return dirs or ['pos', 'neg']


def _check_dir(sess, direction):
    try:
        have = {d for d in ('pos', 'neg')
                if any(len(sess.cond_onsets.get(seg, {}).get('drift_' + d, [])) > 0
                       for seg in sess.order if seg.startswith('G'))}
    except Exception:
        return
    if have and direction not in have:
        raise ValueError('This recording has no "{}" drift trials (only: {}).'.format(
            direction, ', '.join(sorted(have))))


def _size(size_deg):
    try:
        return float(size_deg)
    except (TypeError, ValueError):
        raise ValueError('No grating sizes found - run pipeline step "0C Parse PsychoPy" and check the '
                         'G1/G2 PsychoPy folders contain the grating CSV (see the Log).')


def _rf_edge_times(sf, tf, edge_deg, direction, t_start=-1.0, t_end=11.0):
    out = []
    nr = int((t_end - t_start) * tf) + 10
    nc = sf * edge_deg - 0.5
    for n in range(int(nc) - nr, int(nc) + nr):
        t = (0.5 + n - sf * edge_deg) / tf if direction == 'pos' else (0.5 + n + sf * edge_deg) / tf
        if t_start <= t <= t_end:
            out.append(t)
    return np.array(sorted(out))


def grating_unit(fig, sess, unit=None, size_deg=5.0, direction='pos'):
    _check_dir(sess, direction)
    """Cell 70: G1 vs G2 raster/PSTH with RF edge crossing times."""
    size_deg = _size(size_deg)
    gsegs = [s for s in sess.order if s.startswith('G') and s[1:].isdigit()][:2]
    gp = sess.cfg.grating_params
    key = min(gp, key=lambda k: abs(k - size_deg))
    sf, tf = gp[key]['sf'], gp[key]['tf']
    rf = sess.rf_clean.get(unit)
    if rf:
        rx, rd = rf['x_deg'], rf['diameter']; note = ''
    else:
        xs = [r['x_deg'] for r in sess.rf_clean.values()] or [0]
        ds = [r['diameter'] for r in sess.rf_clean.values()] or [10]
        rx, rd = np.median(xs), np.median(ds); note = ' (no clean RF: population median)'
    enter = rx - rd / 2 if direction == 'pos' else rx + rd / 2
    exitv = rx + rd / 2 if direction == 'pos' else rx - rd / 2
    te, tx = _rf_edge_times(sf, tf, enter, direction), _rf_edge_times(sf, tf, exitv, direction)
    dur, pre, post = 10.0, 1.0, 11.0
    b = float(sess.cfg.bin_s)
    bins = np.arange(-pre, post + b, b); bc = bins[:-1] + b / 2
    st = sess.spike_trains[unit]
    axes = fig.subplots(2, len(gsegs), squeeze=False, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    cols = ['steelblue', 'darkorange']
    for k, seg in enumerate(gsegs):
        df = stimuli.load_grating_final(sess.cfg.seg_psychopy_folder(seg), size_deg, direction)
        on = stimuli.align_grating_onsets(sess.cfg, df, seg, sess.seg_onsets, sess.boundaries, log=lambda *a: None)
        ar, ap = axes[0, k], axes[1, k]
        draw_raster(ar, raster(st, on, pre, post), alpha=0.8)
        for t in te:
            ar.axvline(t * 1000, color='purple', lw=0.8, alpha=0.35); ap.axvline(t * 1000, color='purple', lw=0.8, alpha=0.4)
        for t in tx:
            ar.axvline(t * 1000, color='darkorchid', ls='--', lw=0.6, alpha=0.25)
            ap.axvline(t * 1000, color='darkorchid', ls='--', lw=0.6, alpha=0.3)
        ar.axvspan(0, dur * 1000, alpha=0.05, color=cols[k]); ap.axvspan(0, dur * 1000, alpha=0.05, color=cols[k])
        ar.set_ylim(-0.5, max(len(on), 1) - 0.5); ar.set_ylabel('Trial')
        ar.set_title('{} | {}deg | n={}'.format(seg, size_deg, len(on)), color=cols[k], fontweight='bold')
        m, _ = compute_psth(st, on, bins, b)
        ap.plot(bc * 1000, gaussian_filter1d(m, 2.0), color=cols[k], lw=2)
        ap.set_ylim(0); ap.set_ylabel('FR (Hz)'); ap.set_xlabel('Time from onset (ms)')
    for ax in axes.flat:
        ax.set_xlim(-pre * 1000, post * 1000)
    fig.suptitle('Unit {} | RF={:.1f}deg{} | {}deg | SF={:.3f}cpd TF={}Hz speed={:.0f}dps'.format(
        unit, rx, note, size_deg, sf, tf, gp[key]['speed_dps']), fontweight='bold')
    fig.tight_layout()
    return {'filename': 'unit{}_grating_{}deg_{}.png'.format(unit, size_deg, direction)}

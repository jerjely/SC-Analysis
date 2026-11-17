"""All analysis tabs (built from the plot functions + parameter specs)."""
import numpy as np
from PySide6 import QtCore, QtWidgets

from ..core import baden, ccf as ccfmod, pupil as pmod
from ..plots import (rf_plots, unit_plots, pop_plots, baden_plots, sweep_plots, loom_plots, ccf_plots,
                     pupil_plots, stim_plots)
from .common import FigurePanel, Param as P, unit_choices, log, show_error


def _p1(sess):
    return [c for c in sess.conditions('P1')]


def _speed_conds(sess):
    try:
        return list(sess.speed_dps)
    except Exception:
        return []


def _clean_units(sess):
    try:
        return unit_choices(sess, [c for c in sess.good_ids_by_depth if c in sess.rf_clean])
    except Exception:
        return []


def _units(sess):
    try:
        return unit_choices(sess, sess.good_ids_by_depth)
    except Exception:
        return []


def _g_segs(sess):
    try:
        return [s for s in sess.order if s.startswith('G')]
    except Exception:
        return ['G1', 'G2']


class SubTabs(QtWidgets.QTabWidget):
    """A tab widget whose pages are FigurePanels (refreshed together)."""

    def __init__(self, sess_getter, parent=None):
        super().__init__(parent)
        self.sess_getter = sess_getter
        self.panels = []
        self.setDocumentMode(True)
        self.currentChanged.connect(self._on_change)

    def add(self, name, fn, params=(), help_text='', auto=False, extra_buttons=()):
        p = FigurePanel(self.sess_getter, fn, params, title=name, help_text=help_text, auto=auto,
                        extra_buttons=extra_buttons)
        self.panels.append(p)
        self.addTab(p, name)
        return p

    def refresh(self):
        for p in self.panels:
            p.refresh_choices()

    def _on_change(self, i):
        w = self.widget(i)
        if isinstance(w, FigurePanel):
            w.refresh_choices()


# ---------------------------------------------------------------------------
class OverviewTab(SubTabs):
    def __init__(self, sg, parent=None):
        super().__init__(sg, parent)
        self.add('Session timeline', stim_plots.timeline, (),
                 'Segment layout, TTL onsets and the parsed stimulus sequence (Cell 20).')
        self.add('Units', stim_plots.unit_overview, (), 'Kilosort units: depth vs firing rate.')
        self.add('B1 vs protocol', baden_plots.b1_diagnostic, (),
                 'Cells 8-9: photodiode-recovered B1 edges vs the protocol reference timeline.')
        self.add('PD delays', stim_plots.pd_delays, (), 'TTL → screen delay from the photodiode (P1).')


class RFTab(SubTabs):
    def __init__(self, sg, parent=None):
        super().__init__(sg, parent)
        self.add('Mode maps', rf_plots.rf_mode_maps, [P('min_snr', 'Min SNR', 'float', 6.0),
                                                      P('max_diam', 'Max diameter', 'float', 30.0)],
                 'Cell 1B: RF ellipses for each RFM config and the current clean set.')
        self.add('Interactive', rf_plots.rf_interactive, [
            P('depth_min', 'Depth min', 'int', 0), P('depth_max', 'Depth max', 'int', 2500),
            P('snr_min', 'SNR min', 'float', 6.0),
            P('rf_set', 'RF set', 'choice', 'clean', choices=['clean', 'RFM1', 'RFM2', 'combined']),
            P('sta_set', 'STA', 'choice', 'RFM1', choices=['RFM1', 'RFM2'])],
            'Cell 30: click any contour on the left to see its STA.')
        self.add('Depth-filtered', rf_plots.rf_depth_filtered, [
            P('depth_min', 'Depth min', 'int', 0), P('depth_max', 'Depth max', 'int', 2000),
            P('snr_min', 'SNR min', 'float', 6.0)], 'Cell 29.')
        self.add('Examples', rf_plots.rf_examples, [
            P('depth_min', 'Depth min', 'int', 0), P('depth_max', 'Depth max', 'int', 2500),
            P('snr_min', 'SNR min', 'float', 6.0), P('n_examples', 'Examples', 'int', 4, minv=1, maxv=8)],
            'Cell 31: RF-vs-depth map with example STAs.')
        self.add('STA browser', rf_plots.sta_browser, [
            P('n_show', 'Units per page', 'int', 12, minv=1, maxv=40), P('start', 'Start rank', 'int', 0, minv=0)],
            'Cell 75: best clean units, RFM1 vs RFM2.')
        self.add('STA lags', rf_plots.sta_lags, [P('unit', 'Unit', 'unit', choices=_units),
                                                 P('sta_set', 'STA', 'choice', 'RFM1', choices=['RFM1', 'RFM2'])],
                 'All STA lags of one unit (← → step units).', auto=True)
        self.add('Retinotopy', rf_plots.retinotopy_summary, [P('depth_cutoff', 'Depth cutoff', 'int', 3000)],
                 'Cell 77.')
        self.add('Contours', rf_plots.rf_contours, [
            P('depth_min', 'Depth min', 'int', 0), P('depth_max', 'Depth max', 'int', 3000),
            P('flip_y', 'Screen orientation (flip y)', 'bool', False)], 'Cells 78-79.')


class UnitTab(SubTabs):
    def __init__(self, sg, parent=None):
        super().__init__(sg, parent)
        self.cond_panel = self.add('Condition', unit_plots.unit_condition, [
            P('unit', 'Unit', 'unit', choices=_units),
            P('condition', 'Condition', 'choice', 'P1:black_blink', choices=unit_plots.seg_conditions),
            P('pd_corrected', 'PD-corrected (P1)', 'bool', False),
            P('pre_s', 'Pre (s)', 'float', 0.5), P('post_extra_s', 'Post extra (s)', 'float', 1.0),
            P('sigma', 'Smoothing (bins)', 'float', 1.5)], 'Cell 4: raster + PSTH. ← → step units.', auto=True)
        self.add('B1 sequence', unit_plots.b1_sequence, [P('unit', 'Unit', 'unit', choices=_units),
                                                         P('smooth_mult', 'Smoothing x', 'float', 4.0)],
                 'Cell 4C: whole Baden repeat.', auto=True)
        self.add('Baden browser', unit_plots.baden_browser, [P('unit', 'Unit', 'unit', choices=_units),
                                                             P('sigma', 'Smoothing (bins)', 'float', 2.0)],
                 'Chirp / step / RF-aligned bar / colour responses.', auto=True)
        self.add('Gratings G1 vs G2', unit_plots.grating_unit, [
            P('unit', 'Unit', 'unit', choices=_units),
            P('size_deg', 'Grating size (deg)', 'choice', 5.0, choices=unit_plots.grating_sizes),
            P('direction', 'Drift', 'choice', 'pos', choices=unit_plots.grating_directions)], 'Cell 5.', auto=True)
        self.add('TTL vs PD', unit_plots.ttl_vs_pd, [
            P('units', 'Units (ids, blank = first 4)', 'str', ''),
            P('condition', 'P1 condition', 'choice', 'small_sweep_30dps', choices=_p1),
            P('pre_s', 'Pre (s)', 'float', 0.1), P('post_extra_s', 'Post extra (s)', 'float', 0.5),
            P('sigma', 'Smoothing', 'float', 1.5)], 'Raw TTL vs photodiode-corrected alignment.')

    def show_unit(self, cid):
        for p in self.panels:
            if 'unit' in p.form.widgets:
                p.refresh_choices()
                p.form.set_value('unit', int(cid))
        self.setCurrentIndex(0)
        self.panels[0].redraw()


class PopulationTab(SubTabs):
    def __init__(self, sg, parent=None):
        super().__init__(sg, parent)
        self.add('Condition heatmaps', pop_plots.condition_heatmaps, [
            P('conditions', 'Conditions (multi-select)', 'multi', ['P1:large_sweep'], choices=unit_plots.seg_conditions),
            P('depth_min', 'Depth min', 'int', 0), P('depth_max', 'Depth max', 'int', 3000),
            P('pre_s', 'Pre (s)', 'float', 0.5), P('post_extra_s', 'Post extra (s)', 'float', 1.0),
            P('sigma', 'Smoothing', 'float', 1.5), P('pd_corrected', 'PD-corrected (P1)', 'bool', False),
            P('zmax', '|z| max', 'float', 4.0)], 'Cells 3A/3B: z-scored PSTH by depth, one panel per condition.')
        self.add('B1 heatmap', pop_plots.b1_heatmap, [
            P('routine', 'Routine:condition', 'choice', 'freq_chirp:', choices=pop_plots.b1_routines),
            P('depth_min', 'Depth min', 'int', 0), P('depth_max', 'Depth max', 'int', 3000),
            P('sigma', 'Smoothing', 'float', 1.5)], 'Cell 3C.')
        self.add('Sweep heatmaps', pop_plots.sweep_heatmaps, [
            P('depth_min', 'Depth min', 'int', 0), P('depth_max', 'Depth max', 'int', 3000)], 'Cell S1.')
        self.add('Grating heatmap', pop_plots.grating_heatmap, [
            P('size_deg', 'Grating size (deg)', 'choice', 5.0, choices=unit_plots.grating_sizes),
            P('direction', 'Drift', 'choice', 'pos', choices=unit_plots.grating_directions), P('sigma', 'Smoothing', 'float', 3.0)],
            'Cell 6.')


class ClusteringTab(QtWidgets.QWidget):
    def __init__(self, sg, runner, parent=None):
        super().__init__(parent)
        self.sg = sg
        self.runner = runner
        lay = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QGroupBox('Compute')
        left.setMaximumWidth(250)
        ll = QtWidgets.QVBoxLayout(left)
        self.status = QtWidgets.QLabel('')
        self.status.setWordWrap(True)
        steps = [('1. Quality index', self._qi), ('2. Feature vectors', self._feat),
                 ('3. DS/OS (pooled baseline)', self._ds2), ('   DS/OS (per-trial, suppl.)', self._ds1),
                 ('4. Colour opponency', self._col), ('5. SparsePCA + GMM', self._clu),
                 ('6. Export for pooling', self._exp)]
        for lab, fn in steps:
            b = QtWidgets.QPushButton(lab)
            b.setStyleSheet('text-align:left; padding:4px')
            b.clicked.connect(fn)
            ll.addWidget(b)
        ball = QtWidgets.QPushButton('Run 1-6')
        ball.setStyleSheet('font-weight:bold; padding:6px')
        ball.clicked.connect(self._all)
        ll.addWidget(ball)
        ll.addWidget(self.status)
        note = QtWidgets.QLabel('Speed tuning comes from the sweep picks (Pickers tab). '
                                'Thresholds / flips / permutations live in Session settings > Analysis.')
        note.setWordWrap(True); note.setStyleSheet('color:#666')
        ll.addWidget(note)
        ll.addStretch(1)
        lay.addWidget(left)
        self.tabs = SubTabs(sg)
        t = self.tabs
        t.add('QI', baden_plots.qi_hist, (), 'Cell 44.')
        t.add('BIC', baden_plots.bic_curve, (), 'Model order selection.')
        t.add('Cluster traces', baden_plots.cluster_traces, [P('clusters', 'Clusters (blank = all)', 'str', '')],
              'Cell 55: mean ± SEM feature segments per cluster.')
        t.add('Feature matrix', baden_plots.cluster_heat, (), 'Every unit sorted by cluster.')
        t.add('Cluster profile', self._profile_fig, (), 'Cell 54: DS/OS, colour and speed per cluster.')
        t.add('DS/OS', baden_plots.dsos_hist, [
            P('version', 'Baseline', 'choice', 'pooled', choices=['pooled', 'per-trial']),
            P('filtered', 'Activity-filtered', 'bool', True), P('dsi_thresh', 'DSI thresh', 'float', 0.3),
            P('osi_thresh', 'OSI thresh', 'float', 0.3), P('p_thresh', 'p thresh', 'float', 0.05)], 'Cells 58-64.')
        t.add('Colour', baden_plots.colour_scatter, (), 'Cell 51 colour opponency.')
        lay.addWidget(t, 1)

    def _profile_fig(self, fig, sess):
        df = baden.cluster_profile(sess)
        av = baden.availability_table(sess)
        ax = fig.add_subplot(111); ax.axis('off')
        tb = ax.table(cellText=[[('{:.2f}'.format(v) if isinstance(v, float) else str(v)) for v in r]
                                for r in df.reset_index().values],
                      colLabels=['cluster'] + list(df.columns), loc='center', cellLoc='center')
        tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.3)
        import pandas as pd
        with pd.option_context('display.width', 200, 'display.max_columns', 20):
            txt = df.to_string(float_format=lambda x: '{:.2f}'.format(x)) + '\n\nData availability:\n' + \
                av.to_string(float_format=lambda x: '{:.2f}'.format(x))
        return {'text': txt, 'filename': 'cluster_profile.png'}

    def _run(self, label, fn, then=None):
        sess = self.sg()

        def done(_):
            self._update()
            self.tabs.refresh()
            if then:
                then()
        self.runner.run(label, fn, sess, on_done=done)

    def _qi(self, then=None):
        self._run('QI', baden.run_qi, then)

    def _feat(self, then=None):
        self._run('Feature vectors', baden.run_features, then)

    def _ds2(self, then=None):
        self._run('DS/OS pooled', lambda s, log=print, progress=None: baden.run_dsos(s, log, progress, True), then)

    def _ds1(self, then=None):
        self._run('DS/OS per-trial', lambda s, log=print, progress=None: baden.run_dsos(s, log, progress, False), then)

    def _col(self, then=None):
        self._run('Colour', baden.run_colour, then)

    def _clu(self, then=None):
        self._run('Clustering', baden.run_clustering, then)

    def _exp(self, then=None):
        self._run('Export', lambda s, log=print, progress=None: baden.run_export(s, log), then)

    def _all(self):
        chain = [self._qi, self._feat, self._ds2, self._col, self._clu, self._exp]

        def step(i=0):
            if i < len(chain):
                chain[i](then=lambda: step(i + 1))
        step()

    def _update(self):
        r = self.sg().results
        bits = []
        if 'qi' in r:
            bits.append('QI: {} units'.format(len(r['qi'])))
        if 'features' in r:
            bits.append('features: {} units'.format(len(r['features']['valid_units'])))
        if 'dsos_v2' in r:
            bits.append('DS/OS: {} reliable'.format(len(r['dsos_v2_filtered'])))
        if 'colour' in r:
            bits.append('colour: {} reliable'.format(sum(x['reliable'] for x in r['colour'].values())))
        if 'clustering' in r:
            bits.append('clusters: {}'.format(r['clustering']['best_k']))
        self.status.setText('\n'.join(bits))

    def refresh(self):
        self._update()
        self.tabs.refresh()


class SweepTab(SubTabs):
    def __init__(self, sg, runner, parent=None):
        super().__init__(sg, parent)
        self.runner = runner
        self.add('Latency vs depth', sweep_plots.latency_vs_depth, [
            P('use_exclusions', 'Apply saved exclusions', 'bool', True), P('size_scale', 'Marker scale', 'float', 5.0)],
            'Cells S2/S3: click a dot to exclude that cell from all speeds (saved immediately); right-click restores.')
        self.add('Depth vs arrival', sweep_plots.depth_vs_arrival, [
            P('condition', 'Condition', 'choice', 'small_sweep_25dps', choices=_speed_conds),
            P('depth_min', 'Depth min', 'int', 0), P('depth_max', 'Depth max', 'int', 3000)], 'Cell S4.')
        self.add('Population summary', sweep_plots.population_summary, [
            P('use_clean', 'Drop excluded cells', 'bool', True)], 'Cell S5.')
        self.add('Cell picks', sweep_plots.cell_picks, [P('unit', 'Picked unit', 'unit', choices=sweep_plots.picked_cells)],
                 'Cell S7 (← → step cells).', auto=True,
                 extra_buttons=[('Export all cell plots (S6)', self._export)])

    def _export(self):
        self.runner.run('Export sweep cell plots', sweep_plots.export_all_cell_plots, self.sess_getter())


class LoomTab(SubTabs):
    def __init__(self, sg, parent=None):
        super().__init__(sg, parent)
        lc = loom_plots.picked_conditions
        self.add('Single condition', loom_plots.latency_vs_depth_single,
                 [P('condition', 'Condition', 'choice', 'black_loom_azi+0', choices=lc)], 'Cell L1.')
        self.add('On RF map', loom_plots.latency_on_rf_map,
                 [P('condition', 'Condition', 'choice', 'black_loom_azi+0', choices=lc)], 'Cell L2.')
        self.add('Multi condition', loom_plots.latency_vs_depth_multi, [
            P('conditions', 'Conditions (blank = all)', 'multi', [], choices=lc),
            P('hulls', 'Convex hulls by azimuth', 'bool', False), P('rf_bars', 'RF-size bars', 'bool', False),
            P('hull_clip', 'Hull percentile clip', 'float', 2.5)], 'Cells L3 / L4 / R1.')
        self.add('Bar plots', loom_plots.loom_bars, [P('conditions', 'Conditions (blank = all)', 'multi', [], choices=lc)],
                 'Cell L5.')
        self.add('Classes', loom_plots.classification_summary, (), 'ON / OFF / ON-OFF counts from the picker.')


class CCFTab(QtWidgets.QWidget):
    def __init__(self, sg, runner, parent=None):
        super().__init__(parent)
        self.sg, self.runner = sg, runner
        lay = QtWidgets.QVBoxLayout(self)
        h = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton('Compute CCFs (Cell C1)')
        b.clicked.connect(self._run)
        b2 = QtWidgets.QPushButton('Load saved')
        b2.clicked.connect(self._load)
        self.lab = QtWidgets.QLabel('Segment, z threshold and min FR are in Session settings > CCF & pupil.')
        self.lab.setStyleSheet('color:#666')
        h.addWidget(b); h.addWidget(b2); h.addWidget(self.lab, 1)
        lay.addLayout(h)
        self.tabs = SubTabs(sg)
        self.tabs.add('Pairs on RF map', ccf_plots.pairs_on_rf_map, [P('consistent_only', 'Consistent cells only (C3)', 'bool', False)],
                      'Cell C2 / C3.')
        self.tabs.add('Examples', ccf_plots.ccf_examples, [P('n_examples', 'Examples per type', 'int', 4, minv=1, maxv=8)], 'Cell C4.')
        self.tabs.add('z vs distance', ccf_plots.z_vs_distance, (), 'Cell C5.')
        self.tabs.add('Far pair', ccf_plots.far_pair, [P('rf_dist_min', 'Min RF distance (deg)', 'float', 5.0),
                                                       P('rank', 'Rank', 'int', 0, minv=0)], 'Highest-z pair with distant RFs.')
        self.tabs.add('Any pair', ccf_plots.ccf_pair, [
            P('unit', 'Unit A', 'unit', choices=_units), P('unit_b', 'Unit B', 'choice', choices=_units),
            P('segment', 'Segment', 'choice', 'S1', choices=lambda s: s.order),
            P('max_lag_ms', 'Max lag (ms)', 'float', 50.0), P('bin_ms', 'Bin (ms)', 'float', 1.0)], '')
        lay.addWidget(self.tabs, 1)

    def _run(self):
        self.runner.run('CCF', ccfmod.run_ccf, self.sg(), cancel=True, on_done=lambda _: self._after())

    def _load(self):
        r = ccfmod.load_ccf(self.sg())
        if r is None:
            QtWidgets.QMessageBox.information(self, 'CCF', 'No saved CCF results for the current settings.')
        else:
            log('Loaded {} significant pairs'.format(len(r['sig_pairs'])))
            self._after()

    def _after(self):
        r = self.sg().results.get('ccf')
        if r:
            self.lab.setText('{} significant / {:,} tested pairs (z ≥ {})'.format(len(r['sig_pairs']), r['n_pairs'], r['sig_z']))
        self.tabs.refresh()
        self.tabs.currentWidget().redraw()

    def refresh(self):
        self.tabs.refresh()


class PupilTab(QtWidgets.QWidget):
    def __init__(self, sg, runner, parent=None):
        super().__init__(parent)
        self.sg, self.runner = sg, runner
        lay = QtWidgets.QVBoxLayout(self)
        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QLabel('Segment'))
        self.seg = QtWidgets.QComboBox()
        h.addWidget(self.seg)
        self.force = QtWidgets.QCheckBox('Re-read screen video')
        h.addWidget(self.force)
        b = QtWidgets.QPushButton('Align pupil to imec')
        b.setToolTip('Grating segments: drift-onset grid fit (cell 152). P1: white blink + loom matching (cells 155-158).')
        b.clicked.connect(self._align)
        h.addWidget(b)
        b2 = QtWidgets.QPushButton('Load saved alignment')
        b2.clicked.connect(self._load)
        h.addWidget(b2)
        self.lab = QtWidgets.QLabel('Needs <seg>.avi, <seg>_screen.avi and a DLC .h5 in the PsychoPy segment folder.')
        self.lab.setStyleSheet('color:#666')
        h.addWidget(self.lab, 1)
        lay.addLayout(h)
        t = self.tabs = SubTabs(sg)
        segp = lambda d='G2': P('segment', 'Segment', 'choice', d, choices=pupil_plots.pupil_segments)
        t.add('Alignment check', pupil_plots.alignment_check, [segp()], '')
        t.add('Per-size mean', pupil_plots.per_sf_mean, [segp(), P('pre_s', 'Pre (s)', 'float', 2.0),
                                                        P('post_s', 'Post (s)', 'float', 12.0)], 'Cell PUP3.')
        t.add('Trial browser', pupil_plots.trial_browser, [segp(), P('trial', 'Trial', 'int', 0, minv=0)],
              'Cell PUP4 (← → step trials).', auto=True)
        t.add('Raster + pupil', pupil_plots.raster_pupil, [
            P('segments', 'Segments', 'multi', ['G1', 'G2'], choices=_g_segs),
            P('sf', 'Grating size', 'choice', 4.0, choices=pupil_plots.all_sf), P('trial', 'Trial', 'int', 0, minv=0),
            P('bin_s', 'PSTH bin (s)', 'float', 0.05)], 'Cells PUP5 / 153 (← → step trials).', auto=True)
        t.add('Neural osc. vs size', pupil_plots.neural_osc_vs_sf, [
            P('segment', 'Segment', 'choice', 'G2', choices=_g_segs), P('bin_s', 'Bin (s)', 'float', 0.033),
            P('fmin', 'f min', 'float', 0.2), P('fmax', 'f max', 'float', 6.0)], 'Cell 142 (no pupil needed).')
        t.add('OKN freq vs size', pupil_plots.okn_freq_vs_sf, [
            P('segment', 'Segment', 'choice', 'G2', choices=_g_segs), P('fmin', 'f min', 'float', 0.1),
            P('fmax', 'f max', 'float', 12.0)], 'Cell 143.')
        t.add('OKN reset rate', pupil_plots.okn_reset_rate, [
            P('segment', 'Segment', 'choice', 'G2', choices=_g_segs),
            P('vel_thresh_std', 'Velocity thr (SD)', 'float', 3.0), P('min_sep_s', 'Min separation (s)', 'float', 0.5)],
            'Cell 146.')
        t.add('Slow phase / gain', pupil_plots.slow_phase, [P('segment', 'Segment', 'choice', 'G2', choices=_g_segs)],
              'Cells 147-148.')
        t.add('Gaze-corrected sweep', pupil_plots.gaze_corrected_sweep, [
            P('unit', 'Unit (clean RF)', 'unit', choices=_clean_units), P('bin_s', 'Bin (s)', 'float', 0.010),
            P('post_s', 'Post (s)', 'float', 4.0), P('smooth', 'Smoothing (bins)', 'float', 2.0)],
            'Cell 161 (needs P1 pupil). ← → step units.', auto=True)
        t.add('Loom pupil', pupil_plots.loom_pupil, [
            P('condition', 'Condition', 'choice', 'black_loom', choices=_p1), P('trial', 'Trial', 'int', 0, minv=0),
            P('show_raster', 'Show raster', 'bool', False)], 'Cells 163-165 (← → step trials).', auto=True)
        lay.addWidget(t, 1)

    def refresh(self):
        sess = self.sg()
        self.seg.clear()
        try:
            for s in sess.order:
                if s.startswith('G') or s == 'P1':
                    self.seg.addItem(s)
        except Exception:
            pass
        self.tabs.refresh()

    def _align(self):
        sess = self.sg()
        seg = self.seg.currentText()
        if not seg:
            return
        fn = pmod.align_p1_segment if seg == 'P1' else pmod.align_grating_segment
        self.runner.run('Pupil alignment {}'.format(seg), fn, sess, seg, force=self.force.isChecked(), cancel=True,
                        on_done=lambda _: (self.tabs.refresh(), self.lab.setText('{} aligned'.format(seg))))

    def _load(self):
        sess = self.sg()
        seg = self.seg.currentText()
        try:
            d = pmod.load_pupil(sess, seg)
        except Exception as e:
            show_error(self, 'Pupil', str(e)); return
        if d is None:
            QtWidgets.QMessageBox.information(self, 'Pupil', 'No saved alignment / DLC file for {}'.format(seg))
        else:
            self.lab.setText('{} loaded: {} frames'.format(seg, len(d['pupil_imec_t'])))
            self.tabs.refresh()

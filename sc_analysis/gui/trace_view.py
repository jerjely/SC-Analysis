"""
Recording tab: read-only Neuropixels trace viewer for the concatenated,
sorted recording.  Channels are grouped and coloured by shank and ordered by
depth; Kilosort units are overlaid on their best channel; segment boundaries
and stimulus TTL onsets (with condition names) are drawn on top.
"""
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from scipy.signal import butter, sosfiltfilt

from ..core import npx_io
from .common import log, show_error

SHANK_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#17becf', '#8c564b', '#e377c2']
UNIT_COLORS = [QtGui.QColor(c) for c in (
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c',
    '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
    '#000075', '#808080')]
SEG_BRUSH = {'P': (31, 119, 180, 25), 'R': (44, 160, 44, 25), 'B': (148, 103, 189, 25), 'G': (255, 127, 14, 25),
             'S': (127, 127, 127, 25)}


class TracePlot(pg.PlotWidget):
    wheel_time = QtCore.Signal(float)
    wheel_gain = QtCore.Signal(float)

    def wheelEvent(self, ev):
        d = ev.angleDelta().y() / 120.0
        if ev.modifiers() & QtCore.Qt.ControlModifier:
            self.wheel_gain.emit(d)
        elif ev.modifiers() & QtCore.Qt.ShiftModifier:
            super().wheelEvent(ev)
        else:
            self.wheel_time.emit(-d)
        ev.accept()


class RecordingTab(QtWidgets.QWidget):
    open_unit = QtCore.Signal(int)

    def __init__(self, sess_getter, parent=None):
        super().__init__(parent)
        self.sess_getter = sess_getter
        self.rec = None
        self.rows = None            # binary rows to display (sorted)
        self.ch_idx = None          # sorter channel indices in display order
        self.ch_y = None            # display offset per displayed channel
        self.shank = None
        self.sel_unit = None
        self._onset_labels = None
        self._build()
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._draw)

    # ------------------------------------------------------------------ UI
    def _build(self):
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        bar = QtWidgets.QHBoxLayout()
        self.btn_open = QtWidgets.QPushButton('Load recording')
        self.btn_open.setToolTip('Open the concatenated binary from Session settings / Kilosort params.py')
        self.btn_open.clicked.connect(self.load)
        self.btn_locate = QtWidgets.QPushButton('Locate binary…')
        self.btn_locate.clicked.connect(self.locate)
        bar.addWidget(self.btn_open); bar.addWidget(self.btn_locate)
        bar.addSpacing(10)
        bar.addWidget(QtWidgets.QLabel('t (s)'))
        self.t_spin = QtWidgets.QDoubleSpinBox(); self.t_spin.setDecimals(3); self.t_spin.setRange(0, 1e7)
        self.t_spin.setSingleStep(0.1); self.t_spin.valueChanged.connect(self._schedule)
        bar.addWidget(self.t_spin)
        bar.addWidget(QtWidgets.QLabel('window'))
        self.win = QtWidgets.QComboBox()
        for w in ['0.05', '0.1', '0.2', '0.5', '1', '2', '5', '10']:
            self.win.addItem(w + ' s', float(w))
        self.win.setCurrentIndex(3)
        self.win.currentIndexChanged.connect(self._schedule)
        bar.addWidget(self.win)
        bar.addWidget(QtWidgets.QLabel('gain'))
        self.gain = QtWidgets.QDoubleSpinBox(); self.gain.setRange(0.05, 100); self.gain.setValue(1.0)
        self.gain.setSingleStep(0.25); self.gain.setToolTip('Ctrl + mouse wheel also changes gain')
        self.gain.valueChanged.connect(self._schedule)
        bar.addWidget(self.gain)
        self.hp = QtWidgets.QCheckBox('300 Hz HP'); self.hp.setChecked(True); self.hp.toggled.connect(self._schedule)
        self.car = QtWidgets.QCheckBox('CAR'); self.car.setChecked(True); self.car.toggled.connect(self._schedule)
        self.spk = QtWidgets.QCheckBox('Units'); self.spk.setChecked(True); self.spk.toggled.connect(self._schedule)
        self.allu = QtWidgets.QCheckBox('incl. MUA'); self.allu.toggled.connect(self._schedule)
        self.stim = QtWidgets.QCheckBox('Stimuli'); self.stim.setChecked(True); self.stim.toggled.connect(self._schedule)
        for w in (self.hp, self.car, self.spk, self.allu, self.stim):
            bar.addWidget(w)
        bar.addSpacing(10)
        bar.addWidget(QtWidgets.QLabel('every'))
        self.decim = QtWidgets.QSpinBox(); self.decim.setRange(1, 16); self.decim.setValue(1)
        self.decim.setSuffix(' ch'); self.decim.setToolTip('Show every n-th channel')
        self.decim.valueChanged.connect(self._layout_channels)
        bar.addWidget(self.decim)
        self.shank_box = QtWidgets.QHBoxLayout()
        bar.addLayout(self.shank_box)
        bar.addStretch(1)
        lay.addLayout(bar)

        nav = QtWidgets.QHBoxLayout()
        nav.addWidget(QtWidgets.QLabel('Segment'))
        self.seg_combo = QtWidgets.QComboBox(); self.seg_combo.activated.connect(self._jump_seg)
        nav.addWidget(self.seg_combo)
        nav.addWidget(QtWidgets.QLabel('Stimulus'))
        self.cond_combo = QtWidgets.QComboBox(); self.cond_combo.setMinimumContentsLength(22)
        self.cond_combo.activated.connect(lambda: self._jump_trial(0))
        nav.addWidget(self.cond_combo)
        self.trial_spin = QtWidgets.QSpinBox(); self.trial_spin.setPrefix('trial '); self.trial_spin.setRange(1, 1)
        self.trial_spin.valueChanged.connect(lambda v: self._jump_trial(v - 1))
        nav.addWidget(self.trial_spin)
        self.pre_spin = QtWidgets.QDoubleSpinBox(); self.pre_spin.setPrefix('pre '); self.pre_spin.setSuffix(' s')
        self.pre_spin.setRange(0, 10); self.pre_spin.setValue(0.1); self.pre_spin.setSingleStep(0.05)
        nav.addWidget(self.pre_spin)
        b1 = QtWidgets.QPushButton('◀ page'); b1.clicked.connect(lambda: self._page(-1))
        b2 = QtWidgets.QPushButton('page ▶'); b2.clicked.connect(lambda: self._page(1))
        nav.addWidget(b1); nav.addWidget(b2)
        self.info = QtWidgets.QLabel('')
        self.info.setStyleSheet('color:#555')
        nav.addWidget(self.info, 1)
        lay.addLayout(nav)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left); ll.setContentsMargins(0, 0, 0, 0)
        pg.setConfigOptions(antialias=False)
        self.plot = TracePlot(background='w')
        self.plot.setMenuEnabled(False)
        self.plot.showGrid(x=True, y=False, alpha=0.15)
        self.plot.getPlotItem().getAxis('left').setWidth(60)
        self.plot.setLabel('bottom', 'time (s)')
        self.plot.wheel_time.connect(lambda d: self._page(d * 0.25))
        self.plot.wheel_gain.connect(lambda d: self.gain.setValue(self.gain.value() * (1.25 ** d)))
        self.plot.scene().sigMouseClicked.connect(self._plot_click)
        ll.addWidget(self.plot, 1)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.sliderReleased.connect(self._slider_moved)
        self.slider.valueChanged.connect(lambda v: self.slider.isSliderDown() or self._slider_moved())
        ll.addWidget(self.slider)
        split.addWidget(left)

        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right); rl.setContentsMargins(0, 0, 0, 0)
        self.unit_filter = QtWidgets.QLineEdit(); self.unit_filter.setPlaceholderText('filter units (id or depth range 200-800)')
        self.unit_filter.textChanged.connect(self._fill_units)
        rl.addWidget(self.unit_filter)
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(['unit', 'label', 'depth', 'shank', 'spikes', 'FR'])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.itemSelectionChanged.connect(self._unit_selected)
        self.table.cellDoubleClicked.connect(lambda r, c: self.open_unit.emit(int(self.table.item(r, 0).text())))
        rl.addWidget(self.table, 2)
        row = QtWidgets.QHBoxLayout()
        bp = QtWidgets.QPushButton('◀ spike'); bp.clicked.connect(lambda: self._step_spike(-1))
        bn = QtWidgets.QPushButton('spike ▶'); bn.clicked.connect(lambda: self._step_spike(1))
        bo = QtWidgets.QPushButton('Open in Single unit'); bo.clicked.connect(
            lambda: self.sel_unit is not None and self.open_unit.emit(int(self.sel_unit)))
        row.addWidget(bp); row.addWidget(bn); row.addWidget(bo)
        rl.addLayout(row)
        self.wf = pg.PlotWidget(background='w')
        self.wf.setMinimumHeight(180)
        self.wf.hideAxis('left')
        self.wf.setTitle('template')
        rl.addWidget(self.wf, 1)
        split.addWidget(right)
        split.setStretchFactor(0, 4); split.setStretchFactor(1, 1)
        split.setSizes([1100, 330])
        lay.addWidget(split, 1)
        self.shank_checks = []

    # ------------------------------------------------------------------ loading
    def locate(self):
        sess = self.sess_getter()
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Concatenated recording binary', sess.cfg.save_dir or '',
                                                     'Binary (*.bin *.dat);;All files (*)')
        if p:
            sess.cfg['recording_bin'] = p
            try:
                sess.cfg.save()
            except Exception:
                pass
            sess._rec = None
            self.load()

    def load(self):
        sess = self.sess_getter()
        try:
            self.rec = sess.recording(log)
            sp = sess.spikes
        except Exception as e:
            show_error(self, 'Recording', str(e))
            return
        chpos = sp['channel_pos']
        import os
        cm_path = os.path.join(sess.cfg.sorter_dir, 'channel_map.npy')
        chmap = np.load(cm_path).ravel() if os.path.exists(cm_path) else np.arange(len(chpos))
        self.chmap = chmap.astype(int)
        self.chpos = chpos
        self.shank_all = npx_io.shank_from_x(chpos[:, 0])
        # shank checkboxes
        for c in self.shank_checks:
            self.shank_box.removeWidget(c); c.deleteLater()
        self.shank_checks = []
        for s in sorted(set(self.shank_all)):
            cb = QtWidgets.QCheckBox('S{}'.format(s))
            cb.setChecked(True)
            cb.setStyleSheet('color: {}; font-weight: bold'.format(SHANK_COLORS[s % len(SHANK_COLORS)]))
            cb.toggled.connect(self._layout_channels)
            self.shank_box.addWidget(cb)
            self.shank_checks.append(cb)
        self.t_spin.setMaximum(self.rec.duration)
        self._fill_segments()
        self._fill_units()
        self._layout_channels()
        self.info.setText('{} | {} ch | {:.0f} Hz | {:.1f} s'.format(
            os.path.basename(self.rec.path), self.rec.n_channels, self.rec.fs, self.rec.duration))

    def _fill_segments(self):
        sess = self.sess_getter()
        self.seg_combo.clear(); self.cond_combo.clear()
        self._onset_labels = None
        try:
            b = sess.boundaries
            for s in sess.order:
                self.seg_combo.addItem('{}  ({:.0f}-{:.0f}s)'.format(s, b[s]['start_s'], b[s]['end_s']), s)
        except Exception:
            pass
        try:
            for seg in sess.order:
                for c in sess.conditions(seg):
                    if c != 'rfm_frames':
                        self.cond_combo.addItem('{}:{}'.format(seg, c))
            if sess.step_done('b1'):
                df = sess.df_b1
                for r in sorted(df['routine'].unique()):
                    self.cond_combo.addItem('B1:' + r)
        except Exception:
            pass

    def _fill_units(self):
        sess = self.sess_getter()
        if self.rec is None:
            return
        try:
            sp = sess.spikes
        except Exception:
            return
        lab = sp['labels_df'].set_index('cluster_id')['group'].to_dict()
        txt = self.unit_filter.text().strip()
        dmin, dmax, only = -1e9, 1e9, None
        if '-' in txt:
            try:
                dmin, dmax = [float(x) for x in txt.split('-', 1)]
            except ValueError:
                pass
        elif txt:
            only = txt.lstrip('u')
        tmax = sp['t_max'] or 1
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        for u, st in sp['trains_all'].items():
            d = sp['unit_depths'].get(u, np.nan)
            if only is not None and not str(u).startswith(only):
                continue
            if not (dmin <= (d if np.isfinite(d) else -1) <= dmax) and dmin > -1e9:
                continue
            bc = sp['unit_best_ch'].get(u, -1)
            r = self.table.rowCount()
            self.table.insertRow(r)
            vals = [u, lab.get(u, ''), d, int(self.shank_all[bc]) if bc >= 0 else -1, len(st), len(st) / tmax]
            for c, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem()
                if isinstance(v, (int, np.integer)):
                    it.setData(QtCore.Qt.DisplayRole, int(v))
                elif isinstance(v, (float, np.floating)):
                    it.setData(QtCore.Qt.DisplayRole, round(float(v), 1))
                else:
                    it.setText(str(v))
                if c == 0:
                    it.setForeground(UNIT_COLORS[int(u) % len(UNIT_COLORS)])
                self.table.setItem(r, c, it)
        self.table.setSortingEnabled(True)
        self.table.sortItems(2)
        self.table.resizeColumnsToContents()

    def _layout_channels(self):
        if self.rec is None:
            return
        on = {i for i, c in enumerate(self.shank_checks) if c.isChecked()}
        sh = self.shank_all
        idx = [i for i in range(len(self.chpos)) if sh[i] in on]
        idx.sort(key=lambda i: (-sh[i], self.chpos[i, 1], self.chpos[i, 0]))   # shank 0 on top
        step = self.decim.value()
        if step > 1:
            idx = idx[::step]
        self.ch_idx = np.array(idx, dtype=int)
        self.rows = self.chmap[self.ch_idx]
        # vertical layout: shank blocks side by side in y, gap between shanks
        y, ys, last = 0.0, [], None
        for i in self.ch_idx:
            if last is not None and sh[i] != last:
                y += 6
            ys.append(y); y += 1
            last = sh[i]
        self.ch_y = np.array(ys)
        self.shank = sh[self.ch_idx]
        ax = self.plot.getPlotItem().getAxis('left')
        ticks = []
        for k in range(0, len(self.ch_idx), max(1, len(self.ch_idx) // 24)):
            i = self.ch_idx[k]
            ticks.append((self.ch_y[k], '{:.0f}'.format(self.chpos[i, 1])))
        ax.setTicks([ticks, []])
        self.plot.setLabel('left', 'depth (µm) per shank')
        self.ch_pos_of = {int(c): k for k, c in enumerate(self.ch_idx)}
        self._schedule()

    # ------------------------------------------------------------------ navigation
    def _schedule(self):
        self._timer.start(30)

    def _page(self, frac):
        w = self.win.currentData()
        self.t_spin.setValue(max(0.0, self.t_spin.value() + frac * w))

    def _slider_moved(self):
        if self.rec is None:
            return
        self.t_spin.setValue(self.slider.value() / 1000.0 * max(0, self.rec.duration - self.win.currentData()))

    def _jump_seg(self):
        sess = self.sess_getter()
        s = self.seg_combo.currentData()
        try:
            self.t_spin.setValue(sess.boundaries[s]['start_s'])
        except Exception:
            pass

    def _cond_onsets(self):
        sess = self.sess_getter()
        txt = self.cond_combo.currentText()
        if not txt:
            return np.array([])
        seg, c = txt.split(':', 1)
        if seg == 'B1':
            df = sess.df_b1
            return np.sort(df.loc[df['routine'] == c, 'onset_s'].values)
        return np.sort(np.asarray(sess.cond_onsets[seg][c]))

    def _jump_trial(self, i):
        try:
            on = self._cond_onsets()
        except Exception:
            return
        if not len(on):
            return
        self.trial_spin.blockSignals(True)
        self.trial_spin.setRange(1, len(on))
        i = int(np.clip(i, 0, len(on) - 1))
        self.trial_spin.setValue(i + 1)
        self.trial_spin.blockSignals(False)
        self.t_spin.setValue(max(0.0, on[i] - self.pre_spin.value()))

    def _unit_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        self.sel_unit = int(self.table.item(rows[0].row(), 0).text())
        self._draw_template()
        self._schedule()

    def select_unit(self, cid):
        for r in range(self.table.rowCount()):
            if int(self.table.item(r, 0).text()) == cid:
                self.table.selectRow(r)
                return

    def _step_spike(self, d):
        if self.sel_unit is None:
            return
        st = self.sess_getter().spike_trains[self.sel_unit]
        w = self.win.currentData()
        c = self.t_spin.value() + w / 2
        if d > 0:
            i = np.searchsorted(st, c + 1e-4)
        else:
            i = np.searchsorted(st, c - 1e-4) - 1
        if 0 <= i < len(st):
            self.t_spin.setValue(max(0.0, st[i] - w / 2))

    def _draw_template(self):
        self.wf.clear()
        sess = self.sess_getter()
        sp = sess.spikes
        u = self.sel_unit
        tid = sp['template_of_unit'].get(u, -1)
        if tid < 0:
            return
        tm = np.asarray(sp['templates'][tid])
        bc = sp['unit_best_ch'][u]
        pos = self.chpos
        sh = self.shank_all
        same = np.nonzero(sh == sh[bc])[0]
        near = same[np.argsort(np.abs(pos[same, 1] - pos[bc, 1]) + np.abs(pos[same, 0] - pos[bc, 0]) * 0.01)][:12]
        amp = np.abs(tm[:, near]).max() or 1
        col = UNIT_COLORS[u % len(UNIT_COLORS)]
        xs = sorted(set(np.round(pos[near, 0], 1)))
        ys = sorted(set(np.round(pos[near, 1], 1)))
        T = tm.shape[0]
        for c in near:
            ix = xs.index(round(pos[c, 0], 1))
            iy = ys.index(round(pos[c, 1], 1))
            x = ix * (T * 1.2) + np.arange(T)
            self.wf.plot(x, iy * 1.0 + tm[:, c] / amp * 0.9, pen=pg.mkPen(col, width=2.5 if c == bc else 1.2))
        self.wf.hideAxis('bottom')
        self.wf.setTitle('u{}  best ch {}  depth {:.0f} µm'.format(u, bc, pos[bc, 1]))

    def _plot_click(self, ev):
        if ev.double():
            p = self.plot.getPlotItem().vb.mapSceneToView(ev.scenePos())
            log('t = {:.4f} s'.format(p.x()))

    # ------------------------------------------------------------------ drawing
    def _draw(self):
        if self.rec is None or self.rows is None or not len(self.rows):
            return
        sess = self.sess_getter()
        w = self.win.currentData()
        t0 = self.t_spin.value()
        t1 = min(self.rec.duration, t0 + w)
        self.slider.blockSignals(True)
        self.slider.setValue(int(1000 * t0 / max(1e-9, self.rec.duration - w)))
        self.slider.blockSignals(False)
        pad = 0.05 if self.hp.isChecked() else 0.0
        a0 = max(0.0, t0 - pad)
        d, i0 = self.rec.chunk(a0, t1 + pad, self.rows)
        if d.size == 0:
            return
        if self.rec.uv_per_bit:
            d = d * self.rec.uv_per_bit
        if self.hp.isChecked() and d.shape[0] > 30:
            sos = butter(3, 300.0, btype='high', fs=self.rec.fs, output='sos')
            d = sosfiltfilt(sos, d, axis=0)
        else:
            d = d - np.median(d, axis=0, keepdims=True)
        if self.car.isChecked():
            d = d - np.median(d, axis=1, keepdims=True)
        s0 = int(round((t0 - a0) * self.rec.fs))
        n = int(round((t1 - t0) * self.rec.fs))
        d = d[s0:s0 + n]
        tt = (i0 + s0 + np.arange(d.shape[0])) / self.rec.fs
        # decimate for drawing: keep min/max per pixel bin
        npx = 3000
        if d.shape[0] > npx * 2:
            k = d.shape[0] // npx
            m = d[:k * npx].reshape(npx, k, -1)
            d2 = np.empty((npx * 2, d.shape[1]), dtype=np.float32)
            d2[0::2] = m.min(axis=1); d2[1::2] = m.max(axis=1)
            t2 = np.repeat(tt[:k * npx].reshape(npx, k)[:, 0], 2)
            d, tt = d2, t2
        sc = np.percentile(np.abs(d), 99) if d.size else 1
        scale = self.gain.value() * 0.9 / (sc * 2 + 1e-9)
        self.plot.clear()
        # one curve per shank (NaN separated) - fast
        for s in np.unique(self.shank):
            m = self.shank == s
            Y = d[:, m] * scale + self.ch_y[m][None, :]
            X = np.repeat(tt[:, None], m.sum(), axis=1)
            X = np.vstack([X, np.full((1, m.sum()), np.nan)]).T.ravel()
            Y = np.vstack([Y, np.full((1, m.sum()), np.nan)]).T.ravel()
            self.plot.addItem(pg.PlotCurveItem(X, Y, pen=pg.mkPen(SHANK_COLORS[s % len(SHANK_COLORS)], width=1),
                                               connect='finite'))
        ymin, ymax = self.ch_y.min() - 2, self.ch_y.max() + 2
        # segments + stimuli
        try:
            b = sess.boundaries
            for seg, bb in b.items():
                if bb['end_s'] > t0 and bb['start_s'] < t1:
                    r = pg.LinearRegionItem([max(t0, bb['start_s']), min(t1, bb['end_s'])], movable=False,
                                            brush=pg.mkBrush(*SEG_BRUSH.get(seg[0], (128, 128, 128, 20))))
                    r.setZValue(-10)
                    for ln in r.lines:
                        ln.setPen(pg.mkPen(None))
                    self.plot.addItem(r)
                    tx = pg.TextItem(seg, color=(60, 60, 60), anchor=(0, 0))
                    tx.setPos(max(t0, bb['start_s']), ymax)
                    self.plot.addItem(tx)
        except Exception:
            pass
        if self.stim.isChecked():
            try:
                labs = self._labels(sess)
                lt, ll = labs
                i_a, i_b = np.searchsorted(lt, t0), np.searchsorted(lt, t1)
                for k in range(i_a, min(i_b, i_a + 200)):
                    ln = pg.InfiniteLine(lt[k], angle=90, pen=pg.mkPen((0, 0, 0, 160), width=1, style=QtCore.Qt.DashLine))
                    self.plot.addItem(ln)
                    if i_b - i_a < 40:
                        tx = pg.TextItem(ll[k], color=(0, 0, 0), anchor=(0, 1), angle=90)
                        tx.setPos(lt[k], ymin + 2)
                        self.plot.addItem(tx)
            except Exception:
                pass
        if self.spk.isChecked():
            self._draw_spikes(sess, t0, t1)
        self.plot.setXRange(t0, t1, padding=0)
        self.plot.setYRange(ymin, ymax + 2, padding=0)

    def _labels(self, sess):
        if self._onset_labels is None:
            T, L = [], []
            try:
                for seg in sess.order:
                    for c, on in sess.cond_onsets.get(seg, {}).items():
                        if c in ('rfm_frames', 'all_gratings', 'drift_pos', 'drift_neg') or '_azi' in c:
                            continue
                        for t in on:
                            T.append(float(t)); L.append(c)
                if sess.step_done('b1'):
                    df = sess.df_b1
                    for t, r, c in zip(df['onset_s'], df['routine'], df['condition'].fillna('')):
                        T.append(float(t)); L.append('B1 {} {}'.format(r, c).strip())
            except Exception:
                pass
            try:
                flat = sess.stim_onsets_flat
                known = np.array(sorted(T))
                for t in flat:
                    if not len(known) or np.min(np.abs(known - t)) > 0.02:
                        T.append(float(t)); L.append('TTL')
            except Exception:
                pass
            o = np.argsort(T)
            self._onset_labels = (np.array(T)[o], [L[i] for i in o])
        return self._onset_labels

    def _draw_spikes(self, sess, t0, t1):
        sp = sess.spikes
        lab = sp['labels_df'].set_index('cluster_id')['group'].to_dict()
        xs, ys, brushes, sizes = [], [], [], []
        for u, st in sp['trains_all'].items():
            g = lab.get(u, '')
            if g == 'noise' or (g != 'good' and not self.allu.isChecked() and u != self.sel_unit):
                continue
            bc = sp['unit_best_ch'].get(u, -1)
            if bc < 0 or bc not in self.ch_pos_of:
                continue
            a, b = np.searchsorted(st, t0), np.searchsorted(st, t1)
            if b <= a:
                continue
            n = b - a
            xs.append(st[a:b]); ys.append(np.full(n, self.ch_y[self.ch_pos_of[bc]] + 0.5))
            col = UNIT_COLORS[u % len(UNIT_COLORS)]
            sel = (u == self.sel_unit)
            brushes += [pg.mkBrush(col)] * n
            sizes += [11 if sel else 6] * n
        if xs:
            self.plot.addItem(pg.ScatterPlotItem(np.concatenate(xs), np.concatenate(ys), size=sizes, brush=brushes,
                                                 pen=pg.mkPen(None), symbol='t'))

    def session_changed(self):
        self.rec = None
        self.plot.clear()
        self.table.setRowCount(0)
        self.info.setText('Press "Load recording" to open the concatenated binary.')

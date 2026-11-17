"""Recordings tab: list of completed recordings + pooled classification."""
import os

from PySide6 import QtCore, QtWidgets

from ..core import pooling
from ..plots import pool_plots
from .common import Param as P, log, show_error
from .tabs import SubTabs

COLS = ['', 'recording', 'mouse line', 'units', 'QI-pass', 'RF-only', 'export', 'status']


class PooledState(object):
    def __init__(self):
        self.rl = pooling.RecordingList()
        self.P = None
        self.R = None


class PooledTab(QtWidgets.QWidget):
    open_session = QtCore.Signal(str)

    def __init__(self, runner, settings, parent=None):
        super().__init__(parent)
        self.runner = runner
        self.qs = settings
        self.state = PooledState()
        sg = lambda: self.state
        lay = QtWidgets.QHBoxLayout(self)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left); ll.setContentsMargins(0, 0, 0, 0)
        fl = QtWidgets.QHBoxLayout()
        self.list_label = QtWidgets.QLabel('No recording list')
        self.list_label.setWordWrap(True)
        self.list_label.setStyleSheet('font-weight:bold')
        fl.addWidget(self.list_label, 1)
        for txt, fn in [('New…', self.new_list), ('Open…', self.open_list)]:
            b = QtWidgets.QPushButton(txt); b.clicked.connect(fn); fl.addWidget(b)
        ll.addLayout(fl)
        self.table = QtWidgets.QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.itemChanged.connect(self._toggled)
        self.table.cellDoubleClicked.connect(lambda r, c: self.open_session.emit(self.state.rl.items[r]['folder']))
        self.table.setToolTip('Tick = include in the pooled classification. Double-click opens the recording.')
        ll.addWidget(self.table, 1)
        g = QtWidgets.QGridLayout()
        btns = [('Add recording…', self.add_one), ('Scan folder…', self.scan_folder),
                ('Add current session', self.add_current), ('Remove selected', self.remove_sel),
                ('Export missing', lambda: self.export(False)), ('Re-export selected', self.reexport_sel)]
        for i, (txt, fn) in enumerate(btns):
            b = QtWidgets.QPushButton(txt); b.clicked.connect(fn); g.addWidget(b, i // 2, i % 2)
        ll.addLayout(g)
        box = QtWidgets.QGroupBox('Pooled classification (SparsePCA + GMM)')
        f = QtWidgets.QFormLayout(box)
        self.ncomp = QtWidgets.QSpinBox(); self.ncomp.setRange(2, 40); self.ncomp.setValue(8)
        self.kmin = QtWidgets.QSpinBox(); self.kmin.setRange(1, 100); self.kmin.setValue(2)
        self.kmax = QtWidgets.QSpinBox(); self.kmax.setRange(2, 200); self.kmax.setValue(30)
        self.fixk = QtWidgets.QSpinBox(); self.fixk.setRange(0, 200); self.fixk.setSpecialValueText('BIC')
        self.seed = QtWidgets.QSpinBox(); self.seed.setRange(0, 10 ** 6)
        f.addRow('SparsePCA components', self.ncomp)
        f.addRow('k range (BIC)', self._row(self.kmin, self.kmax))
        f.addRow('Fixed k (0 = BIC)', self.fixk)
        f.addRow('Random seed', self.seed)
        rb = QtWidgets.QPushButton('Run pooled classification')
        rb.setStyleSheet('font-weight:bold; padding:6px')
        rb.clicked.connect(self.run_pooled)
        f.addRow(rb)
        ll.addWidget(box)
        self.summary = QtWidgets.QLabel('')
        self.summary.setWordWrap(True)
        ll.addWidget(self.summary)
        split.addWidget(left)

        self.tabs = SubTabs(sg)
        self.tabs.add('Cluster traces', pool_plots.traces, [P('clusters', 'Clusters (blank = all)', 'str', ''),
                                                            P('qi_only', 'QI-pass units only', 'bool', False)],
                      'Mean ± SEM per cluster for each B1 feature segment, all recordings pooled.')
        self.tabs.add('Cluster detail', pool_plots.cluster_detail,
                      [P('cluster', 'Cluster', 'unit', 0, choices=pool_plots.cluster_choices)],
                      'Every unit of one cluster, direction / speed / colour tuning, depth, RF size and '
                      'which recordings it comes from (← → step clusters).', auto=True)
        self.tabs.add('Composition', pool_plots.composition, [P('normalise', 'Normalise per mouse line', 'bool', True)],
                      'Cluster membership by recording and by mouse line.')
        self.tabs.add('Profile', pool_plots.profile_table, (), 'Depth, RF size, DS/OS, colour and speed per cluster.')
        self.tabs.add('Feature matrix', pool_plots.feature_matrix, (), '')
        self.tabs.add('BIC', pool_plots.bic, (), '')
        split.addWidget(self.tabs)
        split.setSizes([520, 1000])
        lay.addWidget(split)
        last = self.qs.value('recording_list', '')
        if last and os.path.exists(last):
            self.load_list(last)

    @staticmethod
    def _row(*ws):
        w = QtWidgets.QWidget(); h = QtWidgets.QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0)
        for x in ws:
            h.addWidget(x)
        return w

    # ------------------------------------------------------------------ list
    def new_list(self):
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'New recording list', 'sc_recordings.json', 'JSON (*.json)')
        if not p:
            return
        self.state.rl = pooling.RecordingList()
        self.state.rl.save(p)
        self._after_list(p)

    def open_list(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open recording list', '', 'JSON (*.json)')
        if p:
            self.load_list(p)

    def load_list(self, p):
        try:
            self.state.rl = pooling.RecordingList(p)
        except Exception as e:
            show_error(self, 'Recording list', str(e)); return
        self._after_list(p)

    def _after_list(self, p):
        self.qs.setValue('recording_list', p)
        self.state.P = self.state.R = None
        self.list_label.setText(os.path.basename(p) + '\n' + os.path.dirname(p))
        self.fill()
        self._load_saved()

    def _save(self):
        if not self.state.rl.path:
            p, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save recording list', 'sc_recordings.json',
                                                         'JSON (*.json)')
            if not p:
                return False
            self.state.rl.save(p)
            self._after_list(p)
        else:
            self.state.rl.save()
        return True

    def fill(self):
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        names = {}
        for it in self.state.rl.items:
            st = pooling.status(it['folder'])
            dup = names.get(st['name'])
            names.setdefault(st['name'], it['folder'])
            r = self.table.rowCount(); self.table.insertRow(r)
            chk = QtWidgets.QTableWidgetItem()
            chk.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            chk.setCheckState(QtCore.Qt.Checked if it.get('include', True) else QtCore.Qt.Unchecked)
            self.table.setItem(r, 0, chk)
            if not os.path.isdir(it['folder']):
                state = 'folder missing'
            elif dup:
                state = 'duplicate - skipped when pooling'
            elif st['exported']:
                state = 'ready' if st['has_colour_speed'] else 'ready (old export: no colour/speed)'
            elif st['pipeline_ready'] and st['has_session']:
                state = 'needs export'
            elif st['has_session']:
                state = 'pipeline incomplete'
            else:
                state = 'no session file'
            vals = [st['name'], st['mouse_line'], st['n_units'] or '', st['n_qi'] or '', st['n_rf'] or '',
                    st['export_time'], state]
            for c, v in enumerate(vals, start=1):
                x = QtWidgets.QTableWidgetItem(str(v))
                x.setToolTip(it['folder'])
                if c == 7:
                    x.setForeground(QtCore.Qt.darkGreen if state.startswith('ready') else QtCore.Qt.darkRed)
                self.table.setItem(r, c, x)
        self.table.resizeColumnsToContents()
        self.table.blockSignals(False)
        inc = self.state.rl.included()
        self.summary.setText('{} recordings listed, {} included'.format(len(self.state.rl.items), len(inc)))

    def _toggled(self, item):
        if item.column() != 0:
            return
        self.state.rl.items[item.row()]['include'] = item.checkState() == QtCore.Qt.Checked
        if self.state.rl.path:
            self.state.rl.save()
        self.summary.setText('{} recordings listed, {} included'.format(
            len(self.state.rl.items), len(self.state.rl.included())))

    def _selected_folders(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [self.state.rl.items[r]['folder'] for r in rows]

    def add_one(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, 'Recording analysis folder (…_concat)')
        if d:
            self.state.rl.add(d)
            if self._save():
                self.fill()

    def add_current(self):
        w = self.window()
        d = w.sess.cfg.save_dir if hasattr(w, 'sess') else ''
        if not d:
            QtWidgets.QMessageBox.information(self, 'Recordings', 'No session open'); return
        self.state.rl.add(d)
        if self._save():
            self.fill()

    def scan_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, 'Folder containing recording analysis folders')
        if not d:
            return
        found = pooling.scan(d)
        n = sum(self.state.rl.add(f) for f in found)
        log('Scan {}: {} analysis folders, {} new'.format(d, len(found), n))
        if self._save():
            self.fill()

    def remove_sel(self):
        self.state.rl.remove(self._selected_folders())
        if self._save():
            self.fill()

    def export(self, overwrite, folders=None):
        if folders is None:
            folders = [f for f in self.state.rl.included() if not pooling.status(f)['exported']]
        if not folders:
            QtWidgets.QMessageBox.information(self, 'Export', 'Every included recording is already exported.')
            return
        self.runner.run('Export {} recordings'.format(len(folders)), pooling.export_missing, folders,
                        overwrite=overwrite, cancel=True, on_done=lambda _: self.fill())

    def reexport_sel(self):
        f = self._selected_folders()
        if f:
            self.export(True, f)

    # ------------------------------------------------------------------ pooled
    def run_pooled(self):
        folders = self.state.rl.included()
        if not folders:
            QtWidgets.QMessageBox.information(self, 'Pooled classification', 'Add and tick recordings first.')
            return
        kw = dict(n_components=self.ncomp.value(), k_min=self.kmin.value(), k_max=self.kmax.value(),
                  fixed_k=self.fixk.value(), seed=self.seed.value())
        rl = self.state.rl

        def job(log=print, progress=None):
            Pd = pooling.load_pooled(folders, log)
            R = pooling.run_pooled(Pd, log=log, progress=progress, **kw)
            if rl.path:
                rl.settings = kw
                rl.save()
                pooling.save_pooled(rl, Pd, R, log)
            return Pd, R

        def done(out):
            self.state.P, self.state.R = out
            df = self.state.R['df']
            warn = ('\n⚠ BIC minimum at k max - raise k max or fix k' if self.state.R.get('at_k_max') else '')
            self.summary.setText('{} units from {} recordings → {} clusters{}\nresults: {}'.format(
                len(df), df['recording'].nunique(), self.state.R['best_k'], warn, rl.out_dir))
            self.tabs.refresh()
            self.tabs.setCurrentIndex(0)
            self.tabs.panels[0].redraw()
        self.runner.run('Pooled classification', job, on_done=done)

    def _load_saved(self):
        """Re-show the last pooled result of this list, if its recordings are unchanged."""
        rl = self.state.rl
        s = rl.settings or {}
        for w, k in [(self.ncomp, 'n_components'), (self.kmin, 'k_min'), (self.kmax, 'k_max'),
                     (self.fixk, 'fixed_k'), (self.seed, 'seed')]:
            if k in s:
                w.setValue(int(s[k]))
        self.tabs.refresh()

    def refresh(self):
        self.tabs.refresh()

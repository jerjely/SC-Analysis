"""Main window."""
import os
import sys

from PySide6 import QtCore, QtGui, QtWidgets

from .. import __version__
from ..core.config import SessionConfig, SESSION_FILENAME
from ..core.session import Session
from .common import LOG, TaskRunner, log, show_error
from .picker_widgets import BlinkLoomPicker, SweepPicker
from .pipeline_panel import PipelinePanel
from .settings import SettingsDialog, new_session_dialog
from .tabs import (OverviewTab, RFTab, UnitTab, PopulationTab, ClusteringTab, SweepTab, LoomTab, CCFTab,
                   PupilTab)
from .trace_view import RecordingTab
from .pool_tab import PooledTab


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, session_path=None):
        super().__init__()
        self.setWindowTitle('SC Analysis')
        self.resize(1500, 950)
        self.settings = QtCore.QSettings('VolgyiLab', 'SCAnalysis')
        self.sess = Session(SessionConfig())
        sg = self.get_sess

        sb = self.statusBar()
        self.pbar = QtWidgets.QProgressBar(); self.pbar.setMaximumWidth(260); self.pbar.hide()
        sb.addPermanentWidget(self.pbar)
        self.runner = TaskRunner(self, self.pbar, sb)

        self.tabs = QtWidgets.QTabWidget()
        self.rec_tab = RecordingTab(sg)
        self.overview = OverviewTab(sg)
        self.rf_tab = RFTab(sg)
        self.unit_tab = UnitTab(sg)
        self.pop_tab = PopulationTab(sg)
        self.clu_tab = ClusteringTab(sg, self.runner)
        self.pick_tabs = QtWidgets.QTabWidget()
        self.bl_picker = BlinkLoomPicker(sg)
        self.sw_picker = SweepPicker(sg)
        self.pick_tabs.addTab(self.bl_picker, 'Blink / loom picker')
        self.pick_tabs.addTab(self.sw_picker, 'Sweep picker')
        self.sweep_tab = SweepTab(sg, self.runner)
        self.loom_tab = LoomTab(sg)
        self.ccf_tab = CCFTab(sg, self.runner)
        self.pupil_tab = PupilTab(sg, self.runner)
        self.pool_tab = PooledTab(self.runner, self.settings)
        self.pool_tab.open_session.connect(self.open_session)
        for w, name in [(self.rec_tab, 'Recording'), (self.overview, 'Overview'), (self.rf_tab, 'RF maps'),
                        (self.unit_tab, 'Single unit'), (self.pop_tab, 'Population'),
                        (self.clu_tab, 'Baden clustering'), (self.pick_tabs, 'Pickers'), (self.sweep_tab, 'Sweep'),
                        (self.loom_tab, 'Loom'), (self.ccf_tab, 'CCF'), (self.pupil_tab, 'Pupil'),
                        (self.pool_tab, 'Recordings (pooled)')]:
            self.tabs.addTab(w, name)
        self.tabs.currentChanged.connect(self._tab_changed)
        self.setCentralWidget(self.tabs)
        self.rec_tab.open_unit.connect(self._open_unit)

        self.pipe = PipelinePanel(sg, self.runner)
        self.pipe.step_done.connect(lambda k: self.refresh_all())
        d = QtWidgets.QDockWidget('Pipeline', self)
        d.setObjectName('pipeline')
        box = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(box); bl.setContentsMargins(0, 0, 0, 0)
        self.sess_label = QtWidgets.QLabel('No session')
        self.sess_label.setWordWrap(True)
        self.sess_label.setStyleSheet('font-weight:bold; padding:4px')
        bl.addWidget(self.sess_label)
        bset = QtWidgets.QPushButton('Session settings…')
        bset.clicked.connect(self.edit_settings)
        bl.addWidget(bset)
        bl.addWidget(self.pipe, 1)
        d.setWidget(box)
        d.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, d)

        self.logw = QtWidgets.QPlainTextEdit()
        self.logw.setReadOnly(True)
        self.logw.setMaximumBlockCount(20000)
        self.logw.setStyleSheet('font-family: Consolas, Menlo, monospace; font-size: 9pt;')
        LOG.message.connect(self._append_log)
        dl = QtWidgets.QDockWidget('Log', self)
        dl.setObjectName('log')
        dl.setWidget(self.logw)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dl)
        self.resizeDocks([dl], [170], QtCore.Qt.Vertical)
        self.resizeDocks([d], [290], QtCore.Qt.Horizontal)

        self._menus()
        geo = self.settings.value('geometry')
        if geo:
            self.restoreGeometry(geo)
        last = session_path or self.settings.value('last_session', '')
        if last and os.path.exists(last):
            QtCore.QTimer.singleShot(50, lambda: self.open_session(last))
        else:
            self._update_label()

    # ------------------------------------------------------------------
    def get_sess(self):
        return self.sess

    def _menus(self):
        m = self.menuBar().addMenu('&File')
        a = m.addAction('New session from Kilosort folder…'); a.setShortcut('Ctrl+N'); a.triggered.connect(self.new_session)
        a = m.addAction('Open session…'); a.setShortcut('Ctrl+O'); a.triggered.connect(self.open_dialog)
        self.recent_menu = m.addMenu('Recent sessions')
        self._fill_recent()
        a = m.addAction('Save session'); a.setShortcut('Ctrl+S'); a.triggered.connect(self.save_session)
        m.addSeparator()
        a = m.addAction('Open analysis folder'); a.triggered.connect(self.open_folder)
        m.addSeparator()
        a = m.addAction('Quit'); a.setShortcut('Ctrl+Q'); a.triggered.connect(self.close)
        m = self.menuBar().addMenu('&Session')
        a = m.addAction('Settings…'); a.setShortcut('Ctrl+,'); a.triggered.connect(self.edit_settings)
        a = m.addAction('Reload data from disk'); a.setShortcut('F5'); a.triggered.connect(self.reload)
        m = self.menuBar().addMenu('&Help')
        a = m.addAction('About'); a.triggered.connect(self.about)

    def _fill_recent(self):
        self.recent_menu.clear()
        for p in (self.settings.value('recent', []) or [])[:10]:
            a = self.recent_menu.addAction(p)
            a.triggered.connect(lambda _=False, p=p: self.open_session(p))

    def _remember(self, path):
        rec = [p for p in (self.settings.value('recent', []) or []) if p != path]
        rec.insert(0, path)
        self.settings.setValue('recent', rec[:10])
        self.settings.setValue('last_session', path)
        self._fill_recent()

    def _append_log(self, msg):
        self.logw.appendPlainText(msg)
        sbar = self.logw.verticalScrollBar()
        sbar.setValue(sbar.maximum())

    def _update_label(self):
        c = self.sess.cfg
        if c.save_dir:
            self.sess_label.setText('{}\n{}'.format(c.session_id, c.save_dir))
            self.setWindowTitle('SC Analysis — {}'.format(c.session_id))
        else:
            self.sess_label.setText('No session — File > New session from Kilosort folder, or File > Open session')

    # ------------------------------------------------------------------
    def new_session(self):
        cfg = new_session_dialog(self)
        if cfg is None:
            return
        self._set_cfg(cfg)
        if cfg.path() and os.path.exists(cfg.path()):
            self._remember(cfg.path())

    def open_dialog(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open session', '', 'Session (sc_session.json);;JSON (*.json)')
        if p:
            self.open_session(p)

    def open_session(self, path):
        if os.path.isdir(path):
            path = os.path.join(path, SESSION_FILENAME)
        try:
            cfg = SessionConfig.load(path)
        except Exception as e:
            show_error(self, 'Open session', str(e)); return
        if not cfg.save_dir:
            cfg['save_dir'] = os.path.dirname(path)
        self._set_cfg(cfg)
        self._remember(path)
        log('Opened session {}'.format(path))

    def _set_cfg(self, cfg):
        self.sess = Session(cfg)
        self.rec_tab.session_changed()
        self._update_label()
        self.refresh_all()

    def save_session(self):
        try:
            p = self.sess.cfg.save()
            self._remember(p)
            log('Saved {}'.format(p))
        except Exception as e:
            show_error(self, 'Save session', str(e))

    def edit_settings(self):
        dlg = SettingsDialog(self.sess.cfg, self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            res = self.sess.results
            self.sess.reset()
            self.sess.results = res
            if self.sess.cfg.save_dir:
                self.save_session()
            self._update_label()
            self.refresh_all()

    def reload(self):
        self.sess.reset()
        self.rec_tab.session_changed()
        self.refresh_all()
        log('Reloaded from disk')

    def open_folder(self):
        d = self.sess.cfg.save_dir
        if d and os.path.isdir(d):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(d))

    def refresh_all(self):
        self.pipe.refresh()
        for t in (self.overview, self.rf_tab, self.unit_tab, self.pop_tab, self.clu_tab, self.sweep_tab, self.loom_tab,
                  self.ccf_tab, self.pupil_tab):
            try:
                t.refresh()
            except Exception as e:
                log('refresh: {}'.format(e))

    def _tab_changed(self, i):
        w = self.tabs.widget(i)
        if hasattr(w, 'refresh'):
            try:
                w.refresh()
            except Exception:
                pass
        if w is self.pick_tabs:
            for p in (self.bl_picker, self.sw_picker):
                if not p.cells:
                    pass

    def _open_unit(self, cid):
        self.tabs.setCurrentWidget(self.unit_tab)
        self.unit_tab.show_unit(cid)

    def about(self):
        QtWidgets.QMessageBox.about(
            self, 'About SC Analysis',
            '<b>SC Analysis {}</b><br><br>GUI for the SC_analysis_v24 notebook pipeline: '
            'Neuropixels recording viewer, stimulus sync, RF mapping, Baden clustering, '
            'peak pickers, sweep / loom / CCF / pupil analyses and pooled multi-recording '
            'classification.<br><br>Gergely Szarka · Völgyi lab, University of Pécs<br>'
            '<a href="https://github.com/jerjely/SC-Analysis">github.com/jerjely/SC-Analysis</a>'
            '<br>MIT licence'.format(__version__))

    def closeEvent(self, ev):
        self.settings.setValue('geometry', self.saveGeometry())
        super().closeEvent(ev)


ICON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resources')


def app_icon():
    ico = QtGui.QIcon()
    for f in ('icon.png', 'icon_256.png', 'icon_small.png'):
        p = os.path.join(ICON_DIR, f)
        if os.path.exists(p):
            ico.addFile(p)
    return ico


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    for stream in ('stdout', 'stderr'):          # windowed (no console) builds
        if getattr(sys, stream) is None:
            setattr(sys, stream, open(os.devnull, 'w'))
    if '--version' in argv:
        print(__version__)
        return 0
    if '--selftest' in argv:
        from ..selftest import run
        i = argv.index('--selftest')
        rep = argv[i + 1] if len(argv) > i + 1 and not argv[i + 1].startswith('--') else None
        keep = argv[argv.index('--keep') + 1] if '--keep' in argv else None
        return run(rep, keep)
    if sys.platform == 'win32':
        try:  # own taskbar entry + icon instead of the generic pythonw one
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('VolgyiLab.SCAnalysis')
        except Exception:
            pass
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    app.setApplicationName('SC Analysis')
    app.setWindowIcon(app_icon())
    app.setStyle('Fusion')
    args = [a for a in argv[1:] if not a.startswith('-psn_')]   # macOS Finder adds -psn_…
    path = args[0] if args else None
    w = MainWindow(path)
    w.show()
    return app.exec()

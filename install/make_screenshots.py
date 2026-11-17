"""Regenerate docs/screenshots/*.jpg from the synthetic demo recordings.

    python -m sc_analysis --selftest st.txt --keep /tmp/SC-demo
    python install/make_screenshots.py docs/screenshots /tmp/SC-demo
"""
import os, sys, time
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from PySide6 import QtWidgets, QtCore
QtCore.QCoreApplication.setOrganizationName('SCAnalysisShots')
app = QtWidgets.QApplication(['SC Analysis'])
app.setStyle('Fusion')
from sc_analysis.gui.main_window import MainWindow
from sc_analysis.core import baden, pupil, pooling
OUT = sys.argv[1]; D = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else '/tmp/SC-demo')
os.makedirs(OUT, exist_ok=True)
w = MainWindow(D + '/rec1/analysis/sc_session.json'); w.resize(1400, 880); w.show()
def pump(s=0.3):
    t = time.time()
    while time.time() - t < s: app.processEvents(); time.sleep(0.01)
pump(0.6); s = w.sess; q = lambda *a, **k: None
for f in (baden.run_qi, baden.run_features, baden.run_dsos, baden.run_colour, baden.run_clustering): f(s, log=q)
pupil.load_pupil(s, 'G2'); w.refresh_all(); pump(0.3)
def shot(name):
    pump(0.6); w.grab().scaled(1400, 880, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation).save(
        os.path.join(OUT, name + '.jpg'), 'JPG', 88); print('saved', name)
def panel(top, sub, i, unit=None):
    w.tabs.setCurrentWidget(top); sub.setCurrentIndex(i); pump(0.2)
    p = sub.panels[i]; p.refresh_choices()
    if unit is not None and 'unit' in p.form.widgets: p.form.set_value('unit', int(unit))
    p.redraw(); pump(0.5)
good = [int(u) for u in s.good_ids]
log_docks = [d for d in w.findChildren(QtWidgets.QDockWidget) if 'log' in d.windowTitle().lower()]
for d in log_docks: d.hide()
# 1 trace viewer
w.tabs.setCurrentWidget(w.rec_tab); w.rec_tab.load(); pump(0.5); w.rec_tab.t_spin.setValue(4.0); pump(0.3)
w.rec_tab.select_unit(good[2]); shot('01-recording-viewer')
panel(w.rf_tab, w.rf_tab, 3); shot('02-rf-maps')
panel(w.unit_tab, w.unit_tab, 2, good[3]); shot('03-baden-browser')
panel(w.pop_tab, w.pop_tab, 0); shot('04-population-heatmaps')
w.tabs.setCurrentWidget(w.pick_tabs); w.pick_tabs.setCurrentWidget(w.sw_picker); pump(0.3)
w.sw_picker.start(); pump(0.8); shot('05-sweep-picker')
for d in log_docks: d.show()
pt = w.pool_tab; pt.state.rl = pooling.RecordingList(D + '/sc_recordings.json'); pt._after_list(D + '/sc_recordings.json')
w.tabs.setCurrentWidget(pt); pt.kmax.setValue(8); pt.run_pooled(); pump(0.5)
while w.runner.current is not None: pump(0.2)
pt.tabs.setCurrentIndex(0); pt.tabs.panels[0].refresh_choices(); pt.tabs.panels[0].redraw(); shot('06-pooled-classification')

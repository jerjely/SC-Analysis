"""Pipeline dock: run-once setup steps with done/missing status."""
from PySide6 import QtCore, QtGui, QtWidgets

from ..core import pipeline
from ..core.session import STEPS

HELP = {
    'boundaries': 'Read fileTimeSecs from each segment .ap.meta -> segment_boundaries.json',
    'ttl': 'Extract TTL / photodiode edges from every OBX file (per-segment overrides in settings)',
    'b1': 'Pair B1 photodiode edges with the Baden protocol timeline',
    'stimuli': 'Parse PsychoPy CSVs into condition onsets (P1, gratings) + RFM frame times',
    'pd': 'Photodiode-corrected onsets for every P1 TTL (peak-to-trough method)',
    'sta': 'Spike-triggered averages for each usable RFM segment (+ combined)',
    'rffit': '2-D Gaussian fits of every STA',
    'rfmode': 'Apply RF mode / SNR / diameter / shift criteria -> rf_results_clean.npy',
}


class PipelinePanel(QtWidgets.QWidget):
    step_done = QtCore.Signal(str)

    def __init__(self, sess_getter, runner, parent=None):
        super().__init__(parent)
        self.sess_getter = sess_getter
        self.runner = runner
        self.rows = {}
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        grid = QtWidgets.QGridLayout()
        for i, (key, label, _) in enumerate(STEPS):
            st = QtWidgets.QLabel('○')
            st.setFixedWidth(18)
            lb = QtWidgets.QLabel(label)
            lb.setToolTip(HELP.get(key, ''))
            b = QtWidgets.QPushButton('Run')
            b.setFixedWidth(52)
            b.setToolTip(HELP.get(key, ''))
            b.clicked.connect(lambda _=False, k=key: self.run(k))
            grid.addWidget(st, i, 0); grid.addWidget(lb, i, 1); grid.addWidget(b, i, 2)
            self.rows[key] = (st, lb, b)
        lay.addLayout(grid)
        self.overwrite = QtWidgets.QCheckBox('Overwrite existing STA / RF files')
        lay.addWidget(self.overwrite)
        h = QtWidgets.QHBoxLayout()
        self.all_btn = QtWidgets.QPushButton('Run all missing')
        self.all_btn.clicked.connect(self.run_all)
        self.cancel_btn = QtWidgets.QPushButton('Cancel')
        self.cancel_btn.clicked.connect(self._cancel)
        h.addWidget(self.all_btn); h.addWidget(self.cancel_btn)
        lay.addLayout(h)
        lay.addStretch(1)
        self._queue = []

    def refresh(self):
        sess = self.sess_getter()
        for key, (st, lb, b) in self.rows.items():
            done = sess.step_done(key)
            if key == 'b1' and sess.cfg.b1_segment not in (sess.order if sess.step_done('boundaries') else []):
                st.setText('–'); st.setStyleSheet('color:#999'); st.setToolTip('no B1 segment')
                continue
            st.setText('✔' if done else '○')
            st.setStyleSheet('color: #2e7d32; font-weight: bold' if done else 'color: #999')
            st.setToolTip('done' if done else 'not run yet')

    def run(self, key, then=None):
        sess = self.sess_getter()
        label = [l for k, l, _ in STEPS if k == key][0]

        def done(_):
            self.refresh()
            self.step_done.emit(key)
            if then:
                then()
        self.runner.run(label.split('  ', 1)[-1], pipeline.run_step, sess, key,
                        overwrite=self.overwrite.isChecked(), on_done=done,
                        on_fail=lambda m: (self.refresh(), self._queue.clear(), self._fail(label, m)),
                        cancel=key in ('sta', 'rffit'))

    def _fail(self, label, msg):
        QtWidgets.QMessageBox.warning(self, label, msg)

    def run_all(self):
        sess = self.sess_getter()
        self._queue = []
        for key, _, _ in STEPS:
            if key == 'b1':
                try:
                    if sess.cfg.b1_segment not in sess.order:
                        continue
                except Exception:
                    pass
            if not sess.step_done(key) or (key in ('sta', 'rffit') and self.overwrite.isChecked()):
                self._queue.append(key)
        self._next()

    def _next(self):
        if not self._queue:
            return
        k = self._queue.pop(0)
        self.run(k, then=self._next)

    def _cancel(self):
        self._queue = []
        self.runner.cancel()

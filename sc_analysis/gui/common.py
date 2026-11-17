"""
Shared GUI plumbing: background workers, log bus, auto-built parameter
forms and the matplotlib figure panel used by every analysis tab.
"""
import os
import sys
import traceback
import warnings

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from ..core.session import MissingStep  # noqa: E402


# ---------------------------------------------------------------------------
# log bus
# ---------------------------------------------------------------------------
class LogBus(QtCore.QObject):
    message = QtCore.Signal(str)

    def log(self, *args):
        self.message.emit(' '.join(str(a) for a in args))


LOG = LogBus()


def log(*args):
    LOG.log(*args)


# ---------------------------------------------------------------------------
# background worker
# ---------------------------------------------------------------------------
class _WorkerSignals(QtCore.QObject):
    log = QtCore.Signal(str)
    progress = QtCore.Signal(int, int)
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str, str)


class Worker(QtCore.QRunnable):
    """Run fn(log=..., progress=..., cancel=...) in the thread pool."""

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.sig = _WorkerSignals()
        self._cancel = False
        self.setAutoDelete(False)

    def cancel(self):
        self._cancel = True

    def run(self):
        kw = dict(self.kwargs)
        kw.setdefault('log', lambda *a: self.sig.log.emit(' '.join(str(x) for x in a)))
        if kw.pop('_with_progress', True):
            kw.setdefault('progress', lambda i, n: self.sig.progress.emit(int(i), int(n)))
        if kw.pop('_with_cancel', False):
            kw.setdefault('cancel', lambda: self._cancel)
        try:
            out = self.fn(*self.args, **kw)
            self.sig.done.emit(out)
        except Exception as e:  # noqa
            self.sig.failed.emit(str(e), traceback.format_exc())


class TaskRunner(QtCore.QObject):
    """Owns the thread pool, the status bar progress, and one-at-a-time jobs."""
    busy_changed = QtCore.Signal(bool)

    def __init__(self, parent, progress_bar=None, status=None):
        super().__init__(parent)
        self.pool = QtCore.QThreadPool.globalInstance()
        self.pbar = progress_bar
        self.status = status
        self.current = None
        self._keep = []

    def run(self, label, fn, *args, on_done=None, on_fail=None, progress=True, cancel=False, **kwargs):
        if self.current is not None:
            QtWidgets.QMessageBox.information(self.parent(), 'Busy',
                                              'Another job is running ({}). Wait for it to finish.'.format(
                                                  self.current[0]))
            return None
        kwargs['_with_progress'] = progress
        kwargs['_with_cancel'] = cancel
        w = Worker(fn, *args, **kwargs)
        w.sig.log.connect(LOG.log)
        w.sig.progress.connect(self._progress)
        w.sig.done.connect(lambda out: self._finish(w, label, out, on_done))
        w.sig.failed.connect(lambda msg, tb: self._fail(w, label, msg, tb, on_fail))
        self.current = (label, w)
        self._keep.append(w)
        if self.status:
            self.status.showMessage('Running: ' + label)
        if self.pbar:
            self.pbar.setRange(0, 0); self.pbar.show()
        log('▶ ' + label)
        self.busy_changed.emit(True)
        self.pool.start(w)
        return w

    def cancel(self):
        if self.current:
            self.current[1].cancel()
            log('Cancel requested for ' + self.current[0])

    def _progress(self, i, n):
        if self.pbar and n > 0:
            self.pbar.setRange(0, n); self.pbar.setValue(i)

    def _end(self, w):
        self.current = None
        if w in self._keep:
            self._keep.remove(w)
        if self.pbar:
            self.pbar.hide()
        self.busy_changed.emit(False)

    def _finish(self, w, label, out, cb):
        self._end(w)
        log('✔ {} done'.format(label))
        if self.status:
            self.status.showMessage(label + ' done', 5000)
        if cb:
            try:
                cb(out)
            except Exception as e:
                show_error(self.parent(), label, str(e), traceback.format_exc())

    def _fail(self, w, label, msg, tb, cb):
        self._end(w)
        log('✖ {} failed: {}'.format(label, msg))
        if self.status:
            self.status.showMessage(label + ' failed', 8000)
        if cb:
            cb(msg)
        else:
            show_error(self.parent(), label, msg, tb)


def show_error(parent, title, msg, tb=None):
    box = QtWidgets.QMessageBox(parent)
    box.setIcon(QtWidgets.QMessageBox.Warning)
    box.setWindowTitle(title)
    box.setText(msg)
    if tb and 'MissingStep' not in tb:
        box.setDetailedText(tb)
    box.exec()


# ---------------------------------------------------------------------------
# parameter forms
# ---------------------------------------------------------------------------
class Param(object):
    """kind: int | float | str | bool | choice | multi | unit
    choices: list or callable(sess) -> list   (for choice / multi / unit)"""

    def __init__(self, name, label=None, kind='float', default=None, choices=None,
                 minv=None, maxv=None, step=None, tip=None):
        self.name, self.label, self.kind = name, label or name, kind
        self.default, self.choices = default, choices
        self.minv, self.maxv, self.step, self.tip = minv, maxv, step, tip


class ParamForm(QtWidgets.QWidget):
    changed = QtCore.Signal()

    def __init__(self, params, sess_getter, parent=None):
        super().__init__(parent)
        self.params = params
        self.sess_getter = sess_getter
        self.widgets = {}
        lay = QtWidgets.QFormLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setLabelAlignment(QtCore.Qt.AlignRight)
        for p in params:
            w = self._make(p)
            if p.tip:
                w.setToolTip(p.tip)
            self.widgets[p.name] = w
            lay.addRow(p.label, w)

    def _make(self, p):
        k = p.kind
        if k == 'int':
            w = QtWidgets.QSpinBox()
            w.setRange(int(p.minv if p.minv is not None else -10 ** 9),
                       int(p.maxv if p.maxv is not None else 10 ** 9))
            w.setValue(int(p.default if p.default is not None else 0))
            w.valueChanged.connect(self.changed)
        elif k == 'float':
            w = QtWidgets.QDoubleSpinBox()
            w.setDecimals(4)
            w.setRange(float(p.minv if p.minv is not None else -1e9), float(p.maxv if p.maxv is not None else 1e9))
            w.setSingleStep(float(p.step or 0.1))
            w.setValue(float(p.default if p.default is not None else 0))
            w.valueChanged.connect(self.changed)
        elif k == 'bool':
            w = QtWidgets.QCheckBox()
            w.setChecked(bool(p.default))
            w.toggled.connect(self.changed)
        elif k == 'str':
            w = QtWidgets.QLineEdit(str(p.default or ''))
            w.editingFinished.connect(self.changed)
        elif k in ('choice', 'unit'):
            w = QtWidgets.QComboBox()
            w.setMinimumContentsLength(14)
            w.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            if k == 'unit':
                w.setEditable(True)
                w.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
            w.currentIndexChanged.connect(self.changed)
            if k == 'unit':
                box = QtWidgets.QWidget()
                hl = QtWidgets.QHBoxLayout(box); hl.setContentsMargins(0, 0, 0, 0); hl.setSpacing(2)
                bp = QtWidgets.QToolButton(); bp.setText('◀'); bp.setAutoRaise(True)
                bn = QtWidgets.QToolButton(); bn.setText('▶'); bn.setAutoRaise(True)
                bp.clicked.connect(lambda: w.setCurrentIndex(max(0, w.currentIndex() - 1)))
                bn.clicked.connect(lambda: w.setCurrentIndex(min(w.count() - 1, w.currentIndex() + 1)))
                hl.addWidget(bp); hl.addWidget(w, 1); hl.addWidget(bn)
                box.combo = w
                return box
        elif k == 'multi':
            w = QtWidgets.QListWidget()
            w.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
            w.setMaximumHeight(120)
            w.itemSelectionChanged.connect(self.changed)
        else:
            raise ValueError(k)
        return w

    def _combo(self, w):
        return getattr(w, 'combo', w)

    def refresh(self):
        sess = self.sess_getter()
        for p in self.params:
            if p.kind not in ('choice', 'unit', 'multi'):
                continue
            w = self._combo(self.widgets[p.name])
            ch = p.choices(sess) if callable(p.choices) else (p.choices or [])
            try:
                ch = list(ch)
            except Exception:
                ch = []
            w.blockSignals(True)
            if p.kind == 'multi':
                sel = {i.text() for i in w.selectedItems()}
                w.clear()
                for c in ch:
                    it = QtWidgets.QListWidgetItem(str(c))
                    w.addItem(it)
                    want = sel if sel else set(str(x) for x in (p.default or []))
                    if str(c) in want:
                        it.setSelected(True)
            else:
                cur = w.currentText()
                w.clear()
                for c in ch:
                    if isinstance(c, tuple):
                        w.addItem(str(c[1]), c[0])
                    else:
                        w.addItem(str(c), c)
                idx = w.findText(cur) if cur else -1
                if idx < 0 and p.default is not None:
                    idx = w.findText(str(p.default))
                    if idx < 0:
                        idx = w.findData(p.default)
                w.setCurrentIndex(max(0, idx) if w.count() else -1)
            w.blockSignals(False)

    def values(self):
        out = {}
        for p in self.params:
            w = self.widgets[p.name]
            k = p.kind
            if k in ('int',):
                out[p.name] = w.value()
            elif k == 'float':
                out[p.name] = w.value()
            elif k == 'bool':
                out[p.name] = w.isChecked()
            elif k == 'str':
                out[p.name] = w.text()
            elif k in ('choice', 'unit'):
                c = self._combo(w)
                d = c.currentData()
                if d is None and k == 'unit':
                    txt = c.currentText().strip().split(' ')[0].lstrip('u')
                    try:
                        d = int(txt)
                    except ValueError:
                        d = None
                out[p.name] = d if d is not None else c.currentText()
            elif k == 'multi':
                items = w.selectedItems()
                out[p.name] = [i.text() for i in items]
        return out

    def set_value(self, name, value):
        w = self._combo(self.widgets[name])
        if isinstance(w, QtWidgets.QComboBox):
            i = w.findData(value)
            if i < 0:
                i = w.findText(str(value))
            if i >= 0:
                w.setCurrentIndex(i)
        elif isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            w.setValue(value)
        elif isinstance(w, QtWidgets.QCheckBox):
            w.setChecked(bool(value))

    def combo(self, name):
        return self._combo(self.widgets[name])


# ---------------------------------------------------------------------------
# matplotlib figure panel
# ---------------------------------------------------------------------------
class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 6), dpi=100, layout=None)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)


class FigurePanel(QtWidgets.QWidget):
    """Parameter form on the left, figure on the right.

    plot_fn(fig, sess, **values) -> dict | None, keys:
       filename  suggested file name for "Save to session folder"
       text      summary text shown under the form
       events    list of (mpl_event_name, handler) connected until the next redraw
       height    preferred figure height in inches (canvas grows, scroll area)
    """

    def __init__(self, sess_getter, plot_fn, params=(), title='', help_text='',
                 auto=False, parent=None, extra_buttons=()):
        super().__init__(parent)
        self.sess_getter = sess_getter
        self.plot_fn = plot_fn
        self.title = title
        self._cids = []
        self._last = {}
        self.auto = auto
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        left = QtWidgets.QWidget()
        left.setMaximumWidth(330)
        left.setMinimumWidth(240)
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        if title:
            t = QtWidgets.QLabel('<b>{}</b>'.format(title))
            t.setWordWrap(True)
            ll.addWidget(t)
        if help_text:
            h = QtWidgets.QLabel(help_text)
            h.setWordWrap(True)
            h.setStyleSheet('color: #666;')
            ll.addWidget(h)
        self.form = ParamForm(list(params), sess_getter)
        ll.addWidget(self.form)
        row = QtWidgets.QHBoxLayout()
        self.btn = QtWidgets.QPushButton('Plot')
        self.btn.setDefault(True)
        self.btn.clicked.connect(self.redraw)
        self.save_btn = QtWidgets.QPushButton('Save figure')
        self.save_btn.setToolTip('Save PNG + EPS into the session (SAVE_DIR) folder')
        self.save_btn.clicked.connect(self.save_to_session)
        row.addWidget(self.btn); row.addWidget(self.save_btn)
        ll.addLayout(row)
        for label, cb in extra_buttons:
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(cb)
            ll.addWidget(b)
        self.auto_chk = QtWidgets.QCheckBox('Redraw on change')
        self.auto_chk.setChecked(auto)
        ll.addWidget(self.auto_chk)
        self.form.changed.connect(self._maybe_auto)
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setStyleSheet('font-family: Consolas, Menlo, monospace; font-size: 9pt;')
        ll.addWidget(self.text, 1)
        lay.addWidget(left)

        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.canvas = MplCanvas()
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        rl.addWidget(self.toolbar)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.canvas)
        rl.addWidget(self.scroll, 1)
        lay.addWidget(right, 1)
        self.canvas.mpl_connect('key_press_event', self._key)

    def _key(self, ev):
        if ev.key not in ('left', 'right'):
            return
        d = 1 if ev.key == 'right' else -1
        for p in self.form.params:
            w = self.form.widgets[p.name]
            if p.kind == 'unit':
                c = self.form._combo(w)
                c.setCurrentIndex(int(np.clip(c.currentIndex() + d, 0, c.count() - 1)))
                break
            if p.name == 'trial' and p.kind == 'int':
                w.setValue(max(0, w.value() + d))
                break
        else:
            return
        if not self.auto_chk.isChecked():
            self.redraw()

    @property
    def fig(self):
        return self.canvas.fig

    def _maybe_auto(self):
        if self.auto_chk.isChecked():
            QtCore.QTimer.singleShot(0, self.redraw)

    def refresh_choices(self):
        self.form.refresh()

    def redraw(self):
        sess = self.sess_getter()
        for c in self._cids:
            self.canvas.mpl_disconnect(c)
        self._cids = []
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            out = self._plot_once(sess)
            h = out.get('height')
            want = int(h * self.fig.dpi) if h else 0
            if want != getattr(self, '_minh', 0):
                # figure needs a different height: resize the canvas, then lay out again
                self._minh = want
                self.canvas.setMinimumHeight(want)
                QtWidgets.QApplication.processEvents()
                out = self._plot_once(sess)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._last = out
        for ev, fn in out.get('events', []):
            self._cids.append(self.canvas.mpl_connect(ev, fn))
        if 'text' in out:
            self.text.setPlainText(out['text'])
        self.canvas.draw_idle()
        self.canvas.setFocus()

    def _plot_once(self, sess):
        self.fig.clear()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return self.plot_fn(self.fig, sess, **self.form.values()) or {}
        except MissingStep as e:
            self._message(str(e))
        except Exception as e:
            self._message('Error: {}'.format(e))
            log(traceback.format_exc())
        return {}

    def _message(self, msg):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.axis('off')
        ax.text(0.5, 0.5, msg, ha='center', va='center', wrap=True, fontsize=11, color='#a33')
        self.text.setPlainText(msg)

    def save_to_session(self):
        sess = self.sess_getter()
        name = self._last.get('filename') or (self.title or 'figure').replace(' ', '_').lower() + '.png'
        d = sess.cfg.save_dir
        if not d:
            p, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save figure', name,
                                                         'PNG (*.png);;PDF (*.pdf);;EPS (*.eps);;SVG (*.svg)')
            if not p:
                return
            self.fig.savefig(p, dpi=150, bbox_inches='tight', facecolor='white')
            return
        base = os.path.splitext(os.path.join(d, name))[0]
        self.fig.savefig(base + '.png', dpi=150, bbox_inches='tight', facecolor='white')
        try:
            self.fig.savefig(base + '.eps', dpi=150, bbox_inches='tight', facecolor='white')
        except Exception:
            pass
        log('Saved {}.png/.eps'.format(base))


def unit_choices(sess, ids=None):
    """[(cid, 'u12  850µm')]"""
    try:
        d = sess.unit_depths
        ids = sess.good_ids if ids is None else ids
        return [(int(c), 'u{}  {:.0f}µm'.format(c, d.get(c, np.nan))) for c in ids]
    except Exception:
        return []


def hline():
    f = QtWidgets.QFrame()
    f.setFrameShape(QtWidgets.QFrame.HLine)
    f.setFrameShadow(QtWidgets.QFrame.Sunken)
    return f

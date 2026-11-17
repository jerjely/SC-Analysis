"""
Interactive peak pickers (cells P1 / P2) as Qt widgets.

Left click = mark peak, right click = remove last mark in that panel,
Enter / N = accept + next, S = skip, P = previous.  Picks are saved to the
same JSON + CSV files the notebook used after every accepted cell.
"""
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..core import pickers
from .common import MplCanvas, NavigationToolbar2QT, log, show_error


class _PickerBase(QtWidgets.QWidget):
    json_name = ''
    title = ''

    def __init__(self, sess_getter, parent=None):
        super().__init__(parent)
        self.sess_getter = sess_getter
        self.cells = []
        self.all_picks = {}
        self.cur = -1
        self.picks = {}
        self.markers = {}
        self.data = None
        self.panels = []
        lay = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QWidget(); left.setMaximumWidth(280)
        ll = QtWidgets.QVBoxLayout(left); ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QtWidgets.QLabel('<b>{}</b>'.format(self.title)))
        h = QtWidgets.QLabel('Left click = mark peak · right click = undo last in panel\n'
                             'Enter/N = accept + next · S = skip · P = previous\nSaved after every accepted cell.')
        h.setWordWrap(True); h.setStyleSheet('color:#666')
        ll.addWidget(h)
        self.only_todo = QtWidgets.QCheckBox('Show only cells not yet picked')
        self.only_todo.toggled.connect(self._fill)
        ll.addWidget(self.only_todo)
        self.list = QtWidgets.QListWidget()
        self.list.currentRowChanged.connect(self._goto)
        ll.addWidget(self.list, 1)
        self.status = QtWidgets.QLabel('')
        ll.addWidget(self.status)
        g = QtWidgets.QGridLayout()
        b = QtWidgets.QPushButton('Load / refresh'); b.clicked.connect(self.start); g.addWidget(b, 0, 0, 1, 2)
        b = QtWidgets.QPushButton('◀ Prev (P)'); b.clicked.connect(lambda: self._step(-1)); g.addWidget(b, 1, 0)
        b = QtWidgets.QPushButton('Skip (S)'); b.clicked.connect(lambda: self._step(1)); g.addWidget(b, 1, 1)
        b = QtWidgets.QPushButton('Clear marks'); b.clicked.connect(self._clear); g.addWidget(b, 2, 0)
        b = QtWidgets.QPushButton('Delete saved'); b.clicked.connect(self._delete_saved); g.addWidget(b, 2, 1)
        self.accept_btn = QtWidgets.QPushButton('✔ Accept + next (Enter)')
        self.accept_btn.setStyleSheet('font-weight:bold; padding:6px')
        self.accept_btn.clicked.connect(self.accept)
        g.addWidget(self.accept_btn, 3, 0, 1, 2)
        ll.addLayout(g)
        lay.addWidget(left)
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right); rl.setContentsMargins(0, 0, 0, 0)
        self.canvas = MplCanvas()
        rl.addWidget(NavigationToolbar2QT(self.canvas, self))
        rl.addWidget(self.canvas, 1)
        lay.addWidget(right, 1)
        self.canvas.mpl_connect('button_press_event', self._click)
        self.canvas.mpl_connect('key_press_event', self._key)
        for k, fn in [('Return', self.accept), ('Enter', self.accept), ('N', self.accept),
                      ('S', lambda: self._step(1)), ('P', lambda: self._step(-1))]:
            sc = QtGui.QShortcut(QtGui.QKeySequence(k), self)
            sc.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)

    # --- to implement ----------------------------------------------------
    def make_cells(self, sess):
        raise NotImplementedError

    def cell_id(self, c):
        raise NotImplementedError

    def cell_label(self, c):
        return str(self.cell_id(c))

    def build_data(self, sess, c):
        raise NotImplementedError

    def conds(self):
        raise NotImplementedError

    def draw(self, sess, c):
        raise NotImplementedError

    def compile(self, sess, c):
        raise NotImplementedError

    def write_csv(self, sess):
        raise NotImplementedError

    def mark_text(self, cond, x):
        return ''

    # --- common ------------------------------------------------------------
    def start(self):
        sess = self.sess_getter()
        try:
            self.cells = self.make_cells(sess)
        except Exception as e:
            show_error(self, self.title, str(e)); return
        self.all_picks = pickers.load_picks(sess.p(self.json_name), log)
        self._fill()
        log('{}: {} cells, {} already picked'.format(self.title, len(self.cells), len(self.all_picks)))

    def _visible(self):
        if self.only_todo.isChecked():
            return [c for c in self.cells if str(self.cell_id(c)) not in self.all_picks]
        return list(self.cells)

    def _fill(self):
        self.vis = self._visible()
        self.list.blockSignals(True)
        self.list.clear()
        for c in self.vis:
            done = str(self.cell_id(c)) in self.all_picks
            it = QtWidgets.QListWidgetItem(('✔ ' if done else '   ') + self.cell_label(c))
            self.list.addItem(it)
        self.list.blockSignals(False)
        n_done = sum(str(self.cell_id(c)) in self.all_picks for c in self.cells)
        self.status.setText('{} / {} picked'.format(n_done, len(self.cells)))
        if self.vis:
            first = next((i for i, c in enumerate(self.vis) if str(self.cell_id(c)) not in self.all_picks), 0)
            self.list.setCurrentRow(first)

    def _goto(self, row):
        if row < 0 or row >= len(getattr(self, 'vis', [])):
            return
        self.cur = row
        sess = self.sess_getter()
        c = self.vis[row]
        try:
            self.data = self.build_data(sess, c)
        except Exception as e:
            show_error(self, self.title, str(e)); return
        self.picks = {k: [] for k in self.conds()}
        self.markers = {k: [] for k in self.conds()}
        self.canvas.fig.clear()
        self.panels = self.draw(sess, c)
        prev = self.all_picks.get(str(self.cell_id(c)))
        if prev:
            self.restore(prev)
        self.canvas.draw_idle()
        self.canvas.setFocus()

    def restore(self, prev):
        pass

    def _panel(self, ev):
        for ax_list, cond, d, plot_ax in self.panels:
            if ev.inaxes in ax_list:
                return cond, d, plot_ax
        return None, None, None

    def add_mark(self, cond, d, ax, x):
        bc = d['bc'] * 1000
        i = int(np.argmin(np.abs(bc - x)))
        y = float(d['mean_fr'][i])
        self.picks[cond].append(x)
        k = len(self.picks[cond])
        m, = ax.plot(x, y, 'v', color='black', ms=10, mec='white', mew=1.2, zorder=20)
        t = ax.text(x, y * 1.12, '#{}\n{}'.format(k, self.mark_text(cond, x)), ha='center', va='bottom', fontsize=6,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='gray', lw=0.5),
                    zorder=21)
        self.markers[cond].append((m, t))

    def _click(self, ev):
        if ev.xdata is None or self.canvas.toolbar is not None and getattr(self.canvas.toolbar, 'mode', ''):
            return
        cond, d, ax = self._panel(ev)
        if cond is None:
            return
        if ev.button == 1:
            self.add_mark(cond, d, ax, float(ev.xdata))
        elif ev.button == 3 and self.picks[cond]:
            self.picks[cond].pop()
            m, t = self.markers[cond].pop()
            m.remove(); t.remove()
        self.canvas.draw_idle()

    def _key(self, ev):
        if ev.key in ('enter', 'n', 'N'):
            self.accept()
        elif ev.key in ('s', 'S'):
            self._step(1)
        elif ev.key in ('p', 'P'):
            self._step(-1)

    def _clear(self):
        for cond in list(self.markers):
            for m, t in self.markers[cond]:
                m.remove(); t.remove()
            self.markers[cond] = []
            self.picks[cond] = []
        self.canvas.draw_idle()

    def _step(self, d):
        r = self.list.currentRow() + d
        if 0 <= r < self.list.count():
            self.list.setCurrentRow(r)

    def accept(self):
        if self.cur < 0 or self.data is None:
            return
        sess = self.sess_getter()
        c = self.vis[self.cur]
        res = self.compile(sess, c)
        self.all_picks[str(self.cell_id(c))] = res
        pickers.save_picks(self.all_picks, sess.p(self.json_name))
        self.write_csv(sess)
        sess.cache.pop('picks', None)
        it = self.list.item(self.cur)
        it.setText('✔ ' + self.cell_label(c))
        n_done = sum(str(self.cell_id(x)) in self.all_picks for x in self.cells)
        self.status.setText('{} / {} picked'.format(n_done, len(self.cells)))
        log('{}: saved unit {} ({} clicks)'.format(self.title, self.cell_id(c), sum(len(v) for v in self.picks.values())))
        self._step(1)

    def _delete_saved(self):
        if self.cur < 0:
            return
        sess = self.sess_getter()
        c = self.vis[self.cur]
        k = str(self.cell_id(c))
        if k in self.all_picks:
            del self.all_picks[k]
            pickers.save_picks(self.all_picks, sess.p(self.json_name))
            self.write_csv(sess)
            self.list.item(self.cur).setText('   ' + self.cell_label(c))
            self._clear()


class BlinkLoomPicker(_PickerBase):
    json_name = 'blink_loom_picks.json'
    title = 'Blink / loom peak picker'
    pre_s = 0.5

    def make_cells(self, sess):
        if not pickers.blink_loom_conditions(sess):
            raise ValueError('None of the blink/loom picker conditions exist in P1 '
                             '(edit them in Session settings > Analysis)')
        return [int(c) for c in sess.good_ids]

    def cell_id(self, c):
        return c

    def cell_label(self, c):
        d = self.sess_getter().unit_depths.get(c, np.nan)
        return 'u{}  {:.0f}µm'.format(c, d)

    def conds(self):
        return [c for c, _, _ in self._conds]

    def build_data(self, sess, c):
        self._conds = pickers.blink_loom_conditions(sess)
        return pickers.blink_loom_data(sess, c, self.pre_s)

    def mark_text(self, cond, x):
        dur = self.data[cond]['stim_dur'] * 1000
        return 'ON:{:+d}ms\nOFF:{:+d}ms'.format(int(x), int(x - dur))

    def draw(self, sess, c):
        fig = self.canvas.fig
        conds = self._conds
        axes = fig.subplots(2, len(conds), gridspec_kw={'height_ratios': [2, 1]}, squeeze=False)
        fig.subplots_adjust(hspace=0.35, wspace=0.25, top=0.86, bottom=0.10, left=0.05, right=0.98)
        st = sess.spike_trains[c]
        fig.suptitle('Unit {} | {:.0f}µm | base={:.1f}Hz'.format(
            c, sess.unit_depths.get(c, 0), len(st) / max(sess.spikes['t_max'], 1)), fontsize=10, fontweight='bold')
        panels = []
        for k, (cond, lab, col) in enumerate(conds):
            d = self.data[cond]
            ar, ap = axes[0, k], axes[1, k]
            bc = d['bc'] * 1000; dur = d['stim_dur'] * 1000; post = d['post_s'] * 1000
            xs = [r * 1000 for r in d['raster']]
            for j, r in enumerate(xs):
                ar.vlines(r, j, j + 0.8, color='black', lw=0.5, alpha=0.7)
            for ax in (ar, ap):
                ax.axvspan(0, dur, alpha=0.07, color=col); ax.axvline(0, color='k', ls='--', lw=1)
                ax.axvline(dur, color=col, ls='--', lw=1, alpha=0.7); ax.set_xlim(-self.pre_s * 1000, post)
            ar.set_ylim(-0.5, d['n']); ar.tick_params(labelbottom=False)
            ar.set_title('{}\nn={} dur={:.0f}ms'.format(lab, d['n'], dur), fontsize=8, color=col)
            up = d['mean_fr'] + d['sem_fr']
            ap.fill_between(bc, np.clip(d['mean_fr'] - d['sem_fr'], 0, None), up, alpha=0.3, color=col)
            ap.plot(bc, d['mean_fr'], color=col, lw=2)
            ap.set_ylim(0, max(float(up.max()), 1.0) * 1.35)
            ap.set_xlabel('ms', fontsize=8)
            if k == 0:
                ar.set_ylabel('Trial'); ap.set_ylabel('FR (Hz)')
            panels.append(((ar, ap), cond, d, ap))
        return panels

    def restore(self, prev):
        for ax_list, cond, d, ax in self.panels:
            n = int(prev.get(cond + '_n_picks', 0) or 0)
            for k in range(1, n + 1):
                x = prev.get('{}_p{}_ms'.format(cond, k))
                if x is not None and np.isfinite(x):
                    self.add_mark(cond, d, ax, float(x))

    def compile(self, sess, c):
        return pickers.compile_blink_loom(sess, c, self.data, self.picks)

    def write_csv(self, sess):
        pickers.write_blink_loom_csv(sess, self.all_picks)


class SweepPicker(_PickerBase):
    json_name = 'interactive_picks.json'
    title = 'Sweep peak picker'

    def make_cells(self, sess):
        cells = pickers.sweep_cell_list(sess)
        if not cells:
            raise ValueError('No sweep-responsive cells found (needs small_sweep conditions in P1)')
        return cells

    def cell_id(self, c):
        return int(c['cid'])

    def cell_label(self, c):
        src = 'RF' if c['cid'] in self.sess_getter().rf_clean else 'sweep~'
        return 'u{}  RF {:+.1f}°  {:.0f}µm  [{}]'.format(c['cid'], c['rf_x_deg'], c['depth'], src)

    def conds(self):
        return list(self.data.keys())

    def build_data(self, sess, c):
        return pickers.sweep_data(sess, c)

    def mark_text(self, cond, x):
        d = self.data[cond]
        return '→LEAD:{:+.0f}ms\n→TRAIL:{:+.0f}ms'.format(x - d['leading_ms'], x - d['trailing_ms'])

    def draw(self, sess, c):
        fig = self.canvas.fig
        n = len(self.data)
        axes = fig.subplots(n, 1, squeeze=False)[:, 0]
        fig.subplots_adjust(hspace=0.95, left=0.07, right=0.97, top=0.9, bottom=0.06)
        first = next(iter(self.data.values()))
        fig.suptitle('Unit {} | RF centre={:.1f}° RF left={:.1f}° diam={:.1f}° base={:.1f}Hz {:.0f}µm'.format(
            c['cid'], c['rf_x_deg'], first['rf_left'], c['rf_diam'], c['base_fr'], c['depth']),
            fontsize=10, fontweight='bold')
        panels = []
        for ax, (cond, d) in zip(axes, self.data.items()):
            col = sess.speed_colors.get(cond, 'k')
            bc = d['bc'] * 1000; dur = d['dur'] * 1000
            up = d['mean_fr'] + d['sem_fr']
            ax.fill_between(bc, np.clip(d['mean_fr'] - d['sem_fr'], 0, None), up, alpha=0.25, color=col)
            ax.plot(bc, d['mean_fr'], color=col, lw=2)
            ax.axvspan(0, dur, alpha=0.06, color='gray'); ax.axvline(0, color='k', ls='--', lw=1)
            ax.axvline(dur, color='gray', ls='--', lw=1)
            ax.axvline(d['leading_ms'], color=col, lw=2.5, zorder=5, alpha=0.9)
            ax.axvline(d['trailing_ms'], color=col, ls=':', lw=2.5, zorder=5, alpha=0.9)
            yt = max(float(up.max()), 1.0)
            ax.text(d['leading_ms'] + 15, yt * 0.88, 'LEAD', fontsize=6, color=col, fontweight='bold', va='top')
            ax.text(d['trailing_ms'] + 15, yt * 0.88, 'TRAIL', fontsize=6, color=col, fontweight='bold', va='top')
            ax.set_xlim(-500, max(dur + 500, d['trailing_ms'] + 800))
            ax.set_ylim(0, max(yt * 1.3, 5)); ax.set_ylabel('FR (Hz)', fontsize=8)
            ax.set_title('{} dps n={} dur={:.1f}s | lead={:.0f}ms{} trail={:.0f}ms'.format(
                d['spd'], d['n'], d['dur'], d['leading_ms'], ' [C]' if d['clamped'] else '', d['trailing_ms']),
                fontsize=8, color=col)
            panels.append(((ax,), cond, d, ax))
        axes[-1].set_xlabel('Time from sweep onset (ms)')
        return panels

    def restore(self, prev):
        for ax_list, cond, d, ax in self.panels:
            spd = d['spd']
            n = int(prev.get('n_picks_{}dps'.format(spd), 0) or 0)
            for k in range(1, n + 1):
                x = prev.get('p{}_ms_{}dps'.format(k, spd))
                if x is not None and np.isfinite(x):
                    self.add_mark(cond, d, ax, float(x))

    def compile(self, sess, c):
        return pickers.compile_sweep(sess, c, self.data, self.picks)

    def write_csv(self, sess):
        pickers.write_sweep_csv(sess, self.all_picks)

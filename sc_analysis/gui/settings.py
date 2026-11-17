"""Session settings dialog + new-session helper."""
import json
import os
import re

from PySide6 import QtCore, QtWidgets

from ..core.config import FIELD_GROUPS, SessionConfig


class PathEdit(QtWidgets.QWidget):
    def __init__(self, kind, value='', parent=None):
        super().__init__(parent)
        self.kind = kind
        h = QtWidgets.QHBoxLayout(self); h.setContentsMargins(0, 0, 0, 0)
        self.edit = QtWidgets.QLineEdit(value or '')
        b = QtWidgets.QToolButton(); b.setText('…')
        b.clicked.connect(self._browse)
        h.addWidget(self.edit, 1); h.addWidget(b)

    def _browse(self):
        cur = self.edit.text() or os.path.expanduser('~')
        if self.kind == 'path_dir':
            p = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose folder', cur)
        else:
            p, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Choose file', os.path.dirname(cur))
        if p:
            self.edit.setText(os.path.normpath(p))

    def value(self):
        return self.edit.text().strip()


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Session settings')
        self.resize(760, 640)
        self.cfg = cfg
        self.widgets = {}
        lay = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget()
        for group, fields in FIELD_GROUPS:
            w = QtWidgets.QWidget()
            f = QtWidgets.QFormLayout(w)
            for key, label, kind in fields:
                v = cfg.get(key)
                if kind in ('path_dir', 'path_file'):
                    ed = PathEdit(kind, v)
                elif kind == 'bool':
                    ed = QtWidgets.QCheckBox(); ed.setChecked(bool(v))
                elif kind == 'int':
                    ed = QtWidgets.QSpinBox(); ed.setRange(-10 ** 9, 10 ** 9); ed.setValue(int(v or 0))
                elif kind == 'float':
                    ed = QtWidgets.QDoubleSpinBox(); ed.setDecimals(4); ed.setRange(-1e9, 1e9); ed.setValue(float(v or 0))
                elif kind == 'list':
                    ed = QtWidgets.QLineEdit(', '.join(str(x) for x in (v or [])))
                elif kind == 'json':
                    ed = QtWidgets.QPlainTextEdit(json.dumps(v, indent=1))
                    ed.setMaximumHeight(110)
                else:
                    ed = QtWidgets.QLineEdit('' if v is None else str(v))
                if key == 'rf_mode':
                    ed = QtWidgets.QComboBox(); ed.addItems(['consistent', 'RFM1', 'RFM2', 'combined'])
                    ed.setCurrentText(str(v))
                    kind = 'combo'
                self.widgets[key] = (ed, kind)
                f.addRow(label, ed)
            sc = QtWidgets.QScrollArea(); sc.setWidgetResizable(True); sc.setWidget(w)
            tabs.addTab(sc, group)
        lay.addWidget(tabs, 1)
        note = QtWidgets.QLabel('Settings are saved as sc_session.json in the analysis folder. '
                                'Defaults match SC_analysis_v24.ipynb.')
        note.setStyleSheet('color:#666')
        lay.addWidget(note)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self._ok); bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    @staticmethod
    def _parse_list(txt):
        out = []
        for x in [t.strip() for t in txt.split(',') if t.strip()]:
            try:
                out.append(int(x))
            except ValueError:
                try:
                    out.append(float(x))
                except ValueError:
                    out.append(x)
        return out

    def _ok(self):
        vals = {}
        for key, (ed, kind) in self.widgets.items():
            try:
                if kind in ('path_dir', 'path_file'):
                    vals[key] = ed.value()
                elif kind == 'bool':
                    vals[key] = ed.isChecked()
                elif kind in ('int', 'float'):
                    vals[key] = ed.value()
                elif kind == 'list':
                    vals[key] = self._parse_list(ed.text())
                elif kind == 'json':
                    vals[key] = json.loads(ed.toPlainText())
                elif kind == 'combo':
                    vals[key] = ed.currentText()
                else:
                    vals[key] = ed.text().strip()
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, 'Settings', 'Could not parse "{}": {}'.format(key, e))
                return
        self.cfg.update(vals)
        self.accept()


def guess_from_sorter(sorter_dir):
    """Infer SAVE_DIR and recording identity from a KS sorter_output folder
    (notebook layout: <SAVE_DIR>/ks4_output_concat/sorter_output)."""
    from ..core.npx_io import find_sorter_dir
    picked = os.path.normpath(sorter_dir)
    sd = os.path.normpath(find_sorter_dir(picked) or picked)
    save = sd
    if os.path.basename(sd).lower() == 'sorter_output':
        save = os.path.dirname(os.path.dirname(sd))
    elif sd != picked:
        save = picked          # user picked the analysis (…_concat) folder itself
    vals = {'sorter_dir': sd, 'save_dir': save}
    name = os.path.basename(save)
    from ..core.config import split_identity
    m = split_identity(name)
    if m:
        vals.update({'recording_date': m[0], 'mouse_line': m[1], 'recording_region': m[2]})
        prefix = '{}_{}_{}'.format(*m)
        ups = [os.path.dirname(save), os.path.dirname(os.path.dirname(save)),
               os.path.dirname(os.path.dirname(os.path.dirname(save)))]
        for u in ups:
            cand = os.path.join(u, prefix)
            if u and os.path.isdir(cand) and os.path.normpath(cand) != os.path.normpath(save):
                vals['psychopy_root'] = cand
                break
        for u in [save] + ups:
            cand = os.path.join(u, 'sparse_object_frames.npy') if u else ''
            if cand and os.path.exists(cand):
                vals['frames_path'] = cand
                break
        for root in [r'D:\NPX', os.path.join(os.path.dirname(save), 'NPX')]:
            if os.path.isdir(os.path.join(root, prefix + '_P1_g0')) or os.path.isdir(root):
                vals['npx_root'] = root
                break
    # an existing notebook analysis folder: keep RFM choice / order consistent with saved files
    if os.path.exists(os.path.join(save, 'segment_boundaries.json')):
        try:
            with open(os.path.join(save, 'segment_boundaries.json')) as f:
                b = json.load(f)
            b.pop('fs', None)
            vals['recording_order'] = sorted(b, key=lambda s: b[s]['start_s'])
        except Exception:
            pass
    return vals


def new_session_dialog(parent):
    sd = QtWidgets.QFileDialog.getExistingDirectory(parent, 'Choose the Kilosort sorter_output folder (or the …_concat analysis folder)')
    if not sd:
        return None
    cfg = SessionConfig()
    cfg.update(guess_from_sorter(sd))
    p = os.path.join(cfg.save_dir, 'sc_session.json')
    if os.path.exists(p):
        r = QtWidgets.QMessageBox.question(parent, 'Session exists',
                                           'A session file already exists in\n{}\nOpen it instead?'.format(cfg.save_dir))
        if r == QtWidgets.QMessageBox.Yes:
            return SessionConfig.load(p)
    dlg = SettingsDialog(cfg, parent)
    dlg.setWindowTitle('New session - check paths')
    if dlg.exec() != QtWidgets.QDialog.Accepted:
        return None
    try:
        cfg.save()
    except Exception as e:
        QtWidgets.QMessageBox.warning(parent, 'Save', str(e))
    return cfg

"""
Built-in self-test:  SC-Analysis --selftest [report.txt] [--keep DIR]

Generates two small synthetic recordings, runs the whole pipeline, opens the
real main window off-screen, draws every analysis panel, drives both peak
pickers, loads the trace viewer and runs the pooled (multi-recording)
classification. Exit code 0 = everything worked, 1 = something failed.
Used by the GitHub Actions build so a broken installer is never released.
"""
import os
import shutil
import sys
import tempfile
import time
import traceback


class _Report(object):
    def __init__(self, path):
        self.path = path
        self.lines = []
        self.failures = []
        self.t0 = time.time()

    def __call__(self, *msg):
        line = ' '.join(str(m) for m in msg)
        self.lines.append(line)
        if sys.stdout is not None:
            try:
                print(line, flush=True)
            except Exception:
                pass
        if self.path:
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(line + '\n')

    def fail(self, what, detail=''):
        self.failures.append(what)
        self('FAIL', what, detail)


def _bad_text(txt):
    head = txt[:60]
    return txt.startswith('Error') or 'Run pipeline' in txt or txt.startswith('No ') or 'first' in head


def run(report_path=None, keep=None):
    if report_path:
        report_path = os.path.abspath(report_path)
        open(report_path, 'w').close()
    R = _Report(report_path)
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    os.environ.setdefault('MPLBACKEND', 'Agg')
    root = os.path.abspath(keep) if keep else tempfile.mkdtemp(prefix='sc_selftest_')
    try:
        _run(R, root)
    except Exception:
        R.fail('unexpected exception', traceback.format_exc())
    finally:
        if not keep:
            shutil.rmtree(root, ignore_errors=True)
    R('')
    R('{} in {:.0f}s: {}'.format('SELF-TEST PASSED' if not R.failures else 'SELF-TEST FAILED',
                                 time.time() - R.t0,
                                 'all checks ok' if not R.failures else ', '.join(R.failures)))
    return 0 if not R.failures else 1


def _run(R, root):
    import platform
    import numpy as np
    from . import __version__
    R('SC Analysis {} self-test | Python {} | {} {}'.format(__version__, platform.python_version(),
                                                          platform.system(), platform.machine()))
    R('work folder:', root)

    # ---------------------------------------------------------- Qt (off-screen)
    from PySide6 import QtWidgets, QtCore
    QtCore.QCoreApplication.setOrganizationName('SCAnalysisSelfTest')
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(['sc-analysis-selftest'])
    dialogs = []

    def _err(parent, title, msg, tb=None):
        dialogs.append('{}: {}'.format(title, msg))
        R('  error dialog:', title, msg, tb or '')

    import importlib
    for name in ('common', 'picker_widgets', 'trace_view', 'tabs', 'pool_tab', 'pipeline_panel',
                 'settings', 'main_window'):
        mod = importlib.import_module('sc_analysis.gui.' + name)
        if hasattr(mod, 'show_error'):
            mod.show_error = _err
    QtWidgets.QMessageBox.exec = lambda self: 0
    for m in ('information', 'warning', 'critical', 'question'):
        setattr(QtWidgets.QMessageBox, m, staticmethod(lambda *a, **k: 0))

    def pump(sec=0.2):
        t = time.time()
        while time.time() - t < sec:
            app.processEvents()
            time.sleep(0.01)

    # ---------------------------------------------------------- data + pipeline
    from .synthetic import make_synthetic
    from .core.config import SessionConfig
    from .core.session import Session, STEPS
    from .core.pipeline import run_step
    from .core import baden, ccf, pupil, pooling

    quiet = lambda *a, **k: None  # noqa: E731
    sessions = []
    for i, (prefix, seed) in enumerate([('2026.01.01_DEMO_SC', 1), ('2026.01.02_DEMO_SC', 2)]):
        t = time.time()
        vals = make_synthetic(os.path.join(root, 'rec{}'.format(i + 1)), prefix, seed=seed,
                              n_units=24, n_frames=1000, rec_seconds=12.0, log=quiet)
        vals.update({'n_permutations': 100})
        cfg = SessionConfig(vals)
        path = cfg.save()
        R('synthetic recording {} written ({:.1f}s)'.format(prefix, time.time() - t))
        sess = Session(cfg)
        for key, label, _ in STEPS:
            t = time.time()
            try:
                run_step(sess, key, overwrite=True, log=quiet)
                R('  ok   {:28s} {:5.1f}s'.format(label, time.time() - t))
            except Exception:
                R.fail('pipeline {} {}'.format(prefix, label.strip()), traceback.format_exc())
        sessions.append(path)

    # ---------------------------------------------------------- main window
    from .gui.main_window import MainWindow
    w = MainWindow(sessions[0])
    w.resize(1500, 950)
    w.show()
    pump(0.5)
    s = w.sess
    if not len(s.good_ids):
        R.fail('no good units loaded')
    R('main window open: {} good units'.format(len(s.good_ids)))
    for f in (baden.run_qi, baden.run_features, baden.run_dsos, baden.run_colour, baden.run_clustering):
        try:
            f(s, log=quiet)
        except Exception:
            R.fail('Baden ' + f.__name__, traceback.format_exc())
    for f, args in ((baden.run_dsos, dict(pooled=False)), (ccf.run_ccf, {})):
        try:
            f(s, log=quiet, **args)
        except Exception:
            R.fail(f.__name__, traceback.format_exc())
    for seg, align in (('G2', pupil.align_grating_segment), ('P1', pupil.align_p1_segment)):
        try:
            align(s, seg, force=True)
            pupil.load_pupil(s, seg)
        except Exception:
            R.fail('pupil ' + seg, traceback.format_exc())

    # peak pickers (blink/loom + sweep)
    for pk in (w.bl_picker, w.sw_picker):
        try:
            pk.start()
            pump(0.2)
            n = 0
            while pk.cur >= 0 and n < 12:
                for _ax_list, cond, d, ax in pk.panels:
                    i = int(np.argmax(d['mean_fr']))
                    pk.add_mark(cond, d, ax, float(d['bc'][i] * 1000))
                prev = pk.cur
                pk.accept()
                pump(0.05)
                n += 1
                if pk.cur == prev:
                    break
            R('  ok   picker {:21s} {} picks'.format(pk.json_name, len(pk.all_picks)))
            if not pk.all_picks:
                R.fail('picker ' + pk.json_name, 'no picks stored')
        except Exception:
            R.fail('picker ' + getattr(pk, 'json_name', '?'), traceback.format_exc())
    w.refresh_all()

    # every analysis panel
    from .gui.tabs import SubTabs
    n_panels = 0
    for tab, name in [(w.overview, 'Overview'), (w.rf_tab, 'RF'), (w.unit_tab, 'Units'),
                      (w.pop_tab, 'Population'), (w.clu_tab, 'Clustering'), (w.sweep_tab, 'Sweeps'),
                      (w.loom_tab, 'Loom'), (w.ccf_tab, 'CCF'), (w.pupil_tab, 'Pupil')]:
        panels = tab.panels if isinstance(tab, SubTabs) else getattr(getattr(tab, 'tabs', None), 'panels', [])
        w.tabs.setCurrentWidget(tab)
        for pnl in panels:
            t = time.time()
            try:
                pnl.refresh_choices()
                pnl.redraw()
                pump(0.03)
                txt = pnl.text.toPlainText()
            except Exception:
                txt = 'Error ' + traceback.format_exc()
            n_panels += 1
            if _bad_text(txt):
                R.fail('panel {} / {}'.format(name, pnl.title), txt[:300].replace('\n', ' | '))
            else:
                R('  ok   {:12s} {:28s} {:5.2f}s'.format(name, pnl.title[:28], time.time() - t))
    R('{} panels drawn'.format(n_panels))

    # trace viewer on the concatenated binary
    try:
        w.tabs.setCurrentWidget(w.rec_tab)
        w.rec_tab.load()
        pump(0.3)
        w.rec_tab.t_spin.setValue(5.0)
        w.rec_tab.select_unit(int(s.good_ids[0]))
        pump(0.3)
        R('  ok   trace viewer')
    except Exception:
        R.fail('trace viewer', traceback.format_exc())

    # ---------------------------------------------------------- pooled classification
    try:
        pt = w.pool_tab
        lst = os.path.join(root, 'sc_recordings.json')
        rl = pooling.RecordingList()
        rl.save(lst)
        for f in pooling.scan(root):
            rl.add(f)
        rl.save()
        if len(rl.folders()) < 2:
            R.fail('pooled: scan', 'found {} recordings'.format(len(rl.folders())))
        _done, failed = pooling.export_missing(rl.folders(), overwrite=True, log=quiet)
        if failed:
            R.fail('pooled: export', str(failed))
        P = pooling.load_pooled(rl.included(), log=quiet)
        Rp = pooling.run_pooled(P, n_components=6, k_max=8, n_init_bic=2, n_init_final=4, log=quiet)
        pooling.save_pooled(rl, P, Rp, log=quiet)
        R('  ok   pooled clustering: {} units from {} recordings'.format(len(P['df']), len(rl.folders())))
        pt.state.rl = pooling.RecordingList(lst)
        pt._after_list(lst)
        pt.kmax.setValue(8)
        w.tabs.setCurrentWidget(pt)
        pump(0.2)
        pt.run_pooled()
        pump(0.3)
        while w.runner.current is not None:
            pump(0.2)
        pump(0.3)
        for i, p in enumerate(pt.tabs.panels):
            pt.tabs.setCurrentIndex(i)
            p.refresh_choices()
            p.redraw()
            pump(0.05)
            txt = p.text.toPlainText()
            if _bad_text(txt):
                R.fail('pooled panel ' + p.title, txt[:300].replace('\n', ' | '))
            else:
                R('  ok   Pooled       {:28s}'.format(p.title[:28]))
    except Exception:
        R.fail('pooled classification', traceback.format_exc())

    if dialogs:
        R.fail('{} error dialog(s)'.format(len(dialogs)), ' || '.join(dialogs)[:500])
    w.close()
    pump(0.1)


if __name__ == '__main__':
    sys.exit(run(sys.argv[1] if len(sys.argv) > 1 else None))

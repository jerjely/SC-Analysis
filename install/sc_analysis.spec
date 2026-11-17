# -*- mode: python ; coding: utf-8 -*-
# PyInstaller build file for SC Analysis (used by GitHub Actions and for local builds):
#     python -m PyInstaller --noconfirm --distpath dist --workpath build install/sc_analysis.spec
# Output: dist/SC-Analysis/ (Windows/Linux folder app)  or  dist/SC Analysis.app (macOS)
import os
import re
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, '..'))
RES = os.path.join(ROOT, 'sc_analysis', 'resources')
with open(os.path.join(ROOT, 'sc_analysis', '_version.py')) as f:
    VERSION = re.search(r"__version__\s*=\s*'([^']+)'", f.read()).group(1)

datas = collect_data_files('sc_analysis', includes=['resources/*'])
hidden = collect_submodules('sc_analysis') + [
    'sklearn.mixture', 'sklearn.decomposition', 'sklearn.utils._typedefs',
    'tables', 'matplotlib.backends.backend_qtagg',
]
excludes = ['tkinter', 'PyQt5', 'PyQt6', 'PySide2', 'IPython', 'jupyter', 'notebook',
            'pyqtgraph.examples', 'pyqtgraph.opengl', 'OpenGL', 'pytest', 'gi']

a = Analysis(
    [os.path.join(SPECPATH, 'launcher.py')],
    pathex=[ROOT],
    datas=datas,
    hiddenimports=hidden,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

if sys.platform == 'darwin':
    icon = os.path.join(RES, 'icon.icns')
elif sys.platform == 'win32':
    icon = os.path.join(RES, 'icon.ico')
else:
    icon = None

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name='SC-Analysis',
    console=False,                 # no console window (self-test writes a report file)
    icon=icon,
    target_arch=os.environ.get('TARGET_ARCH') or None,
    version=os.path.join(SPECPATH, 'win_version.txt') if sys.platform == 'win32' and
    os.path.exists(os.path.join(SPECPATH, 'win_version.txt')) else None,
)
coll = COLLECT(exe, a.binaries, a.datas, name='SC-Analysis')

if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='SC Analysis.app',
        icon=icon,
        bundle_identifier='hu.pte.volgyilab.scanalysis',
        version=VERSION,
        info_plist={
            'CFBundleName': 'SC Analysis',
            'CFBundleDisplayName': 'SC Analysis',
            'CFBundleShortVersionString': VERSION,
            'CFBundleVersion': VERSION,
            'NSHighResolutionCapable': True,
            'LSMinimumSystemVersion': '11.0',
            'NSHumanReadableCopyright': 'MIT licence, Gergely Szarka',
        },
    )

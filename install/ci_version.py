"""Decide the build version and stamp it into the app (GitHub Actions + local use).

Order: --version argument, a pushed tag vX.Y.Z (GITHUB_REF), the "version" box of
Actions -> Run workflow (INPUT_VERSION), else the current sc_analysis/_version.py
plus "-test" (build-and-test-only runs). Writes sc_analysis/_version.py and
install/win_version.txt, and 'version=…' / 'release=true|false' to $GITHUB_OUTPUT.
"""
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
VFILE = os.path.join(ROOT, 'sc_analysis', '_version.py')
PAT = re.compile(r'^\d+\.\d+\.\d+([.-][0-9A-Za-z.]+)?$')


def current():
    with open(VFILE) as f:
        return re.search(r"__version__\s*=\s*'([^']+)'", f.read()).group(1)


def decide(argv):
    if len(argv) > 1 and argv[1].strip():
        return argv[1].strip().lstrip('vV'), False
    ref = os.environ.get('GITHUB_REF', '')
    if ref.startswith('refs/tags/'):
        return ref[len('refs/tags/'):].lstrip('vV'), True
    inp = os.environ.get('INPUT_VERSION', '').strip()
    if inp:
        return inp.lstrip('vV'), True
    base = current().split('-')[0]
    return base + '-test', False


def main(argv):
    version, release = decide(argv)
    if not PAT.match(version):
        sys.exit('Bad version {!r}: use numbers like 1.2.0 (tag v1.2.0)'.format(version))
    with open(VFILE, 'w') as f:
        f.write("# Overwritten by install/ci_version.py from the git tag when GitHub builds a release.\n"
                "__version__ = '{}'\n".format(version))
    nums = [int(x) for x in re.findall(r'\d+', version)[:3]] + [0]
    t = '({}, {}, {}, 0)'.format(*nums[:3])
    with open(os.path.join(ROOT, 'install', 'win_version.txt'), 'w', encoding='utf-8') as f:
        f.write("""VSVersionInfo(
  ffi=FixedFileInfo(filevers={t}, prodvers={t}),
  kids=[
    StringFileInfo([StringTable('040904B0', [
      StringStruct('CompanyName', 'Volgyi lab, University of Pecs'),
      StringStruct('FileDescription', 'SC Analysis'),
      StringStruct('FileVersion', '{v}'),
      StringStruct('InternalName', 'SC-Analysis'),
      StringStruct('LegalCopyright', 'MIT licence, Gergely Szarka'),
      StringStruct('OriginalFilename', 'SC-Analysis.exe'),
      StringStruct('ProductName', 'SC Analysis'),
      StringStruct('ProductVersion', '{v}')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
""".format(t=t, v=version))
    out = os.environ.get('GITHUB_OUTPUT')
    if out:
        with open(out, 'a') as f:
            f.write('version={}\nrelease={}\n'.format(version, 'true' if release else 'false'))
    print('version', version, '| release' if release else '| build + test only')


if __name__ == '__main__':
    main(sys.argv)

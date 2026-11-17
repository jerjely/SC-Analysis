"""
File I/O: SpikeGLX meta files, OBX analog files, Kilosort output, and the
concatenated recording binary used by the trace viewer.
"""
import ast
import glob
import os

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# SpikeGLX meta
# ---------------------------------------------------------------------------
def read_meta(path):
    meta = {}
    with open(path) as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=', 1)
                meta[k.lstrip('~')] = v
    return meta


def find_ap_meta(folder):
    m = glob.glob(os.path.join(folder, '*_t0.imec0.ap.meta'))
    if not m:
        m = glob.glob(os.path.join(folder, '*.ap.meta'))
    if not m:
        m = glob.glob(os.path.join(folder, '**', '*.ap.meta'), recursive=True)
    return m[0] if m else None


def discover_recording_order(cfg, log=print):
    """Cell 1: order segments by fileCreateTime of their .ap.meta."""
    folders = glob.glob(os.path.join(cfg.npx_root, '{}_*_g0'.format(cfg.prefix)))
    seg_times = {}
    for folder in folders:
        seg = os.path.basename(folder).replace(cfg.prefix + '_', '').replace('_g0', '')
        meta = find_ap_meta(folder)
        if not meta:
            continue
        m = read_meta(meta)
        if 'fileCreateTime' in m:
            seg_times[seg] = m['fileCreateTime']
    order = [s for s, _ in sorted(seg_times.items(), key=lambda x: x[1])]
    log('Auto-detected order: {}'.format(order))
    return order


def get_duration_from_meta(folder):
    meta_path = find_ap_meta(folder)
    if not meta_path:
        raise FileNotFoundError('No .ap.meta file in {}'.format(folder))
    meta = read_meta(meta_path)
    return float(meta['fileTimeSecs']), float(meta['imSampRate'])


# ---------------------------------------------------------------------------
# OBX
# ---------------------------------------------------------------------------
def get_obx_sample_rate(obx_bin_path, fallback=30303.0, log=print):
    meta_path = obx_bin_path.replace('.obx.bin', '.obx.meta')
    if not os.path.exists(meta_path):
        cands = glob.glob(os.path.join(os.path.dirname(obx_bin_path), '*.obx.meta'))
        if not cands:
            log('  [WARN] no .obx.meta for {}, using fallback {} Hz'.format(obx_bin_path, fallback))
            return fallback
        meta_path = cands[0]
    meta = read_meta(meta_path)
    if 'obSampRate' in meta:
        return float(meta['obSampRate'])
    log('  [WARN] obSampRate not in {}, using fallback {} Hz'.format(meta_path, fallback))
    return fallback


def discover_obx_files(cfg, order, log=print):
    folders = glob.glob(os.path.join(cfg.npx_root, '{}_*_g0'.format(cfg.prefix)))
    folder_map = {}
    for folder in folders:
        seg = os.path.basename(folder).replace(cfg.prefix + '_', '').replace('_g0', '')
        obxs = glob.glob(os.path.join(folder, '*.obx.bin'))
        if not obxs:
            obxs = glob.glob(os.path.join(folder, '**', '*.obx.bin'), recursive=True)
        if obxs:
            folder_map[seg] = obxs[0]
    result = {n: folder_map[n] for n in order if n in folder_map}
    missing = [n for n in order if n not in folder_map]
    log('Discovered OBX files:')
    for name, path in result.items():
        log('  {:6s} -> {}'.format(name, path))
    if missing:
        log('  WARNING: missing: {}'.format(missing))
    return result


def obx_memmap(path, n_chans=14):
    raw = np.memmap(path, dtype=np.int16, mode='r')
    n = raw.size - (raw.size % n_chans)
    return raw[:n].reshape(-1, n_chans)


def obx_channel_volts(path, ch, n_chans=14):
    raw = obx_memmap(path, n_chans)
    return raw[:, ch].astype(np.float32) * (5.0 / 32768)


# ---------------------------------------------------------------------------
# Kilosort
# ---------------------------------------------------------------------------
def find_sorter_dir(folder, max_depth=4):
    """Return the folder that holds spike_times.npy: the folder itself or the
    best match below it (prefers ".../sorter_output", then the newest)."""
    if not folder or not os.path.isdir(folder):
        return None
    if os.path.exists(os.path.join(folder, 'spike_times.npy')):
        return folder
    hits = []
    base = folder.rstrip('\\/').count(os.sep)
    for root, dirs, files in os.walk(folder):
        if root.count(os.sep) - base >= max_depth:
            dirs[:] = []
        if 'spike_times.npy' in files:
            hits.append(root)
    if not hits:
        return None
    hits.sort(key=lambda h: (os.path.basename(h).lower() != 'sorter_output',
                             -os.path.getmtime(os.path.join(h, 'spike_times.npy'))))
    return hits[0]


def read_params_py(sorter_dir):
    p = os.path.join(sorter_dir, 'params.py')
    out = {}
    if not os.path.exists(p):
        return out
    with open(p) as f:
        for line in f:
            if '=' not in line:
                continue
            k, v = line.split('=', 1)
            k = k.strip()
            v = v.strip()
            try:
                out[k] = ast.literal_eval(v)
            except Exception:
                out[k] = v.strip('\'"')
    return out


def load_cluster_labels(sorter_dir):
    """cluster_group.tsv (Phy) preferred, then cluster_KSLabel.tsv."""
    for name, col in [('cluster_group.tsv', 'group'), ('cluster_KSLabel.tsv', 'KSLabel')]:
        p = os.path.join(sorter_dir, name)
        if os.path.exists(p):
            df = pd.read_csv(p, sep='\t')
            if col not in df.columns:
                col = df.columns[1]
            df = df.rename(columns={col: 'group'})
            return df[['cluster_id', 'group']], name
    raise FileNotFoundError('No cluster_group.tsv / cluster_KSLabel.tsv in {}'.format(sorter_dir))


def shank_from_x(xpos, gap_um=100.0):
    """Assign shank index to each channel x-position (gap > gap_um = new shank)."""
    xpos = np.asarray(xpos, dtype=float)
    ux = np.unique(np.round(xpos, 1))
    shank_of = {}
    s = 0
    for i, x in enumerate(ux):
        if i > 0 and x - ux[i - 1] > gap_um:
            s += 1
        shank_of[x] = s
    return np.array([shank_of[x] for x in np.round(xpos, 1)], dtype=int)


# ---------------------------------------------------------------------------
# concatenated recording binary (trace viewer)
# ---------------------------------------------------------------------------
class RecordingBinary(object):
    """memmap view of a flat int16 binary (SpikeGLX .ap.bin or KS .dat)."""

    def __init__(self, path, n_channels, fs, dtype='int16', uv_per_bit=None):
        self.path = path
        self.n_channels = int(n_channels)
        self.fs = float(fs)
        raw = np.memmap(path, dtype=np.dtype(dtype), mode='r')
        n = raw.size - (raw.size % self.n_channels)
        self.data = raw[:n].reshape(-1, self.n_channels)
        self.n_samples = self.data.shape[0]
        self.duration = self.n_samples / self.fs
        self.uv_per_bit = uv_per_bit

    def chunk(self, t0, t1, rows=None):
        i0 = max(0, int(t0 * self.fs))
        i1 = min(self.n_samples, int(t1 * self.fs))
        if i1 <= i0:
            return np.zeros((0, 0), dtype=np.float32), i0
        d = self.data[i0:i1]
        if rows is not None:
            d = d[:, rows]
        return np.asarray(d, dtype=np.float32), i0


def open_recording_binary(cfg, log=print):
    """Find the concatenated binary for the viewer.

    Order: cfg.recording_bin  ->  sorter params.py dat_path (abs or relative)
           -> any *.ap.bin / *.dat in the sorter dir or its parents.
    SpikeGLX .meta next to the file gives channel count / rate / gain."""
    sd = find_sorter_dir(cfg.sorter_dir) or cfg.sorter_dir
    params = read_params_py(sd) if sd else {}
    cands = []
    if cfg.recording_bin:
        cands.append(cfg.recording_bin)
    dp = params.get('dat_path')
    if isinstance(dp, (list, tuple)):
        dp = dp[0] if dp else None
    if dp:
        cands.append(dp)
        cands.append(os.path.join(sd, dp))
        cands.append(os.path.join(sd, os.path.basename(str(dp))))
        cands.append(os.path.join(os.path.dirname(sd), os.path.basename(str(dp))))
    for d in [sd, os.path.dirname(sd or ''),
              os.path.dirname(os.path.dirname(sd or ''))]:
        if d and os.path.isdir(d):
            cands += sorted(glob.glob(os.path.join(d, '*.ap.bin')))
            cands += sorted(glob.glob(os.path.join(d, '*.dat')))
    path = None
    for c in cands:
        if c and os.path.isfile(c):
            path = c
            break
    if path is None:
        raise FileNotFoundError('Concatenated recording binary not found. Set it in '
                                'Session settings > Paths (tried: {})'.format(
                                    ', '.join(str(c) for c in cands[:4])))
    n_ch = params.get('n_channels_dat')
    fs = params.get('sample_rate', 30000.0)
    dtype = params.get('dtype', 'int16')
    uv = None
    meta_path = path[:-4] + '.meta' if path.lower().endswith('.bin') else None
    if meta_path and os.path.exists(meta_path):
        m = read_meta(meta_path)
        n_ch = int(m.get('nSavedChans', n_ch or 385))
        fs = float(m.get('imSampRate', fs))
        try:
            rng = float(m.get('imAiRangeMax', 0.6))
            maxint = float(m.get('imMaxInt', 512))
            # NP2.0 gain 80, NP1.0 typical 500 AP gain
            gain = 80.0 if m.get('imDatPrb_type', '0') in ('21', '24', '2003', '2004', '2013', '2014') else 500.0
            uv = rng / maxint / gain * 1e6
        except Exception:
            uv = None
    if n_ch is None:
        n_ch = 385
    log('Recording binary: {}  ({} ch, {} Hz)'.format(path, n_ch, fs))
    return RecordingBinary(path, n_ch, fs, dtype, uv)

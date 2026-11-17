"""
Session configuration: every setting that lived in the CONFIG cell (and the
scattered per-cell constants) of SC_analysis_v24.ipynb, in one place.

Saved as  <SAVE_DIR>/sc_session.json  so a session can be reopened later.
"""
import copy
import json
import os
import re


def split_identity(name):
    """'2026.08.10_Thy1GCaMP3_WT_SC_concat' -> ('2026.08.10', 'Thy1GCaMP3_WT', 'SC')."""
    base = str(name).replace('\\', '/').rstrip('/').split('/')[-1]
    for suf in ('_concat',):
        if base.lower().endswith(suf):
            base = base[:-len(suf)]
    m = re.match(r'(\d{4}\.\d{2}\.\d{2})_(.+)_([A-Za-z0-9]+)$', base)
    return m.groups() if m else None


def _unique_match(root, suffix, default):
    """Fallback: the single folder in root whose name ends with suffix."""
    try:
        hits = [d for d in os.listdir(root) if d.endswith(suffix) and os.path.isdir(os.path.join(root, d))]
    except Exception:
        return default
    return os.path.join(root, hits[0]) if len(hits) == 1 else default


# ---------------------------------------------------------------------------
# defaults (values copied from the v24 notebook)
# ---------------------------------------------------------------------------
DEFAULTS = {
    # ---- identity ----
    'recording_date':   '2026.08.10',
    'mouse_line':       'Thy1GCaMP3_WT',
    'recording_region': 'SC',

    # ---- paths ----
    'npx_root':      r'D:\NPX',          # raw SpikeGLX segment folders  <prefix>_<SEG>_g0
    'save_dir':      '',                 # analysis output folder (…_concat)
    'sorter_dir':    '',                 # Kilosort4 sorter_output
    'psychopy_root': '',                 # PsychoPy folders  <prefix>_<SEG>
    'frames_path':   '',                 # sparse_object_frames.npy
    'recording_bin': '',                 # concatenated .ap.bin / .dat for the trace viewer
                                         # ('' = take dat_path from sorter params.py)

    # ---- segments ----
    'recording_order': [],               # [] = auto-detect from fileCreateTime in .ap.meta
    'rfm_good':        ['RFM1', 'RFM2'],

    # ---- screen / stimulus geometry ----
    'screen_width_cm':  42.5,
    'screen_dist_cm':   15.0,
    'screen_px_w':      1600,
    'screen_px_h':      900,
    'sweep_azi_deg':    15.0,
    'travel_width_deg': 100.0,
    'circle_diam_deg':  5.0,

    # ---- general analysis ----
    'bin_s':   0.01,
    'sigma':   1.5,
    'grating_csv_subsample': 1,          # 2 = old recordings, 1 = new
    'depth_cutoff': None,                # None = 3000

    # ---- OBX / TTL extraction (cell 0B) ----
    'obx_n_chans':     14,
    'ttl_analog_ch':   4,
    'ttl_threshold_v': 1.0,
    'ttl_min_gap_s':   0.05,
    'obx_fallback_fs': 30303.0,
    # per-segment overrides; threshold_v may be 'dynamic'
    'segment_ttl_overrides': {'B1': {'analog_ch': 0, 'threshold_v': 'dynamic'}},

    # ---- B1 (Baden battery) protocol constants ----
    'b1_segment':        'B1',
    'b1_n_repeats':      5,
    'b1_inner_reps':     2,
    'b1_bar_directions': [0, 90, 180, 270],
    'b1_color_names':    ['blue', 'green'],
    'b1_bar_dur':        3.85,
    'b1_inter_repeat':   4.0,

    # ---- PsychoPy parsing (cell 0C) ----
    'n_frames_rfm':        1000,
    'loom_azi_offset_deg': -20.0,
    'small_sweep_durations': {
        'small_sweep_10dps': 10.0, 'small_sweep_20dps': 5.0,
        'small_sweep_25dps': 4.0,  'small_sweep_30dps': 3.333,
        'small_sweep_40dps': 2.5,  'small_sweep_50dps': 2.0,
    },

    # ---- photodiode correction (P1) ----
    'pd_segment':          'P1',
    'pd_analog_ch':        0,
    'pd_ttl_ch':           4,
    'pd_search_window_s':  0.25,
    'pd_baseline_window_s': 0.2,
    'pd_amp_threshold_mult': 1.5,
    'pd_distance_ms':      8.0,

    # ---- STA / RF (cells 7, 8, 1B) ----
    'sta_ds':          10,
    'sta_lags_s':      [-0.250, -0.200, -0.150, -0.125, -0.100, -0.075, -0.050, -0.025],
    'sta_min_spikes':  100,
    'rf_search_radius': 20.0,
    'rf_mode':         'consistent',     # RFM1 | RFM2 | combined | consistent
    'max_shift_deg':   10.0,
    'min_snr':         6.0,
    'max_rf_diam':     30.0,

    # ---- Baden features / QI / clustering ----
    'bin_s_feature':     0.02,
    'smooth_sigma_bins': 2,
    'qi_threshold':      0.2,
    'bar_speed_deg':     40.0,
    'screen_diag_deg':   134.0,
    'y_sign_flip':       True,
    'x_sign_flip':       False,
    'n_permutations':    1000,
    'ds_min_total_spikes': 20,
    'ds_min_peak_rate':    2.0,
    'color_min_total_spikes': 10,
    'color_min_peak_resp':    1.0,
    'n_sparse_components': 8,
    'max_clusters':        30,
    'gmm_n_init_bic':      10,
    'gmm_n_init_final':    20,

    # ---- sweep picker ----
    'sigma_per_speed': {
        'small_sweep_10dps': 6.0, 'small_sweep_20dps': 4.0,
        'small_sweep_25dps': 2.5, 'small_sweep_30dps': 3.0,
        'small_sweep_40dps': 1.5, 'small_sweep_50dps': 1.5,
    },
    # ---- blink / loom picker ----
    'blink_loom_conditions': [
        ['white_blink',       'White Blink',        'goldenrod'],
        ['black_blink',       'Black Blink',        'dimgray'],
        ['white_loom_azi+0',  'White Loom Azi +0',  'steelblue'],
        ['white_loom_azi+20', 'White Loom Azi +20', 'dodgerblue'],
        ['black_loom_azi+0',  'Black Loom Azi +0',  'firebrick'],
        ['black_loom_azi+20', 'Black Loom Azi +20', 'darkred'],
    ],
    'sigma_blink': 2.0,
    'sigma_loom':  3.0,

    # ---- CCF ----
    'ccf_segment':   'S1',
    'ccf_max_lag_s': 0.050,
    'ccf_bin_s':     0.001,
    'ccf_sig_z':     5.0,
    'ccf_min_fr':    1.0,
    'ccf_peak_win_ms': 10.0,

    # ---- pupil ----
    'pupil_segments':   ['G1', 'G2'],
    'pupil_crop':       [1280, 2880, 0, 900],    # x1, x2, y1, y2 in the screen video
    'pupil_like_thresh': 0.5,
    'pupil_cam_offset_s': -0.272,
    'cam_deg_per_px':   0.4454,
    'pupil_bodyparts':  ['Dorsal', 'Ventral', 'Nasal', 'Temporal',
                         'DorsoNasal', 'VentroNasal', 'DorsoTemporal', 'VentroTemporal'],
}

# grating parameters from Cell 1 (degrees -> sf, tf)
GRATING_TABLE = [
    (0.3, 1.666667, 10.0), (0.6, 0.833333, 5.0), (0.9, 0.555556, 3.333),
    (1.2, 0.416667, 2.5), (1.5, 0.333333, 2.0), (2.0, 0.250000, 1.5),
    (2.5, 0.200000, 1.2), (3.0, 0.166667, 1.0), (3.5, 0.142857, 0.857),
    (4.0, 0.125000, 0.75), (4.5, 0.111111, 0.667), (5.0, 0.100000, 0.6),
]

SPEED_COLORS = {
    'small_sweep_10dps': '#1f77b4', 'small_sweep_20dps': '#ff7f0e',
    'small_sweep_25dps': '#2ca02c', 'small_sweep_30dps': '#d62728',
    'small_sweep_40dps': '#9467bd', 'small_sweep_50dps': '#8c564b',
}

SESSION_FILENAME = 'sc_session.json'


class SessionConfig(object):
    """Dict-like config with attribute access and JSON persistence."""

    def __init__(self, values=None):
        self._v = copy.deepcopy(DEFAULTS)
        if values:
            for k, v in values.items():
                self._v[k] = v

    # attribute / item access ---------------------------------------------
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        try:
            return self._v[name]
        except KeyError:
            raise AttributeError(name)

    def __getitem__(self, k):
        return self._v[k]

    def __setitem__(self, k, v):
        self._v[k] = v

    def get(self, k, default=None):
        return self._v.get(k, default)

    def as_dict(self):
        return copy.deepcopy(self._v)

    def update(self, d):
        for k, v in d.items():
            self._v[k] = v

    # derived ----------------------------------------------------------------
    @property
    def prefix(self):
        return '{}_{}_{}'.format(self.recording_date, self.mouse_line,
                                 self.recording_region)

    @property
    def session_id(self):
        return self.prefix

    def seg_npx_folder(self, seg):
        want = os.path.join(self.npx_root, '{}_{}_g0'.format(self.prefix, seg))
        return want if os.path.isdir(want) else _unique_match(self.npx_root, '_{}_g0'.format(seg), want)

    def seg_psychopy_folder(self, seg):
        want = os.path.join(self.psychopy_root, '{}_{}'.format(self.prefix, seg))
        return want if os.path.isdir(want) else _unique_match(self.psychopy_root, '_{}'.format(seg), want)

    @property
    def speed_dps(self):
        out = {}
        for cond in self.small_sweep_durations:
            try:
                out[cond] = int(cond.split('_')[-1].replace('dps', ''))
            except ValueError:
                pass
        return out

    @property
    def grating_params(self):
        gp = {}
        for deg, sf, tf in GRATING_TABLE:
            gp[deg] = {'sf': sf, 'tf': tf, 'speed_dps': tf / sf,
                       'period_s': 1.0 / tf, 'bar_width_deg': 1.0 / sf}
        return gp

    # persistence ------------------------------------------------------------
    def path(self):
        return os.path.join(self.save_dir, SESSION_FILENAME) if self.save_dir else ''

    def save(self, path=None):
        path = path or self.path()
        if not path:
            raise ValueError('No save_dir set')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self._v, f, indent=2)
        return path

    @classmethod
    def load(cls, path):
        with open(path) as f:
            vals = json.load(f)
        cfg = cls(vals)
        if cfg.fix_identity():
            try:
                cfg.save(path)
            except Exception:
                pass
        return cfg

    def fix_identity(self):
        """Repair configs where the folder suffix '_concat' was taken as the region
        (v1.0/1.1 guess bug). Returns True if anything changed."""
        reg = str(self.get('recording_region', ''))
        line = str(self.get('mouse_line', ''))
        if reg.lower() == 'concat' and '_' in line:
            line, reg = line.rsplit('_', 1)
            self['mouse_line'], self['recording_region'] = line, reg
            return True
        return False


# ---------------------------------------------------------------------------
# grouped field descriptions for the settings dialog
# (key, label, kind)  kind: path_dir | path_file | str | int | float | bool | list | json
# ---------------------------------------------------------------------------
FIELD_GROUPS = [
    ('Recording', [
        ('recording_date', 'Recording date', 'str'),
        ('mouse_line', 'Mouse line', 'str'),
        ('recording_region', 'Region', 'str'),
        ('recording_order', 'Segment order ([] = auto)', 'list'),
        ('rfm_good', 'Usable RFM segments', 'list'),
    ]),
    ('Paths', [
        ('save_dir', 'Analysis folder (SAVE_DIR)', 'path_dir'),
        ('sorter_dir', 'Kilosort sorter_output', 'path_dir'),
        ('recording_bin', 'Concatenated binary (blank = params.py)', 'path_file'),
        ('npx_root', 'Raw NPX root (segment folders)', 'path_dir'),
        ('psychopy_root', 'PsychoPy root', 'path_dir'),
        ('frames_path', 'Sparse-noise frames .npy', 'path_file'),
    ]),
    ('Screen & stimulus', [
        ('screen_width_cm', 'Screen width (cm)', 'float'),
        ('screen_dist_cm', 'Screen distance (cm)', 'float'),
        ('screen_px_w', 'Screen width (px)', 'int'),
        ('screen_px_h', 'Screen height (px)', 'int'),
        ('sweep_azi_deg', 'Sweep azimuth (deg)', 'float'),
        ('travel_width_deg', 'Sweep travel width (deg)', 'float'),
        ('circle_diam_deg', 'Sweep circle diameter (deg)', 'float'),
        ('loom_azi_offset_deg', 'Offset-loom azimuth shift (deg)', 'float'),
        ('grating_csv_subsample', 'Grating CSV subsample', 'int'),
        ('n_frames_rfm', 'Sparse-noise frames per repeat', 'int'),
        ('small_sweep_durations', 'Small-sweep durations (s)', 'json'),
    ]),
    ('Sync / TTL', [
        ('obx_n_chans', 'OBX channels', 'int'),
        ('ttl_analog_ch', 'TTL analog channel', 'int'),
        ('ttl_threshold_v', 'TTL threshold (V)', 'float'),
        ('ttl_min_gap_s', 'Min edge gap (s)', 'float'),
        ('segment_ttl_overrides', 'Per-segment TTL overrides', 'json'),
        ('pd_segment', 'PD-correct segment', 'str'),
        ('pd_analog_ch', 'Photodiode channel', 'int'),
        ('pd_amp_threshold_mult', 'PD p2p threshold x', 'float'),
    ]),
    ('B1 protocol', [
        ('b1_segment', 'B1 segment name', 'str'),
        ('b1_n_repeats', 'Repeats', 'int'),
        ('b1_inner_reps', 'Inner reps (short)', 'int'),
        ('b1_bar_directions', 'Bar directions', 'list'),
        ('b1_color_names', 'Colour names', 'list'),
        ('b1_bar_dur', 'Bar duration (s)', 'float'),
    ]),
    ('RF', [
        ('sta_ds', 'STA downsample', 'int'),
        ('sta_lags_s', 'STA lags (s)', 'list'),
        ('sta_min_spikes', 'Min spikes for STA', 'int'),
        ('rf_search_radius', 'Fit search radius (deg)', 'float'),
        ('rf_mode', 'RF mode', 'str'),
        ('max_shift_deg', 'Max RFM1-RFM2 shift (deg)', 'float'),
        ('min_snr', 'Min SNR', 'float'),
        ('max_rf_diam', 'Max RF diameter (deg)', 'float'),
    ]),
    ('Analysis', [
        ('bin_s', 'PSTH bin (s)', 'float'),
        ('sigma', 'PSTH smoothing (bins)', 'float'),
        ('depth_cutoff', 'Retinotopy depth cutoff (µm)', 'json'),
        ('bin_s_feature', 'Feature bin (s)', 'float'),
        ('smooth_sigma_bins', 'Feature smoothing (bins)', 'float'),
        ('qi_threshold', 'QI threshold', 'float'),
        ('y_sign_flip', 'Bar Y sign flip', 'bool'),
        ('x_sign_flip', 'Bar X sign flip', 'bool'),
        ('n_permutations', 'Permutations', 'int'),
        ('n_sparse_components', 'SparsePCA components', 'int'),
        ('max_clusters', 'Max clusters', 'int'),
        ('sigma_per_speed', 'Sweep smoothing per speed', 'json'),
        ('blink_loom_conditions', 'Blink/loom picker conditions', 'json'),
    ]),
    ('CCF & pupil', [
        ('ccf_segment', 'CCF segment', 'str'),
        ('ccf_sig_z', 'CCF z threshold', 'float'),
        ('ccf_min_fr', 'CCF min FR (Hz)', 'float'),
        ('pupil_segments', 'Pupil segments', 'list'),
        ('pupil_crop', 'Screen-video crop x1,x2,y1,y2', 'list'),
        ('pupil_cam_offset_s', 'Camera offset (s)', 'float'),
        ('cam_deg_per_px', 'Eye camera deg/px', 'float'),
    ]),
]

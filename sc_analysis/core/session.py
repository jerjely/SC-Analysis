"""
Session: holds the configuration and everything Cell 1 ("Load Everything")
used to put in the notebook namespace, loaded lazily from SAVE_DIR / SORTER_DIR.
"""
import json
import os

import numpy as np
import pandas as pd

from . import npx_io, sync, rf as rfmod
from .config import SessionConfig, SPEED_COLORS


class MissingStep(Exception):
    """Raised when an analysis needs a pipeline step that has not been run."""


# pipeline step definitions: key, label, output files that mark it as done
STEPS = [
    ('boundaries', '0A  Segment boundaries', ['segment_boundaries.json']),
    ('ttl', '0B  TTL / photodiode edges', ['stim_onsets_concat_s.npy']),
    ('b1', 'B1  Label Baden events', ['B1_photodiode_events_labeled.csv']),
    ('stimuli', '0C  Parse PsychoPy', ['all_condition_onsets.npy', 'rfm_data.npy']),
    ('pd', 'PD  Photodiode-correct P1', ['{pd}_pd_corrected_onsets_s.npy']),
    ('sta', '7   STAs', ['all_sta_hires_{rfm0}.npy']),
    ('rffit', '8   RF fits', ['rf_results_hires_{rfm0}.npy']),
    ('rfmode', '1B  RF mode -> clean RFs', ['rf_results_clean.npy']),
]


class Session(object):
    def __init__(self, cfg=None):
        self.cfg = cfg or SessionConfig()
        self.reset()

    def reset(self):
        self._spikes = None
        self._cond = None
        self._bound = None
        self._seg_onsets = None
        self._rf = None
        self._df_b1 = None
        self._pd = None
        self._rec = None
        self._obx_files = None
        self.cache = {}
        self.results = {}       # computed analysis results (QI, features, clustering, ccf, pupil …)

    def invalidate(self, *what):
        for w in what:
            setattr(self, '_' + w, None)
        self.cache = {}

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------
    def p(self, *names):
        return os.path.join(self.cfg.save_dir, *names)

    def step_files(self, key):
        rfm0 = (list(self.cfg.rfm_good) or ['RFM1'])[0]
        for k, _, files in STEPS:
            if k == key:
                return [f.format(pd=self.cfg.pd_segment, rfm0=rfm0) for f in files]
        return []

    def step_done(self, key):
        if not self.cfg.save_dir:
            return False
        return all(os.path.exists(self.p(f)) for f in self.step_files(key))

    def require(self, key):
        if not self.step_done(key):
            label = [l for k, l, _ in STEPS if k == key][0]
            raise MissingStep('Run pipeline step "{}" first.'.format(label.split('  ', 1)[-1]))

    @property
    def geom(self):
        return rfmod.ScreenGeom(self.cfg)

    @property
    def order(self):
        if self.cfg.recording_order:
            return list(self.cfg.recording_order)
        b = self.boundaries_or_none()
        if b:
            return sorted(b, key=lambda s: b[s]['start_s'])
        return npx_io.discover_recording_order(self.cfg, log=lambda *a: None)

    # ------------------------------------------------------------------
    # boundaries / onsets
    # ------------------------------------------------------------------
    def boundaries_or_none(self):
        try:
            return self.boundaries
        except Exception:
            return None

    @property
    def boundaries(self):
        if self._bound is None:
            self.require('boundaries')
            self._bound = sync.load_boundaries(self.cfg.save_dir)
        return self._bound[0]

    @property
    def fs_imec(self):
        self.boundaries
        return self._bound[1]

    @property
    def seg_onsets(self):
        """all_onsets_concat: raw TTL onsets per segment (from stim_onsets_concat_s.npy)."""
        if self._seg_onsets is None:
            self.require('ttl')
            self._seg_onsets = sync.onsets_by_segment_from_flat(self.cfg.save_dir, self.boundaries)
        return self._seg_onsets[0]

    @property
    def stim_onsets_flat(self):
        self.seg_onsets
        return self._seg_onsets[1]

    @property
    def obx_files(self):
        if self._obx_files is None:
            p = self.p('onsets_by_segment.npy')
            files = {}
            if os.path.exists(p):
                files = np.load(p, allow_pickle=True).item().get('obx_files', {})
                files = {k: v for k, v in files.items() if os.path.exists(v)}
            if not files:
                files = npx_io.discover_obx_files(self.cfg, self.order, log=lambda *a: None)
            self._obx_files = files
        return self._obx_files

    # ------------------------------------------------------------------
    # conditions
    # ------------------------------------------------------------------
    def _load_cond(self):
        if self._cond is None:
            self.require('stimuli')
            on = np.load(self.p('all_condition_onsets.npy'), allow_pickle=True).item()
            off = np.load(self.p('all_condition_offsets.npy'), allow_pickle=True).item()
            rd = np.load(self.p('rfm_data.npy'), allow_pickle=True).item()
            self._cond = (on, off, rd)
        return self._cond

    @property
    def cond_onsets(self):
        return self._load_cond()[0]

    @property
    def cond_offsets(self):
        return self._load_cond()[1]

    @property
    def rfm_data(self):
        return self._load_cond()[2]

    def conditions(self, seg='P1'):
        try:
            return sorted(self.cond_onsets.get(seg, {}).keys())
        except Exception:
            return []

    def onsets(self, seg, cond, pd_corrected=False):
        on = np.asarray(self.cond_onsets[seg][cond], dtype=float)
        off = np.asarray(self.cond_offsets[seg][cond], dtype=float)
        if pd_corrected and seg == self.cfg.pd_segment and self.pd_onsets is not None:
            on = self.pd_correct(on, seg)
        return on, off

    @property
    def pd_onsets(self):
        if self._pd is None:
            p = self.p('{}_pd_corrected_onsets_s.npy'.format(self.cfg.pd_segment))
            self._pd = np.load(p) if os.path.exists(p) else False
        return None if self._pd is False else self._pd

    def pd_correct(self, raw_onsets, seg='P1'):
        """Cell 15 get_pd_corrected_onsets, NaN-safe (falls back to raw where PD missing)."""
        pdo = self.pd_onsets
        base = self.seg_onsets[seg]
        if pdo is None or len(base) == 0:
            return raw_onsets
        idx = np.array([np.argmin(np.abs(base - c)) for c in raw_onsets])
        delta = pdo[idx] - base[idx]
        delta = np.where(np.isfinite(delta), delta, 0.0)
        return raw_onsets + delta

    @property
    def df_b1(self):
        if self._df_b1 is None:
            self.require('b1')
            self._df_b1 = pd.read_csv(self.p('B1_photodiode_events_labeled.csv'))
        return self._df_b1

    # ------------------------------------------------------------------
    # spikes (Cell 1)
    # ------------------------------------------------------------------
    def _load_spikes(self):
        if self._spikes is not None:
            return self._spikes
        sd = npx_io.find_sorter_dir(self.cfg.sorter_dir)
        if sd is None:
            raise MissingStep('No spike_times.npy found in or below the Kilosort folder\n{!r}.\n'
                              'Set "Kilosort sorter_output" in Session settings > Paths.'.format(self.cfg.sorter_dir))
        if os.path.normpath(sd) != os.path.normpath(self.cfg.sorter_dir):
            self.cfg['sorter_dir'] = sd
            try:
                if self.cfg.save_dir:
                    self.cfg.save()
            except Exception:
                pass
        fs = self.fs_imec if self.step_done('boundaries') else \
            float(npx_io.read_params_py(sd).get('sample_rate', 30000.0))
        st = np.load(os.path.join(sd, 'spike_times.npy')).squeeze() / fs
        sc = np.load(os.path.join(sd, 'spike_clusters.npy')).squeeze()
        stp = os.path.join(sd, 'spike_templates.npy')
        stt = np.load(stp).squeeze() if os.path.exists(stp) else sc.copy()
        labels_df, label_src = npx_io.load_cluster_labels(sd)
        good_ids = labels_df[labels_df['group'] == 'good']['cluster_id'].values
        order = np.argsort(sc, kind='stable')
        sc_sorted = sc[order]
        uniq, starts = np.unique(sc_sorted, return_index=True)
        ends = np.r_[starts[1:], len(sc_sorted)]
        trains, tmpl_of = {}, {}
        for u, a, b in zip(uniq, starts, ends):
            idx = order[a:b]
            trains[int(u)] = np.sort(st[idx])
            tm = stt[idx]
            tmpl_of[int(u)] = int(np.bincount(tm).argmax()) if len(tm) else -1
        chpos = np.load(os.path.join(sd, 'channel_positions.npy'))
        templates = np.load(os.path.join(sd, 'templates.npy'), mmap_mode='r')
        tamp = np.asarray(templates.max(axis=1) - templates.min(axis=1))
        best_ch = np.argmax(tamp, axis=1)
        depths, xpos, bestch = {}, {}, {}
        for u in trains:
            tid = tmpl_of[u]
            if tid < 0 or tid >= len(best_ch):
                depths[u] = xpos[u] = np.nan; bestch[u] = -1
                continue
            ch = best_ch[tid]
            depths[u] = float(chpos[ch, 1]); xpos[u] = float(chpos[ch, 0]); bestch[u] = int(ch)
        good_ids = np.array([g for g in good_ids if int(g) in trains], dtype=int)
        self._spikes = {
            'spike_times_s': st, 'spike_clusters': sc, 'trains_all': trains,
            'good_ids': good_ids, 'labels_df': labels_df, 'label_src': label_src,
            'channel_pos': chpos, 'templates': templates, 'best_ch_of_template': best_ch,
            'template_of_unit': tmpl_of, 'unit_depths': depths, 'unit_xpos': xpos,
            'unit_best_ch': bestch, 't_max': float(st.max()) if len(st) else 0.0,
        }
        return self._spikes

    @property
    def good_ids(self):
        return self._load_spikes()['good_ids']

    @property
    def spike_trains(self):
        return self._load_spikes()['trains_all']

    @property
    def unit_depths(self):
        return self._load_spikes()['unit_depths']

    @property
    def unit_xpos(self):
        return self._load_spikes()['unit_xpos']

    @property
    def spikes(self):
        return self._load_spikes()

    @property
    def good_ids_by_depth(self):
        d = self.unit_depths
        return sorted(self.good_ids, key=lambda c: d.get(c, np.nan)
                      if np.isfinite(d.get(c, np.nan)) else 1e9)

    def units_in_depth(self, dmin, dmax):
        d = self.unit_depths
        out = [(c, d[c]) for c in self.good_ids_by_depth
               if np.isfinite(d.get(c, np.nan)) and dmin <= d[c] <= dmax]
        return [c for c, _ in out], np.array([x for _, x in out])

    # ------------------------------------------------------------------
    # RF results
    # ------------------------------------------------------------------
    def _load_rf(self):
        if self._rf is None:
            good = list(self.cfg.rfm_good) or ['RFM1']
            r1 = rfmod.load_rf(self.cfg.save_dir, good[0])
            r2 = rfmod.load_rf(self.cfg.save_dir, good[1]) if len(good) > 1 else r1
            rc = rfmod.load_rf(self.cfg.save_dir, 'combined')

            def lsta(name):
                p = self.p('all_sta_hires_{}.npy'.format(name))
                return np.load(p, allow_pickle=True).item() if os.path.exists(p) else {}
            s1 = lsta(good[0])
            s2 = lsta(good[1]) if len(good) > 1 else s1
            p = self.p('rf_results_clean.npy')
            clean = np.load(p, allow_pickle=True).item() if os.path.exists(p) else {}
            self._rf = {'rfm1': r1, 'rfm2': r2, 'combined': rc, 'sta1': s1, 'sta2': s2,
                        'clean': clean}
        return self._rf

    @property
    def rf_clean(self):
        return self._load_rf()['clean']

    @property
    def rf_rfm1(self):
        return self._load_rf()['rfm1']

    @property
    def rf_rfm2(self):
        return self._load_rf()['rfm2']

    @property
    def rf_combined(self):
        return self._load_rf()['combined']

    @property
    def sta_rfm1(self):
        return self._load_rf()['sta1']

    @property
    def sta_rfm2(self):
        return self._load_rf()['sta2']

    # ------------------------------------------------------------------
    # sweep cells (end of Cell 1)
    # ------------------------------------------------------------------
    @property
    def sweep_geometry(self):
        c = self.cfg
        return {'sweep_start_deg': c.sweep_azi_deg - c.travel_width_deg / 2.0,
                'sweep_end_deg': c.sweep_azi_deg + c.travel_width_deg / 2.0,
                'circle_diam_deg': c.circle_diam_deg}

    @property
    def speed_dps(self):
        sp = self.cfg.speed_dps
        conds = set(self.conditions('P1'))
        return {k: v for k, v in sorted(sp.items(), key=lambda kv: kv[1]) if k in conds}

    @property
    def speed_colors(self):
        return SPEED_COLORS

    @property
    def sweep_cells(self):
        if 'sweep_cells' in self.cache:
            return self.cache['sweep_cells']
        from .psth import compute_psth
        from scipy.ndimage import gaussian_filter1d
        bin_s, sigma = self.cfg.bin_s, self.cfg.sigma
        ref = 'small_sweep_25dps'
        if ref not in self.conditions('P1'):
            cand = list(self.speed_dps)
            if not cand:
                self.cache['sweep_cells'] = []
                return []
            ref = cand[len(cand) // 2]
        on_ref, off_ref = self.onsets('P1', ref)
        dur = np.mean(off_ref - on_ref)
        spd = self.cfg.speed_dps.get(ref, 25)
        bins = np.arange(-0.5, dur + 1.5, bin_s)
        bc = bins[:-1] + bin_s / 2
        resp = (bc >= 0) & (bc < dur)
        out = []
        for cid in self.good_ids:
            m, _ = compute_psth(self.spike_trains[cid], on_ref, bins, bin_s)
            m = gaussian_filter1d(m, sigma=sigma)
            base = m[(bc >= -0.5) & (bc < 0)].mean()
            s = m.copy(); s[~resp] = 0
            pt = bc[np.argmax(s)]
            pf = m[resp].max()
            ev = pf - base
            if pf > 5.0 and ev > 2.0 and 0.0 < pt < dur - 0.3:
                out.append({'cid': int(cid), 'peak_t': pt, 'peak_fr': pf, 'base_fr': base,
                            'evoked_fr': ev, 'rf_pos_deg': pt * spd,
                            'depth': self.unit_depths.get(cid, np.nan)})
        out.sort(key=lambda x: x['peak_t'])
        self.cache['sweep_cells'] = out
        return out

    # ------------------------------------------------------------------
    # recording binary (viewer)
    # ------------------------------------------------------------------
    def recording(self, log=print):
        if self._rec is None:
            try:
                self._load_spikes()     # resolves sorter_dir first
            except Exception:
                pass
            self._rec = npx_io.open_recording_binary(self.cfg, log)
        return self._rec

    # ------------------------------------------------------------------
    # picks
    # ------------------------------------------------------------------
    def picks_df(self, which):
        """which = 'sweep' | 'blink_loom'  -> DataFrame or None."""
        name = {'sweep': 'interactive_picks', 'blink_loom': 'blink_loom_picks'}[which]
        csvp = self.p(name + '.csv')
        jsp = self.p(name + '.json')
        if os.path.exists(jsp):
            try:
                with open(jsp) as f:
                    d = json.load(f)
                if d:
                    df = pd.DataFrame(list(d.values()))
                    if which == 'sweep' and 'rf_left_deg' in df.columns:
                        df = df.sort_values('rf_left_deg').reset_index(drop=True)
                    return df
            except Exception:
                pass
        if os.path.exists(csvp):
            return pd.read_csv(csvp)
        return None

    def sweep_excluded(self):
        p = self.p('sweep_excluded_cids.json')
        if os.path.exists(p):
            with open(p) as f:
                return set(int(x) for x in json.load(f))
        return set()

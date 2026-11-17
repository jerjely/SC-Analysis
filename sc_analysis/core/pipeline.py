"""Run individual pipeline steps (used by the GUI Pipeline panel)."""
import os

import numpy as np

from . import npx_io, sync, stimuli, rf


def run_step(sess, key, overwrite=False, log=print, progress=None, cancel=None):
    cfg = sess.cfg
    if not cfg.save_dir:
        raise ValueError('Set the analysis folder (SAVE_DIR) in Session settings first')
    os.makedirs(cfg.save_dir, exist_ok=True)
    if key == 'boundaries':
        order = list(cfg.recording_order) or npx_io.discover_recording_order(cfg, log)
        if not order:
            raise ValueError('No segment folders "{}_*_g0" with .ap.meta found in {}'.format(cfg.prefix, cfg.npx_root))
        sync.compute_boundaries(cfg, order, log)
        sess.invalidate('bound', 'seg_onsets', 'spikes')
    elif key == 'ttl':
        b = sess.boundaries
        order = sess.order
        sync.extract_all_onsets(cfg, order, b, log)
        sess.invalidate('seg_onsets', 'obx_files')
    elif key == 'b1':
        seg = cfg.b1_segment
        on = sess.seg_onsets.get(seg)
        if on is None or not len(on):
            raise ValueError('No onsets for segment {}'.format(seg))
        sync.run_b1_labelling(cfg, on, log)
        sess.invalidate('df_b1')
    elif key == 'stimuli':
        stimuli.parse_all(cfg, sess.order, sess.seg_onsets, log)
        sess.invalidate('cond')
    elif key == 'pd':
        if cfg.pd_segment not in sess.obx_files:
            raise ValueError('No OBX file for {}'.format(cfg.pd_segment))
        sync.run_pd_correction(cfg, sess.seg_onsets, sess.boundaries, sess.obx_files, log)
        sess.invalidate('pd')
    elif key == 'sta':
        if not cfg.frames_path or not os.path.exists(cfg.frames_path):
            raise ValueError('Sparse-noise frames file not found: {!r}'.format(cfg.frames_path))
        rf.run_sta(cfg, sess.rfm_data, sess.good_ids, sess.spike_trains, overwrite, log, progress, cancel)
        sess.invalidate('rf')
    elif key == 'rffit':
        rf.run_rf_fits(cfg, sess.unit_depths, overwrite, log, progress, cancel)
        sess.invalidate('rf')
    elif key == 'rfmode':
        rf.build_rf_clean(cfg, log)
        sess.invalidate('rf')
    else:
        raise ValueError(key)
    sess.cache = {}
    return key

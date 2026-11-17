"""PSTH / raster helpers shared by all analyses (fast searchsorted versions)."""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def trial_counts(spike_times, onsets, bins):
    """(n_trials, n_bins) spike counts, bins relative to onset."""
    onsets = np.asarray(onsets, dtype=float)
    out = np.zeros((len(onsets), len(bins) - 1), dtype=np.float32)
    if len(spike_times) == 0:
        return out
    lo = np.searchsorted(spike_times, onsets + bins[0], side='left')
    hi = np.searchsorted(spike_times, onsets + bins[-1], side='left')
    for i, t0 in enumerate(onsets):
        if hi[i] > lo[i]:
            out[i], _ = np.histogram(spike_times[lo[i]:hi[i]] - t0, bins=bins)
    return out


def compute_psth(spike_times, onsets, bins, bin_s=None):
    """Cell 1 compute_psth: mean and SEM firing rate (Hz)."""
    if bin_s is None:
        bin_s = bins[1] - bins[0]
    if len(onsets) == 0:
        return np.zeros(len(bins) - 1), np.zeros(len(bins) - 1)
    c = trial_counts(spike_times, onsets, bins)
    return c.mean(axis=0) / bin_s, (c.std(axis=0) / np.sqrt(len(onsets))) / bin_s


def raster(spike_times, onsets, pre_s, post_s):
    """list of arrays of spike times (s) relative to each onset."""
    onsets = np.asarray(onsets, dtype=float)
    lo = np.searchsorted(spike_times, onsets - pre_s, side='left')
    hi = np.searchsorted(spike_times, onsets + post_s, side='left')
    return [spike_times[a:b] - t0 for a, b, t0 in zip(lo, hi, onsets)]


def draw_raster(ax, rows, scale=1000.0, color='black', lw=0.5, alpha=0.7, height=0.8):
    """Fast raster: one LineCollection-like vlines call."""
    xs, y0, y1 = [], [], []
    for j, r in enumerate(rows):
        if len(r):
            xs.append(r * scale)
            y0.append(np.full(len(r), j, dtype=float))
    if xs:
        x = np.concatenate(xs)
        y = np.concatenate(y0)
        ax.vlines(x, y, y + height, color=color, linewidth=lw, alpha=alpha)


def compute_trial_psths(spike_times, onsets, dur, bin_size):
    """Cell 43: stimulus-only [0, dur) trial matrix in Hz."""
    bins = np.arange(0, dur + bin_size, bin_size)
    onsets = np.asarray(onsets, dtype=float)
    out = np.zeros((len(onsets), len(bins) - 1))
    lo = np.searchsorted(spike_times, onsets, side='left')
    hi = np.searchsorted(spike_times, onsets + dur, side='left')
    for i, t0 in enumerate(onsets):
        if hi[i] > lo[i]:
            cnt, _ = np.histogram(spike_times[lo[i]:hi[i]] - t0, bins=bins)
            out[i] = cnt
    return out / bin_size


def compute_trial_psths_windowed(spike_times, onsets, pre_s, post_s, bin_size):
    """Cell 47: window [onset-pre, onset+post), bins[0] = onset - pre."""
    bins = np.arange(0, pre_s + post_s + bin_size, bin_size)
    onsets = np.asarray(onsets, dtype=float)
    out = np.zeros((len(onsets), len(bins) - 1))
    lo = np.searchsorted(spike_times, onsets - pre_s, side='left')
    hi = np.searchsorted(spike_times, onsets + post_s, side='left')
    for i, t0 in enumerate(onsets):
        if hi[i] > lo[i]:
            cnt, _ = np.histogram(spike_times[lo[i]:hi[i]] - (t0 - pre_s), bins=bins)
            out[i] = cnt
    return out / bin_size


def smoothed_trials(counts, bin_s, sigma):
    """per-trial smoothed rates (picker style)."""
    return np.array([gaussian_filter1d(c / bin_s, sigma=sigma) for c in counts]) \
        if len(counts) else np.zeros((0, counts.shape[1] if counts.ndim == 2 else 0))


def zscore_rows(mat, base_mask):
    bm = mat[:, base_mask].mean(axis=1, keepdims=True)
    bs = mat[:, base_mask].std(axis=1, keepdims=True) + 1e-6
    return (mat - bm) / bs


def population_matrix(spike_trains, cids, onsets, bins, sigma):
    bin_s = bins[1] - bins[0]
    mat = np.zeros((len(cids), len(bins) - 1), dtype=np.float32)
    for i, cid in enumerate(cids):
        m, _ = compute_psth(spike_trains[cid], onsets, bins, bin_s)
        mat[i] = gaussian_filter1d(m, sigma=sigma) if sigma else m
    return mat

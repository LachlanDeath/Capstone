import numpy as np


def smooth_median(freqs, n=5):
    """Causal median of the last n frames (ignores NaN). Works live."""
    out = np.full(len(freqs), np.nan)
    for k in range(len(freqs)):
        win = freqs[max(0, k - n + 1):k + 1]
        if np.sum(~np.isnan(win)) >= (n + 1) // 2:          # need a majority of real values
            out[k] = np.nanmedian(win)
    return out


def stability_gate(freqs, n=5, max_dev_cents=30):
    """NaN out frames whose last n values spread more than max_dev_cents (std, in cents)."""
    out = freqs.copy()
    for k in range(len(freqs)):
        win = freqs[max(0, k - n + 1):k + 1]
        win = win[~np.isnan(win)]
        if len(win) < n:                                     # not enough history yet
            out[k] = np.nan
            continue
        cents = 1200 * np.log2(win / np.median(win))         # each value's offset from the median
        if np.std(cents) > max_dev_cents:
            out[k] = np.nan
    return out
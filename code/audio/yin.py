# yin.py - YIN pitch detection (de Cheveigné & Kawahara, 2002), written to be readable
#
# Idea: a periodic signal looks the same if you shift it by one period.
# YIN shifts the window by every possible lag "tau" and finds the smallest
# shift where the signal best matches itself. That shift = the period.
#   f0 = fs / period
#
# The paper builds YIN in 6 steps:
#   1. autocorrelation
#   2. difference function
#   3. cumulative mean normalised difference (CMND)
#   4. absolute threshold
#   5. parabolic interpolation
#   6. best local estimate
import numpy as np


def _analyse(x, start, W, tau_min, tau_max, threshold):
    """Steps 1-5 on one analysis window starting at `start`.

    Returns (period, dip) where dip is the CMND value at the chosen minimum
    (lower = more confident), or None if no clear pitch.
    """
    frame = x[start:start + W]                 # the window we compare against
    span = x[start:start + W + tau_max]        # the window plus room to shift it by up to tau_max

    # Step 1: autocorrelation
    # r[tau] = sum of x[j] * x[j + tau]: big when the signal lines up with itself shifted by tau.
    # On its own, picking the biggest r[tau] makes lots of octave errors (the paper reports ~10%),
    # so YIN doesn't use r directly - it uses it to build the difference function in step 2.
    r = np.correlate(span, frame, mode="valid")     # r[0..tau_max]
    if r[0] < 1e-10:                           # r[0] is the window's energy: silence -> no pitch
        return None

    # Step 2: difference function
    # d[tau] = sum of (x[j] - x[j + tau])^2: how different the signal is from itself shifted by tau.
    # Expanding the square gives d[tau] = energy(window) + energy(shifted window) - 2 * r[tau]
    # (paper eq. 7), so it comes straight from the autocorrelation.
    # small d[tau] -> good match -> tau could be the period
    sq = np.concatenate(([0.0], np.cumsum(span * span)))
    energy_shifted = sq[W:W + tau_max + 1] - sq[:tau_max + 1]   # energy of x[start+tau .. start+tau+W]
    d = r[0] + energy_shifted - 2.0 * r
    d[0] = 0.0
    d = np.maximum(d, 0.0)                     # remove tiny negative values from rounding

    # Step 3: cumulative mean normalised difference (CMND)
    # divide each d[tau] by the average of d[1..tau], so values are around 1 and
    # a good match is close to 0. This stops tau = 0 (and tiny shifts) from always winning.
    cmnd = np.ones(tau_max + 1)
    running_sum = np.cumsum(d[1:])
    taus = np.arange(1, tau_max + 1)
    cmnd[1:] = d[1:] * taus / np.maximum(running_sum, 1e-12)

    # Step 4: absolute threshold
    # take the FIRST tau where cmnd dips below the threshold (the first dip is the true
    # period; later dips are 2x, 3x the period), then walk down to the bottom of that dip
    tau = max(tau_min, 1)
    while tau < tau_max and cmnd[tau] >= threshold:
        tau += 1
    if tau >= tau_max:                         # never dipped below threshold -> no clear pitch
        return None
    while tau + 1 < tau_max and cmnd[tau + 1] < cmnd[tau]:
        tau += 1

    # Step 5: parabolic interpolation
    # the true period is usually between two whole samples; fit a parabola through the
    # dip and its neighbours to find the in-between minimum. The paper fits the raw
    # difference function d (not cmnd), because cmnd's shape biases the result slightly.
    a, b, c = d[tau - 1], d[tau], d[tau + 1]
    curve = a - 2 * b + c
    shift = 0.5 * (a - c) / curve if curve > 0 else 0.0
    period = tau + shift

    return period, cmnd[tau]


def yin(buffer, fs, fmin=60.0, fmax=1100.0, threshold=0.4, window_s=0.060):
    """Return the pitch of `buffer` in Hz, or None if no clear pitch.

    `window_s` is the integration window length in seconds (the paper uses 25 ms).
    The buffer should be at least window + 2 * longest period long so step 6 has
    room to slide the window (4800 samples at 48 kHz is plenty).
    """
    x = buffer.astype(np.float64)

    # Which shifts to try: a high note has a short period, a low note a long one
    tau_min = int(fs / fmax)                   # 1100 Hz at 48 kHz -> 43 samples
    tau_max = int(fs / fmin)                   # 60 Hz   at 48 kHz -> 800 samples
    W = int(window_s * fs)                     # how many samples we compare at each shift
    W = min(W, len(x) - tau_max - 2)
    if W <= tau_max // 2:
        raise ValueError("buffer too short for fmin")

    # Step 6: best local estimate
    # A single window can land on a bad spot (note onset, noise burst) and pick a wrong dip.
    # So run steps 1-5 on windows slid around the centre of the buffer, by up to half the
    # longest period each way, and keep the estimate with the deepest (most confident) dip.
    # The paper then re-runs at the original position, only allowing periods within +-20%
    # of that best estimate, so the answer is for "now" but can't jump to a wrong octave.
    slack = len(x) - (W + tau_max)             # how far the window can move inside the buffer
    centre = slack // 2
    reach = min(tau_max // 2, centre)
    step = max(1, reach // 4)                  # check a handful of positions, not every sample
    starts = range(centre - reach, centre + reach + 1, step)

    best = None
    for s in starts:
        est = _analyse(x, s, W, tau_min, tau_max, threshold)
        if est is not None and (best is None or est[1] < best[1]):
            best = est
    if best is None:
        return None

    lo = max(tau_min, int(0.8 * best[0]))
    hi = min(tau_max, int(np.ceil(1.2 * best[0])) + 1)
    final = _analyse(x, centre, W, lo, hi, threshold)
    period = final[0] if final is not None else best[0]

    return fs / period


if __name__ == "__main__":
    # quick self-test: a 196 Hz tone (cello G string) with a few harmonics
    import time
    fs = 48000
    t = np.arange(4800) / fs
    tone = np.sin(2 * np.pi * 196 * t) + 0.5 * np.sin(2 * np.pi * 392 * t) + 0.3 * np.sin(2 * np.pi * 588 * t)
    start = time.perf_counter()
    f0 = yin(tone, fs)
    print(f"detected {f0:.2f} Hz (true 196.00 Hz) in {1000 * (time.perf_counter() - start):.2f} ms")

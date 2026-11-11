# yin.py - YIN pitch detection (de Cheveigné & Kawahara, 2002), written to be readable
#
# Idea: a periodic signal looks the same if you shift it by one period.
# YIN shifts the window by every possible lag "tau" and finds the smallest
# shift where the signal best matches itself. That shift = the period.
#   f0 = fs / period
import numpy as np


def yin(buffer, fs, fmin=60.0, fmax=1100.0, threshold=0.4):
    """Return the pitch of `buffer` in Hz, or None if no clear pitch."""
    x = buffer.astype(np.float64)

    # Which shifts to try: a high note has a short period, a low note a long one
    tau_min = int(fs / fmax)              # 1100 Hz at 48 kHz -> 43 samples
    tau_max = int(fs / fmin)              # 60 Hz   at 48 kHz -> 800 samples
    W = len(x) - tau_max                  # how many samples we compare at each shift

    # Step 1: difference function
    # d[tau] = sum of (x[j] - x[j + tau])^2: how different the signal is from itself shifted by tau
    # small d[tau] -> good match -> tau could be the period
    d = np.zeros(tau_max + 1)
    for tau in range(1, tau_max + 1):
        diff = x[:W] - x[tau:tau + W]
        d[tau] = np.sum(diff * diff)

    # Step 2: cumulative mean normalised difference (CMND)
    # divide each d[tau] by the average of d[1..tau], so values are around 1 and
    # a good match is close to 0. This stops tau = 0 (and tiny shifts) from always winning.
    cmnd = np.ones(tau_max + 1)
    running_sum = np.cumsum(d[1:])
    taus = np.arange(1, tau_max + 1)
    cmnd[1:] = d[1:] * taus / np.maximum(running_sum, 1e-12)

    # Step 3: absolute threshold
    # take the FIRST tau where cmnd dips below the threshold (the first dip is the true
    # period; later dips are 2x, 3x the period), then walk down to the bottom of that dip
    tau = tau_min
    while tau < tau_max and cmnd[tau] >= threshold:
        tau += 1
    if tau == tau_max:                    # never dipped below threshold -> no clear pitch
        return None
    while tau + 1 < tau_max and cmnd[tau + 1] < cmnd[tau]:
        tau += 1

    # Step 4: parabolic interpolation
    # the true period is usually between two whole samples; fit a parabola through the
    # dip and its neighbours to find the in-between minimum
    a, b, c = cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]
    shift = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    period = tau + shift

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
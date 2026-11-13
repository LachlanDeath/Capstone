import numpy as np


def _analyse(x, start, W, tau_min, tau_max, threshold):
    """Steps 1-5 on one window. Returns (period, dip) or None if no clear pitch."""
    frame = x[start:start + W]
    span = x[start:start + W + tau_max]

    # Step 1: autocorrelation
    r = np.correlate(span, frame, mode="valid")
    if r[0] < 1e-10:
        return None

    # Step 2: difference function
    sq = np.concatenate(([0.0], np.cumsum(span * span)))
    energy_shifted = sq[W:W + tau_max + 1] - sq[:tau_max + 1]
    d = r[0] + energy_shifted - 2.0 * r
    d[0] = 0.0
    d = np.maximum(d, 0.0)

    # Step 3: cumulative mean normalised difference
    cmnd = np.ones(tau_max + 1)
    running_sum = np.cumsum(d[1:])
    taus = np.arange(1, tau_max + 1)
    cmnd[1:] = d[1:] * taus / np.maximum(running_sum, 1e-12)

    # Step 4: absolute threshold
    tau = max(tau_min, 1)
    while tau < tau_max and cmnd[tau] >= threshold:
        tau += 1
    if tau >= tau_max:
        return None
    while tau + 1 < tau_max and cmnd[tau + 1] < cmnd[tau]:
        tau += 1

    # Step 5: parabolic interpolation
    a, b, c = d[tau - 1], d[tau], d[tau + 1]
    curve = a - 2 * b + c
    shift = 0.5 * (a - c) / curve if curve > 0 else 0.0
    period = tau + shift

    return period, cmnd[tau]


def yin(buffer, fs, fmin=60.0, fmax=1100.0, threshold=0.4, window_s=2048 / 48000):
    """Return the pitch of `buffer` in Hz, or None if no clear pitch."""
    x = buffer.astype(np.float64)

    tau_min = int(fs / fmax)
    tau_max = int(fs / fmin)
    W = int(window_s * fs)
    W = min(W, len(x) - tau_max - 2)
    if W <= tau_max // 2:
        raise ValueError("buffer too short for fmin")

    # Step 6: best local estimate
    slack = len(x) - (W + tau_max)
    centre = slack // 2
    reach = min(tau_max // 2, centre)
    step = max(1, reach // 4)
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
    import time
    fs = 48000
    t = np.arange(4800) / fs
    tone = np.sin(2 * np.pi * 196 * t) + 0.5 * np.sin(2 * np.pi * 392 * t) + 0.3 * np.sin(2 * np.pi * 588 * t)
    start = time.perf_counter()
    f0 = yin(tone, fs)
    print(f"detected {f0:.2f} Hz (true 196.00 Hz) in {1000 * (time.perf_counter() - start):.2f} ms")

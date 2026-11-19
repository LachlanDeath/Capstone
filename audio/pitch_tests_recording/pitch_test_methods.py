"""YIN steps 1-2, all sizes in samples.
 
Eq. 1 from the paper: every lag is compared over the full W samples,
so each frame must hold W + tau_max samples.
    W = 1024, tau_max = 1024  ->  frame = 2048 samples (42.7 ms at 48 kHz)
    lowest detectable f0 = fs / tau_max = 46.9 Hz
"""
import numpy as np
from voicing_gate import is_silent
 
W = 1024        # YIN integration window (samples)
TAU_MAX = 1024    # largest lag (samples) -> lowest f0
FRAME = W + TAU_MAX
 
 
def acf(x, W=W, tau_max=TAU_MAX):
    """Step 1: r(tau) = sum_{j=0}^{W-1} x[j] * x[j+tau], tau = 0..tau_max"""
    return np.array([np.dot(x[:W], x[tau:tau + W]) for tau in range(tau_max + 1)])
 
 
def diff_fn(x, W=W, tau_max=TAU_MAX):
    """Step 2: d(tau) = sum_{j=0}^{W-1} (x[j] - x[j+tau])^2, tau = 0..tau_max"""
    ref = x[:W]
    return np.array([np.sum((ref - x[tau:tau + W]) ** 2) for tau in range(tau_max + 1)])
 
 
def pitch_acf(x, fs, fmax=1000, W=W, tau_max=TAU_MAX):
    assert len(x) >= W + tau_max, f"need {W + tau_max} samples, got {len(x)}"
    tau_min = int(fs / fmax)                       # skip lags above fmax (and the tau=0 peak)
    r = acf(x, W, tau_max)
    return fs / (tau_min + np.argmax(r[tau_min:]))
 
 
def pitch_diff(x, fs, fmax=1000, W=W, tau_max=TAU_MAX):
    assert len(x) >= W + tau_max, f"need {W + tau_max} samples, got {len(x)}"
    tau_min = int(fs / fmax)                       # skip lags above fmax (and the tau=0 dip)
    d = diff_fn(x, W, tau_max)
    return fs / (tau_min + np.argmin(d[tau_min:]))

def pitch_diff_mod(x, fs, fmax=1000, margin=0.1, W=W, tau_max=TAU_MAX):
    assert len(x) >= W + tau_max, f"need {W + tau_max} samples, got {len(x)}"
    tau_min = int(fs / fmax)
    d = diff_fn(x, W, tau_max)

    best_tau, best_val = tau_min, d[tau_min]
    for tau in range(tau_min + 1, tau_max + 1):
        if d[tau] < best_val - margin:        # lower than the last min by at least margin
            best_tau, best_val = tau, d[tau]
    return fs / best_tau

def cmndf(d):
    """Step 3: d'(tau) = d(tau) / mean(d[1..tau]),  d'(0) = 1"""
    dp = np.ones_like(d)
    dp[1:] = d[1:] * np.arange(1, len(d)) / np.cumsum(d[1:])
    return dp


def pitch_cmndf(x, fs, fmax=1000, W=W, tau_max=TAU_MAX):
    assert len(x) >= W + tau_max, f"need {W + tau_max} samples, got {len(x)}"
    tau_min = int(fs / fmax)                       # now only sets the highest pitch
    dp = cmndf(diff_fn(x, W, tau_max))
    return fs / (tau_min + np.argmin(dp[tau_min:]))

def pitch_threshold(x, fs, fmax=1000, threshold=0.1, W=W, tau_max=TAU_MAX):
    assert len(x) >= W + tau_max, f"need {W + tau_max} samples, got {len(x)}"
    tau_min = int(fs / fmax)
    dp = cmndf(diff_fn(x, W, tau_max))

    for tau in range(tau_min, tau_max):
        if dp[tau] < threshold:                                  # first dip below threshold
            while tau + 1 <= tau_max and dp[tau + 1] < dp[tau]:  # slide to its bottom
                tau += 1
            return fs / tau
    return fs / (tau_min + np.argmin(dp[tau_min:]))              # none found: global min

def parabolic(d, tau):
    """Step 5: fit a parabola through d[tau-1], d[tau], d[tau+1]; return its minimum."""
    if tau <= 0 or tau >= len(d) - 1:
        return float(tau)                          # no neighbours on one side
    a, b, c = d[tau - 1], d[tau], d[tau + 1]
    den = a - 2 * b + c
    return tau + (a - c) / (2 * den) if den != 0 else float(tau)


def pitch_parabolic(x, fs, fmax=1000, threshold=0.0, W=W, tau_max=TAU_MAX):
    assert len(x) >= W + tau_max, f"need {W + tau_max} samples, got {len(x)}"
    tau_min = int(fs / fmax)
    d = diff_fn(x, W, tau_max)
    dp = cmndf(d)

    tau = None
    for t in range(tau_min, tau_max):                            # step 4: pick the dip
        if dp[t] < threshold:
            while t + 1 <= tau_max and dp[t + 1] < dp[t]:
                t += 1
            tau = t
            break
    if tau is None:
        tau = tau_min + np.argmin(dp[tau_min:])

    return fs / parabolic(d, tau)                                # step 5: refine on raw d


def pitch_local(x, fs, history, hop=480, fmin=50, fmax=200, threshold=0.1, n_back=5, W=W, tau_max=TAU_MAX, silence_db=-20):
    assert len(x) >= W + tau_max, f"need {W + tau_max} samples, got {len(x)}"

    if is_silent(x[-hop:], silence_db):          # silence gate: check the newest hop first
        history.clear()                          # don't carry the old note into the next one
        return None

    tau_min = int(fs / fmax)
    tau_hi = min(tau_max, int(fs / fmin))
    d = diff_fn(x, W, tau_max)
    dp = cmndf(d)

    tau = None                                                   # step 4
    for t in range(tau_min, tau_hi):
        if dp[t] < threshold:
            while t + 1 <= tau_hi and dp[t + 1] < dp[t]:
                t += 1
            tau = t
            break
    if tau is None:
        tau = tau_min + np.argmin(dp[tau_min:tau_hi + 1])

    # step 6 (look-back): if a recent frame was more confident, search near its period
    if history:
        best_tau, best_conf = min(history, key=lambda h: h[1])
        if best_conf < dp[tau]:
            lo = max(tau_min, int(best_tau * 0.8))
            hi = min(tau_hi, int(best_tau * 1.2))
            tau = lo + np.argmin(dp[lo:hi + 1])

    history.append((tau, dp[tau]))                               # remember this frame
    del history[:-n_back]                                        # keep only the last n_back

    return fs / parabolic(d, tau)                                # step 5


def track_pitch(x, fs, hop=480, frame=FRAME, mode="yin", fmin=50, fmax=1000, threshold=0.1, n_back=5,silence_db=-20):
    """Slide a frame along x by hop samples. Returns (times, freqs); freqs is NaN where silent."""
    starts = range(0, len(x) - frame + 1, hop)
    times, freqs = [], []
    history = []
    if fmin <= 0 or fmax <= 0 or fmin >= fmax:
        raise ValueError(f"invalid fmin/fmax: {fmin}/{fmax}")
    for s in starts:
        chunk = x[s:s + frame]

        if mode == "parabolic":
            f0 = pitch_parabolic(chunk, fs, fmax=fmax)
        elif mode == "yin":
            f0 = pitch_local(chunk, fs, history=history, hop=hop, fmin=fmin, fmax=fmax, threshold=threshold, n_back=n_back, silence_db=silence_db)
        elif mode == "diff":
            f0 = pitch_diff(chunk, fs, fmax=fmax)
        elif mode == "diff_mod":
            f0 = pitch_diff_mod(chunk, fs, fmax=fmax)
        elif mode == "cmndf":
            f0 = pitch_cmndf(chunk, fs, fmax=fmax)
        else:
            raise ValueError(f"unknown mode {mode}")
        
        if f0 is None or not (fmin <= f0 <= fmax):
            f0 = np.nan

        times.append((s + frame) / fs)   # time of the newest sample in the frame
        freqs.append(f0)

    print(f"mode {mode}: fmin {fmin} Hz, fmax {fmax} Hz")
    print(f"threshold {threshold}") if mode == "yin" or mode == "parabolic" else None
    print(f"n_back {n_back}") if mode == "yin" else None

    return np.array(times), np.array(freqs)



if __name__ == "__main__":
    fs, f0 = 48000, 220.0
    t = np.arange(FRAME) / fs
    tone = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 6))
    crescendo = tone * np.linspace(0.2, 1.0, FRAME)
 
    for name, sig in [("steady", tone), ("crescendo", crescendo)]:
        print(f"{name:10s}  ACF: {pitch_acf(sig, fs):7.2f} Hz   "
              f"Diff: {pitch_diff(sig, fs):7.2f} Hz   (true {f0} Hz)")
 
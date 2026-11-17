"""YIN steps 1-2, all sizes in samples.
 
Eq. 1 from the paper: every lag is compared over the full W samples,
so each frame must hold W + tau_max samples.
    W = 1024, tau_max = 1024  ->  frame = 2048 samples (42.7 ms at 48 kHz)
    lowest detectable f0 = fs / tau_max = 46.9 Hz
"""
import numpy as np
 
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


if __name__ == "__main__":
    fs, f0 = 48000, 220.0
    t = np.arange(FRAME) / fs
    tone = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 6))
    crescendo = tone * np.linspace(0.2, 1.0, FRAME)
 
    for name, sig in [("steady", tone), ("crescendo", crescendo)]:
        print(f"{name:10s}  ACF: {pitch_acf(sig, fs):7.2f} Hz   "
              f"Diff: {pitch_diff(sig, fs):7.2f} Hz   (true {f0} Hz)")
 
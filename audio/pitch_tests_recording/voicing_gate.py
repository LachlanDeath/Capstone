import numpy as np


def power(x):
    """Mean power of a window: (1/N) * sum(x[j] * x[j])"""
    return np.dot(x, x) / len(x)


def loudness_db(x):
    """Loudness in dB relative to full scale (0 dB = loudest possible, amplitude 1.0)."""
    return 10 * np.log10(power(x) + 1e-12)


def is_silent(x, threshold_db=-50):
    """True if the window is quieter than threshold_db (aubio's default is -50 dB)."""
    return loudness_db(x) < threshold_db
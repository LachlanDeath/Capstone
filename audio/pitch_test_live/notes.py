import numpy as np

NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def freq_to_note(f, a4=440.0):
    """Nearest note name and cents offset, e.g. 66.0 Hz -> ("C2", +15.5). (None, nan) for no pitch."""
    if f is None or np.isnan(f) or f <= 0:
        return None, np.nan
    n = 12 * np.log2(f / a4) + 69          # semitone number (A4 = 69), not rounded
    nearest = int(round(n))
    cents = 100 * (n - nearest)            # how far from the nearest note, -50 to +50
    name = NAMES[nearest % 12] + str(nearest // 12 - 1)
    return name, cents

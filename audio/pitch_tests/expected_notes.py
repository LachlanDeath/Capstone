import numpy as np

NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# "C2" -> 65.41 Hz, "Db2" -> 69.30 Hz, ... (A4 = 440)
FREQ = {f"{n}{octave}": 440 * 2 ** ((12 * (octave + 1) + i - 69) / 12)
        for octave in range(2, 7) for i, n in enumerate(NAMES)}

# C string (sulC)
SCALE = ["C2", "Db2", "D2", "Eb2", "E2", "F2", "Gb2", "G2", "Ab2", "A2", "Bb2", "B2"]

# G string (sulG)
# SCALE = ["G2", "Ab2", "A2", "Bb2", "B2", "C3", "Db3", "D3", "Eb3", "E3", "F3", "Gb3"]

# D string (sulD)
# SCALE = ["D3", "Eb3", "E3", "F3", "Gb3", "G3", "Ab3", "A3", "Bb3", "B3", "C4", "Db4"]

# A string (sulA)
# SCALE = ["A3", "Bb3", "B3", "C4", "Db4", "D4", "Eb4", "E4", "F4", "Gb4", "G4", "Ab4"]


def find_notes(x, fs, scale=SCALE, smooth=0.3, thresh_db=-30):
    """Find notes from the raw signal's loudness. Returns [(t_start, t_end, freq), ...].

    1. Smooth the signal's power over `smooth` seconds (kills flicker).
    2. 'On' = within `thresh_db` of the loudest point (no absolute level to tune).
    3. Each on-stretch is one note, in order.
    """
    n = int(smooth * fs)
    env = np.convolve(x ** 2, np.ones(n) / n, mode="same")      # smoothed power
    env_db = 10 * np.log10(env / env.max() + 1e-12)
    on = (env_db > thresh_db).astype(int)

    edges = np.diff(np.concatenate([[0], on, [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]

    print(f"found {len(starts)} notes, scale has {len(scale)}")
    return [(s / fs, e / fs, FREQ[note]) for note, s, e in zip(scale, starts, ends)]


def expected_pitch(times, x, fs, scale=SCALE, **kw):
    """Expected f0 at each time in `times` (NaN between notes). Same length as freqs."""
    exp = np.full(len(times), np.nan)
    for t0, t1, f in find_notes(x, fs, scale, **kw):
        exp[(times >= t0) & (times <= t1)] = f
    return exp
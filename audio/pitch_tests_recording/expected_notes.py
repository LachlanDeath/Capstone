import numpy as np

"""C string (sulC)
SCALE = ["C2", "Db2", "D2", "Eb2", "E2", "F2", "Gb2", "G2", "Ab2", "A2", "Bb2", "B2"]
G string (sulG)
SCALE = ["G2", "Ab2", "A2", "Bb2", "B2", "C3", "Db3", "D3", "Eb3", "E3", "F3", "Gb3"]
D string (sulD)
SCALE = ["D3", "Eb3", "E3", "F3", "Gb3", "G3", "Ab3", "A3", "Bb3", "B3", "C4", "Db4"]
A string (sulA)
SCALE = ["A3", "Bb3", "B3", "C4", "Db4", "D4", "Eb4", "E4", "F4", "Gb4", "G4", "Ab4"]"""


NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# "C2" -> 65.41 Hz, "Db2" -> 69.30 Hz, ... (A4 = 440)
FREQ = {f"{n}{octave}": 440 * 2 ** ((12 * (octave + 1) + i - 69) / 12)
        for octave in range(2, 7) for i, n in enumerate(NAMES)}

def expected_pitch(times, label_file):
    """Expected f0 at each time from an Audacity label file (start, end, note). NaN elsewhere."""
    exp = np.full(len(times), np.nan)
    for line in open(label_file):
        start, end, note = line.split()[:3]
        exp[(times >= float(start)) & (times <= float(end))] = FREQ[note]
    return exp

def error_tolerance(freqs, freqs_expected, cents=50):
    """True where the detected pitch is within ±cents of the expected note.
    Frames with no expected note (NaN) are ignored."""
    lo = freqs_expected * 2 ** (-cents / 1200)
    hi = freqs_expected * 2 ** ( cents / 1200)
    ok = (freqs >= lo) & (freqs <= hi)
    ok[np.isnan(freqs_expected) & np.isnan(freqs)] = np.nan  
    return ok
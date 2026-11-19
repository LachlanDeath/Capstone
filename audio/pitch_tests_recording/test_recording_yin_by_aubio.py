from pathlib import Path
import numpy as np
import soundfile as sf
import aubio
from scipy.signal import resample_poly
from plotting import plot_pitch_tracking
from pitch_test_methods import FRAME
from expected_notes import expected_pitch, error_tolerance

HERE = Path(__file__).resolve().parent
FILE = HERE / "data" / "Cello.arco.pp.sulC.C2B2.mono.aif"
LABELS = HERE / "data" / "Cello.arco.pp.sulC.C2B2.mono.txt"
TOLERANCE = 100                          # cents
fs, hop, frame = 48000, 480, FRAME       # same hop/frame as your YIN for a fair comparison


def track_aubio(x, fs, hop=480, frame=2048, method="yin", tolerance=0.1, fmin=65, fmax=200):
    """Run aubio hop by hop. Returns (times, freqs) like track_pitch; NaN where aubio gives no pitch."""
    p = aubio.pitch(method, frame, hop, fs)
    p.set_unit("Hz")
    p.set_tolerance(tolerance)
    times, freqs = [], []
    for k in range(len(x) // hop):
        f0 = p(x[k * hop:(k + 1) * hop].astype(np.float32))[0]   # exactly one hop per call
        end = (k + 1) * hop                                      # newest sample in aubio's buffer
        if end < frame:                                          # buffer not full yet: skip
            continue
        times.append(end / fs)                                   # same time convention as track_pitch
        freqs.append(f0 if fmin <= f0 <= fmax else np.nan)      # aubio returns 0 for "no pitch"
    return np.array(times), np.array(freqs)


# Main
if __name__ == "__main__":
    x, fs_in = sf.read(FILE)
    x = resample_poly(x, fs, fs_in).astype(np.float32)   # 44100 -> 48000

    for method in ["yin", "yinfast", "yinfft", "mcomb", "fcomb", "specacf", "schmitt"]:
        times, freqs = track_aubio(x, fs, hop=hop, frame=frame, method=method, tolerance=0.1)
        freqs_expected = expected_pitch(times, LABELS)
        correct = error_tolerance(freqs, freqs_expected, cents=TOLERANCE)
        print(f"aubio {method}: {100 * np.nanmean(correct):.1f}% of labelled hops "
              f"within ±{TOLERANCE} cents")
        plot_pitch_tracking(x, fs, times, freqs, freqs_expected, mode=f"aubio {method}")
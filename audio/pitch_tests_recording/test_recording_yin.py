from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
from filters import smooth_median, stability_gate
from plotting import plot_pitch_tracking
from pitch_test_methods import track_pitch, FRAME
from expected_notes import expected_pitch, error_tolerance
from voicing_gate import power
 
HERE = Path(__file__).resolve().parent
FILE = HERE / "data" / "Cello.arco.pp.sulC.C2B2.mono.aif"
LABELS = HERE / "data" / "Cello.arco.pp.sulC.C2B2.mono.txt"
TOLERANCE = 100  # cents
 
fs, hop, frame = 48000, 480, FRAME       # frame must equal W + tau_max in pitch_diff
                   # below this = silence (tune by printing power)
resolution = [1, 5, 10, 20, 50, 100]   # for testing n_back


# Main
if __name__ == "__main__":
    x, fs_in = sf.read(FILE)
    x = resample_poly(x, fs, fs_in).astype(np.float32)   # 44100 -> 48000

    for i in resolution:
        # times, freqs = track_pitch(x, fs)
        times, freqs = track_pitch(x, fs, hop=hop, frame=frame, mode="yin", fmin=65, fmax=1000, threshold=0.1, n_back=i, silence_db=-50)
        freqs_expected = expected_pitch(times, LABELS)

         # raw
        raw = 100 * np.nanmean(error_tolerance(freqs, freqs_expected, cents=TOLERANCE))

        # smoothed
        freqs_smooth = smooth_median(stability_gate(freqs, n=5, max_dev_cents=50), n=i)
        smooth = 100 * np.nanmean(error_tolerance(freqs_smooth, freqs_expected, cents=TOLERANCE))

        print(f"raw: {raw:.1f}%   smoothed: {smooth:.1f}%   (±{TOLERANCE} cents)")

        plot_pitch_tracking(x, fs, times, freqs, freqs_expected, mode="yin raw", threshold=0.1, n_back=i)
        plot_pitch_tracking(x, fs, times, freqs_smooth, freqs_expected, mode="yin smoothed", threshold=0.1, n_back=i)
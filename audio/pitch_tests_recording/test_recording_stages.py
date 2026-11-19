from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
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

    # times, freqs = track_pitch(x, fs)
    times, freqs = track_pitch(x, fs, hop=hop, frame=frame, mode="diff_mod", fmin=65, fmax=200)
    freqs_expected = expected_pitch(times, x, fs)
    error = freqs - freqs_expected
    error_total = np.nanmean(np.abs(error))
    print(f"diff_mod mean error = {error_total:.1f} Hz,")
    plot_pitch_tracking(x, fs, times, freqs, freqs_expected, mode="diff_mod_no_power_thresh")

    # times, freqs = track_pitch(x, fs)
    times, freqs = track_pitch(x, fs, hop=hop, frame=frame, mode="diff", fmin=65, fmax=200)
    freqs_expected = expected_pitch(times, x, fs)
    error = freqs - freqs_expected
    error_total = np.nanmean(np.abs(error))
    print(f"diff mean error = {error_total:.1f} Hz,")   
    plot_pitch_tracking(x, fs, times, freqs, freqs_expected, mode="diff_no_power_thresh")

    # times, freqs = track_pitch(x, fs)
    times, freqs = track_pitch(x, fs, hop=hop, frame=frame, mode="cmndf", fmin=65, fmax=200)
    freqs_expected = expected_pitch(times, x, fs)
    error = freqs - freqs_expected
    error_total = np.nanmean(np.abs(error))
    print(f"cmndf mean error = {error_total:.1f} Hz,")
    plot_pitch_tracking(x, fs, times, freqs, freqs_expected, mode="cmndf_no_power_thresh")

    # times, freqs = track_pitch(x, fs)
    times, freqs = track_pitch(x, fs, hop=hop, frame=frame, mode="parabolic", fmin=65, fmax=200, threshold=0.1)
    freqs_expected = expected_pitch(times, x, fs)
    error = freqs - freqs_expected
    error_total = np.nanmean(np.abs(error))
    print(f"parabolic mean error = {error_total:.1f} Hz,")
    plot_pitch_tracking(x, fs, times, freqs, freqs_expected, mode="parabolic_no_power_thresh", threshold=0.1, n_back=None)

    # times, freqs = track_pitch(x, fs)
    times, freqs = track_pitch(x, fs, hop=hop, frame=frame, mode="yin", fmin=65, fmax=200, threshold=0.1, n_back=20)
    freqs_expected = expected_pitch(times, LABELS)
    note_identified_correctly = error_tolerance(freqs, freqs_expected, cents=TOLERANCE)
    error_total = np.nanmean(np.abs(note_identified_correctly))*100
    print(f"yin number of notes ID correctly = {error_total:.5f}%, with a tolerance of {TOLERANCE} cents")
    plot_pitch_tracking(x, fs, times, freqs, freqs_expected, mode="yin_no_power_thresh", threshold=0.1, n_back=20)
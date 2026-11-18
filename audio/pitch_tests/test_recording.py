from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
from plotting import plot_pitch_tracking
from pitch_test_methods import track_pitch, FRAME
from expected_notes import expected_pitch
 
HERE = Path(__file__).resolve().parent
FILE = HERE / "data" / "Cello.arco.pp.sulC.C2B2.mono.aif"
 
fs, hop, frame = 48000, 480, FRAME       # frame must equal W + tau_max in pitch_diff
                   # below this = silence (tune by printing power)
resolution = [1, 5, 10, 20, 50, 100]   # for testing n_back


# Main
if __name__ == "__main__":
    x, fs_in = sf.read(FILE)
    x = resample_poly(x, fs, fs_in).astype(np.float32)   # 44100 -> 48000

    # # times, freqs = track_pitch(x, fs)
    # times, freqs = track_pitch(x, fs, hop=hop, frame=frame, mode="diff_mod", fmin=65, fmax=200)
    # freqs_expected = expected_pitch(times, x, fs)
    # error = freqs - freqs_expected
    # error_total = np.nanmean(np.abs(error))
    # print(f"diff_mod mean error = {error_total:.1f} Hz,")
    # plot_pitch_tracking(x, fs, times, freqs, freqs_expected, mode="diff_mod")

    # # times, freqs = track_pitch(x, fs)
    # times, freqs = track_pitch(x, fs, hop=hop, frame=frame, mode="diff", fmin=65, fmax=200)
    # freqs_expected = expected_pitch(times, x, fs)
    # error = freqs - freqs_expected
    # error_total = np.nanmean(np.abs(error))
    # print(f"diff mean error = {error_total:.1f} Hz,")   
    # plot_pitch_tracking(x, fs, times, freqs, freqs_expected, mode="diff")

    # # times, freqs = track_pitch(x, fs)
    # times, freqs = track_pitch(x, fs, hop=hop, frame=frame, mode="cmndf", fmin=65, fmax=200)
    # freqs_expected = expected_pitch(times, x, fs)
    # error = freqs - freqs_expected
    # error_total = np.nanmean(np.abs(error))
    # print(f"cmndf mean error = {error_total:.1f} Hz,")
    # plot_pitch_tracking(x, fs, times, freqs, freqs_expected, mode="cmndf")

    # # times, freqs = track_pitch(x, fs)
    # times, freqs = track_pitch(x, fs, hop=hop, frame=frame, mode="parabolic", fmin=65, fmax=200, threshold=0.1)
    # freqs_expected = expected_pitch(times, x, fs)
    # error = freqs - freqs_expected
    # error_total = np.nanmean(np.abs(error))
    # print(f"parabolic mean error = {error_total:.1f} Hz,")
    # plot_pitch_tracking(x, fs, times, freqs, freqs_expected, mode="parabolic", threshold=0.1, n_back=None)

    for i in resolution:
        # times, freqs = track_pitch(x, fs)
        times, freqs = track_pitch(x, fs, hop=hop, frame=frame, mode="yin", fmin=65, fmax=200, threshold=0.1, n_back=i)
        freqs_expected = expected_pitch(times, x, fs)
        error = freqs - freqs_expected
        error_total = np.nanmean(np.abs(error))
        print(f"yin mean error = {error_total:.1f} Hz,")
        plot_pitch_tracking(x, fs, times, freqs, freqs_expected, mode="yin", threshold=0.1, n_back=i)
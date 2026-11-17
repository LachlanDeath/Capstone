from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
from pitch_test_methods import pitch_diff, pitch_diff_mod, FRAME
from voicing_gate import power
from expected_notes import expected_pitch
 
HERE = Path(__file__).resolve().parent
FILE = HERE / "data" / "Cello.arco.pp.sulC.C2B2.mono.aif"
 
fs, hop, frame = 48000, 480, FRAME       # frame must equal W + tau_max in pitch_diff
                   # below this = silence (tune by printing power)
 
 
def track_pitch(x, fs, hop=hop, frame=frame):
    """Slide a frame along x by hop samples. Returns (times, freqs); freqs is NaN where silent."""
    starts = range(0, len(x) - frame + 1, hop)
    times, freqs = [], []
    for s in starts:
        chunk = x[s:s + frame]


        f0 = pitch_diff_mod(chunk, fs, 200)



        times.append((s + frame) / fs)   # time of the newest sample in the frame
        freqs.append(f0)
    return np.array(times), np.array(freqs)
 


# Plotting 
if __name__ == "__main__":
    x, fs_in = sf.read(FILE)
    x = resample_poly(x, fs, fs_in).astype(np.float32)   # 44100 -> 48000
    times, freqs = track_pitch(x, fs)
    freqs_expected = expected_pitch(times, x, fs)
    error = freqs - freqs_expected

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    fig.suptitle("method 1 pitch track: Cello C2-B2 (pp)")

    ax1.plot(np.arange(len(x)) / fs, x, lw=0.5)
    ax1.set_ylabel("amplitude")

    ax2.plot(times, freqs, ".", ms=2, label="predicted")
    ax2.plot(times, freqs_expected, ".", ms=2, label="expected")

    ax2.set_ylabel("f0 (Hz)")
    ax2.set_xlabel("time (s)")
    ax2.legend()

    plt.tight_layout()
    plt.show()
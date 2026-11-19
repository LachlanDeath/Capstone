from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_pitch_tracking(x, fs=48000, times=None, freqs=None, freqs_expected=None,
                        mode=None, threshold=None, n_back=None):
    """Plot the waveform and pitch track, save it as data/<title>.png next to this file, then close."""

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    fig.suptitle(mode + f" pitch tracking: Cello C2-B2 (pp), threshold {threshold}, n_back {n_back}")

    ax1.plot(np.arange(len(x)) / fs, x, lw=0.5)
    ax1.set_ylabel("amplitude")

    if times is not None and freqs is not None:
        ax2.plot(times, freqs, ".", ms=2, label="predicted")
    if times is not None and freqs_expected is not None:
        ax2.plot(times, freqs_expected, ".", ms=2, label="expected")

    ax2.set_ylabel("f0 (Hz)")
    ax2.set_xlabel("time (s)")
    ax2.legend()
    plt.tight_layout()

    out_dir = Path(__file__).resolve().parent / "plots"          # data/ next to this file
    out_dir.mkdir(exist_ok=True)
    name = re.sub(r"[^\w\-]+", "_", mode + f" pitch tracking: Cello C2-B2 (pp), threshold {threshold}, n_back {n_back}").strip("_")           # make the title filename-safe
    fig.savefig(out_dir / f"{name}.png", dpi=150)
    plt.close(fig)

import queue
import numpy as np
import sounddevice as sd
from pitch_test_methods import stream_pitch, FRAME
from notes import freq_to_note

fs, hop = 48000, 480

blocks = queue.Queue()                   # audio thread -> main loop, nothing dropped or doubled
print(sd.query_devices())                # list all audio devices


def callback(data, frames, time, status):        # audio thread: just hand the block over
    blocks.put(data[:, 0].copy())


def live_blocks():
    while True:
        yield blocks.get()                       # waits for the next 480 samples


with sd.InputStream(device="Scarlett", samplerate=fs, channels=1, blocksize=hop, callback=callback):
    for k, (t, f0, f_smooth) in enumerate(stream_pitch(live_blocks(), fs, hop=hop, frame=FRAME,
                                                        mode="yin", fmin=65, fmax=1500, threshold=0.1,
                                                        n_back=5, n_smooth=5, max_dev_cents=30)):
        note, cents = freq_to_note(f_smooth)
        print("no pitch" if note is None else f"{note} ({cents:+.1f} cents)")

        if k % 100 == 0 and blocks.qsize() > 5:          # check once a second
            print(f"falling behind: {blocks.qsize()} blocks waiting")
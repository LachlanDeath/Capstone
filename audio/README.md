# Current functions in audio subsystem

## MusicXML frequency extraction

This will eventually be used to compare the values read from a piece of music to the output of our pitch detection algorithm.

- A simple MusicXML file (a simple scale) is generated with `musicXML_build.py`.
- The file is read, and the frequency of each note is extracted and printed in the terminal with `musicXML_read.py`.

## Pitch detection

This will be used to capture the pitch of the live audio.

- Audio is captured by the audio interface and microphone (Scarlett Solo, CB 100). This is done in `get_audio.py`.
- The YIN algorithm is used to get the pitch from the captured audio data. `yin.py` is called in `get_audio.py`.

## Requirements

The libraries needed to run these files are:

- `sounddevice`
- `numpy`
- `music21`

You can install these individually or install `requirements.txt`, but the latter will also install all the video subsystem libraries. Using a `.venv` is recommended.

Install only the audio libraries:

```bash
pip install sounddevice numpy music21
```

Install all requirements (audio and video):

```bash
pip install -r requirements.txt
```

## PortAudio

PortAudio is used with `sounddevice` to make the USB audio data usable in Python. No audio drivers are required, as the Scarlett audio interface is class-compliant.

### How to install PortAudio

**Linux**

```bash
sudo apt install libportaudio2
pip install sounddevice
```

**Mac**

```bash
pip install sounddevice      # PortAudio is bundled, nothing else needed
```

**Windows**

```bash
pip install sounddevice      # PortAudio is bundled
```
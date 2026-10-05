## Capture
#### Frequency
- [x] 48kHz microphone. Standard smartphone sample rate.
- [ ] 96kHz microphone. Only use if the 48kHz is insufficient.
#### Location 
- [x] Stand mounted. 
  - Easy to setup.
  - Cheap.
- [ ] Cello mounted
  - Higher sound quality.
  - Wired (cable management). 
  - Wireless (cost).
#### Options

## Pre-processing
#### Filtering 
- High pass filter. 60Hz
- Low pass filter. 2kHz
#### Framing
- Frame to must include two full cycles of the lowest note (60Hz). 
- Window 33.3 ms (covers 2 cycles of lowest note).
- Hop 10 ms (100 estimates per second). Increase or decrease as needed. 
#### Voice gate
- Compute power of the signal and compare against a preset minimum.
## Pitch detection 
- [x] YIN/pYIN
- [ ] YAAPT
- [ ] CREPE
## Post-processing
#### Octave correction 
- Remove outlier frames. 
#### Match to score 
- Compare to MusicXML file.

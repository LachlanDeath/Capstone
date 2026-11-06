from music21 import converter
import numpy as np
s = converter.parse("test_scale.musicxml").expandRepeats().stripTies()
for n in s.flatten().notes:
    print(n.offset, n.quarterLength, n.pitches, n.pitches[0].frequency)

"""Simple MusicXML test: write a C major scale to a .musicxml file, then parse it back."""

import xml.etree.ElementTree as ET

OUTPUT_FILE = "test_scale.musicxml"

# (step, octave) for one octave of C major, each a quarter note
SCALE = [("C", 4), ("D", 4), ("E", 4), ("F", 4), ("G", 4), ("A", 4), ("B", 4), ("C", 5)]
DIVISIONS = 1  # divisions per quarter note
BEATS_PER_MEASURE = 4


def build_score(notes):
    score = ET.Element("score-partwise", version="4.0")

    part_list = ET.SubElement(score, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = "Piano"

    part = ET.SubElement(score, "part", id="P1")

    for i in range(0, len(notes), BEATS_PER_MEASURE):
        measure = ET.SubElement(part, "measure", number=str(i // BEATS_PER_MEASURE + 1))

        if i == 0:
            attributes = ET.SubElement(measure, "attributes")
            ET.SubElement(attributes, "divisions").text = str(DIVISIONS)
            key = ET.SubElement(attributes, "key")
            ET.SubElement(key, "fifths").text = "0"
            time = ET.SubElement(attributes, "time")
            ET.SubElement(time, "beats").text = str(BEATS_PER_MEASURE)
            ET.SubElement(time, "beat-type").text = "4"
            clef = ET.SubElement(attributes, "clef")
            ET.SubElement(clef, "sign").text = "G"
            ET.SubElement(clef, "line").text = "2"

        for step, octave in notes[i:i + BEATS_PER_MEASURE]:
            note = ET.SubElement(measure, "note")
            pitch = ET.SubElement(note, "pitch")
            ET.SubElement(pitch, "step").text = step
            ET.SubElement(pitch, "octave").text = str(octave)
            ET.SubElement(note, "duration").text = str(DIVISIONS)
            ET.SubElement(note, "type").text = "quarter"

    return ET.ElementTree(score)


def write_score(tree, path):
    ET.indent(tree, space="  ")
    with open(path, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n')
        f.write(
            b'<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
            b'"http://www.musicxml.org/dtds/partwise.dtd">\n'
        )
        tree.write(f, encoding="utf-8", xml_declaration=False)


def read_notes(path):
    root = ET.parse(path).getroot()
    notes = []
    for note in root.iter("note"):
        pitch = note.find("pitch")
        notes.append((pitch.findtext("step"), int(pitch.findtext("octave"))))
    return notes


def note_to_freq(step, octave):
    semitones = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    midi = 12 * (octave + 1) + semitones[step]
    return 440.0 * 2 ** ((midi - 69) / 12)


if __name__ == "__main__":
    write_score(build_score(SCALE), OUTPUT_FILE)
    print(f"Wrote {OUTPUT_FILE}")

    parsed = read_notes(OUTPUT_FILE)
    assert parsed == SCALE, f"Round-trip mismatch: {parsed}"
    print("Round-trip OK. Notes and expected frequencies:")
    for step, octave in parsed:
        print(f"  {step}{octave}: {note_to_freq(step, octave):7.2f} Hz")

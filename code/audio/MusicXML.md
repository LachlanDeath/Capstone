# MusicXML Test Program (`build_musicXML.py`)

Here's what each part of `build_musicXML.py` does, from top to bottom.

## Constants

- **`OUTPUT_FILE`**: the name of the file the score is saved to.
- **`SCALE`**: the notes to write. Each note is a `(step, octave)` pair, such as `("C", 4)` for middle C.
- **`DIVISIONS`**: how many time units make up one quarter note. MusicXML measures every note length in these units. With `1`, a quarter note has a duration of 1, a half note 2, and so on. To use eighth notes you'd set it to `2` and give each eighth a duration of 1.
- **`BEATS_PER_MEASURE`**: beats per bar, set for 4/4 time. It also controls how many notes go in each bar.

## `build_score(notes)`

This builds the score in memory as a tree of XML elements. `ET.Element` makes the root element, and `ET.SubElement(parent, tag)` adds a child under a parent.

The tree looks like this:

```
score-partwise              root: score is organised part → measure → note
├── part-list               lists every instrument in the score
│   └── score-part id=P1    declares part "P1"
│       └── part-name       display name ("Piano")
└── part id=P1              the music for P1 (the ids must match)
    ├── measure 1
    │   ├── attributes      only in the first bar
    │   │   ├── divisions   time units per quarter note
    │   │   ├── key/fifths  key signature as sharps (+) or flats (-); 0 = C major
    │   │   ├── time        beats / beat-type (4/4)
    │   │   └── clef        sign G on line 2 = treble clef
    │   ├── note ×4
    │   │   ├── pitch       step (letter) + octave
    │   │   ├── duration    length in divisions, used for timing and playback
    │   │   └── type        how the note is drawn ("quarter")
    └── measure 2 ...
```

How the loop works:

- `range(0, len(notes), BEATS_PER_MEASURE)` steps through the notes four at a time, and each step makes a new `<measure>`.
- `i // BEATS_PER_MEASURE + 1` works out the bar number: 1, 2, and so on.
- The `<attributes>` block only goes in bar 1, because these settings stay in effect until something changes them.
- `notes[i:i + BEATS_PER_MEASURE]` takes the four notes for the current bar.

`duration` and `type` are separate on purpose. `duration` is the actual length in time, which is what matters for audio comparison. `type` only controls how the note looks on the page.

The function returns the tree wrapped in an `ET.ElementTree`, which is the form that can be written to a file.

## `write_score(tree, path)`

This saves the tree to disk.

- `ET.indent` adds line breaks and spaces so the file is easy to read. It doesn't change what the file means.
- It writes two header lines first:
  - the **XML declaration** (`<?xml ...?>`), which states the file's character encoding
  - the **DOCTYPE**, which tells programs like MuseScore that this is a MusicXML 4.0 file in partwise layout
- `tree.write(..., xml_declaration=False)` then writes the rest. The flag stops it from adding a second declaration, since one was already written by hand.

## `read_notes(path)`

This reads a MusicXML file and turns it back into a list of `(step, octave)` pairs.

- `ET.parse(path).getroot()` loads the file and returns the `<score-partwise>` element.
- `root.iter("note")` finds every `<note>` at any depth, across all bars, in the order they appear.
- `findtext("step")` / `findtext("octave")` get the text inside those child elements, and the octave is converted to an `int`.

One limitation: rests are `<note>` elements with no `<pitch>`, so `pitch` would be `None` and the code would crash. Sharps and flats (`<alter>`) are also ignored for now. Neither comes up in this scale, but a real piece will have both.

## `note_to_freq(step, octave)`

This converts a note name to its frequency in Hz, using standard tuning (A4 = 440 Hz).

1. `semitones` gives how far each letter is above C (C=0, D=2, E=4, F=5, ...). E to F and B to C are only a semitone apart, which is why the numbers aren't evenly spaced.
2. `midi = 12 * (octave + 1) + semitones[step]` gives the note's MIDI number. C4 comes out as 60 and A4 as 69.
3. `440 * 2 ** ((midi - 69) / 12)`: each semitone multiplies the frequency by 2^(1/12), and 12 semitones (one octave) doubles it.

These are the target frequencies to compare against YIN pitch estimates.

## `if __name__ == "__main__":`

This block only runs when the file is run directly (`python3 test.py`). It doesn't run if another script imports the file, so the functions can be reused elsewhere.

1. It builds the score and writes it to disk.
2. It reads the file back and uses `assert` to check the notes match `SCALE`. If they don't, the program stops with an error.
3. It prints each note with its expected frequency. `:7.2f` formats the number to two decimal places, padded to 7 characters wide.

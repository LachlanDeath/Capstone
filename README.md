## A Visual and Audio Processing System for Real-Time Cello Posture and Performance Feedback

**MREN 403 Capstone** · Lachlan Death, Cohen Little, Max Sloan, Christian Vandermeer

**Design Goal:**
A real-time feedback system for cello students. It combines two subsystems and shows their output on a display next to the player's live video feed:

- **Video:** a camera-based vision system that checks posture and bow position.
- **Audio:** real-time pitch detection (YIN-based) that identifies the note being played and how in tune it is.

## How to read this repository

This repository holds the code used to develop, test and run the cello tutor. It is split into two directories, `audio/` and `video/`. The subsystems are still in development and are not yet integrated. To run either one, see the README inside its folder.

There are currently two main development branches: `pitch-detection` and `mediapipe-baseline`. Because they are under active development, they may contain incomplete code, depend on an editor's local files, or lack documentation. You're welcome to look through them, but only expect code explicitly marked as complete to run. all code in the main directory if functional and if you follow the dedicated READMEs in both directories you will be able to run the code. 

All non-code work is kept in the project Google Drive: **https://drive.google.com/drive/folders/1mII-Jv8q8QVNS6psOojhaKh2EvGQKYGC?usp=drive_link**

### Recommended reading order

1. **Project proposal:** in the `deliverables` folder. Start here.
2. **Design documents:** the `design_documents` folder holds the calculations behind our design decisions, plus flowcharts of the two subsystems.
3. **Results:** the `results` folder holds the test results so far and is a good marker of progress.
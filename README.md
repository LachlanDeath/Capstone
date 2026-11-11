## A Visual and Audio Processing System for Real-Time Cello Posture and Performance Feedback

**MREN 403 Capstone** · Lachlan Death, Cohen Little, Max Sloan, Christian Vandermeer

A real-time feedback system for cello students. It combines two subsystems and shows their output on a display next to the player's live video feed:

- **Video:** a camera-based vision system (MediaPipe) that checks posture and bow position.
- **Audio:** real-time pitch detection (YIN-based) that identifies the note being played and how in tune it is.

---

## Contents

- [Repository Structure](#repository-structure)
- [How to Read This Portfolio](#how-to-read-this-portfolio)
- [Getting Started](#getting-started)

---

## Repository Structure

```
capstone_repo/
├── deliverables/     Formal submissions (project proposal)
├── workshop/         Working documents: ideas, requirements, MVP, risks, safety, stakeholders, assignments
├── meetings/
│   ├── notes/        Weekly meeting notes
│   └── presentations/  Presentation slides
├── design/
│   ├── audio/        Flowchart, subsystem breakdown, hardware requirements and datasheets
│   └── video/        Methodology and theory, research ethics board (GREB) application
├── code/
│   ├── audio/        Pitch detection notes and code
│   └── video/        MediaPipe pose and hand tracking
├── references/
│   ├── audio/        Pitch detection papers and links
│   └── video/        Posture feedback papers and literature review
├── calculations/     Engineering calculations
├── data/             Recorded data sets (not tracked in git)
└── requirements.txt  Python dependencies
```

---

## How to Read This Portfolio

| Step | What to read | Why |
|---|---|---|
| 1 | [`deliverables/MREN-403-Project-Proposal.pdf`](deliverables/MREN-403-Project-Proposal.pdf) | Project scope and goals |
| 2 | [`workshop/Requirements.md`](workshop/Requirements.md), [`workshop/Min_viable_product.md`](workshop/Min_viable_product.md) | What the system must do |
| 3 | [`meetings/notes/`](meetings/notes/) | Week-by-week progress |
| 4 | [`design/audio/`](design/audio/), [`design/video/`](design/video/) | Subsystem design |
| 5 | [`references/`](references/) | Research behind the design decisions |
| 6 | [`code/`](code/) | Implementation |

---

## Getting Started

> **Note:** MediaPipe requires **Python 3.11 or 3.12**. It does not support Python 3.14.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

To run the video demo, follow the full setup (including model downloads) in [`code/video/mediapipe/README.md`](code/video/mediapipe/README.md).

# Cello Posture Tutor

This portfolio prototype uses a webcam to compare a cellist's visible upper-body posture with a reference pose. It overlays the target as a cyan "ghost" and gives simple screen-direction hints when detected elbows, wrists, or the cello scroll are far from the target. MediaPipe estimates the body pose locally; optional cello and bow keypoints come from a Roboflow workflow.

## What is in this submission

| Path | Purpose |
| --- | --- |
| `video/code/prototypes/cello_tutor_1.py` | **Live tutor entry point.** Opens the webcam, draws the body pose and target, and optionally tracks cello/bow points. |
| `video/code/prototypes/cello_tutor_1_session.py` | Recording version of the live tutor. Saves the annotated session video and a per-frame CSV log. |
| `video/code/tools/capture_images.py` | Captures still webcam frames for the dataset. |
| `video/code/tools/pose_cello_image.py` | Adds body, hand, cello, and bow landmarks to one selected image. |
| `video/code/roboflow_workflow.py` | Sends an image to Roboflow and parses its cello/bow keypoints; also has an offline parser self-test. |
| `video/code/main.py` | **Not the tutor.** Only imports MediaPipe and prints its version as a minimal installation check. |
| `video/code/pose_landmarker_lite.task`, `video/code/hand_landmarker.task` | MediaPipe model assets. They are ignored by Git and must be downloaded from Gdrive/code for code to work. |
| `video/dataset/` | Dataset material, including `MaxRoom1/raw/`, `Youtube/`, and `labels.json` with image keypoint labels. |
| `requirements.txt` (repository root) | Pinned Python dependencies for this project. |
| `.gitignore` | Repository-root rules that exclude local model assets (`*.task`), virtual environments, secrets, and generated video files. |
| `video/MEDIAPIPE_SETUP.md` | Earlier MediaPipe setup notes. Its example paths are from the previous layout; use the commands in this README for this submission. |

## Setup (Windows PowerShell)

Run these commands from the repository root. Python 3.11 or 3.12 is recommended by the setup notes.

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\requirements.txt
Set-Location .\video\code
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" -OutFile .\pose_landmarker_lite.task
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -OutFile .\hand_landmarker.task
Set-Location ..\..
```

The `.task` files are local downloads, not submitted/tracked files. If PowerShell blocks virtual-environment activation, allow it for this terminal only, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Cello/bow detection requires a Roboflow API key with access to the configured workflow. Set it in the current PowerShell session; do not put a real key in source code or commit it:

```powershell
$env:ROBOFLOW_API_KEY = "paste-your-key-here"
```

The default tutor sends camera frames to Roboflow for cello/bow inference, at up to five requests per second and 3,000 requests per run. That service requires internet access, and each inference consumes account credits. The body pose runs locally. To use the tutor without cloud cello tracking or an API key, add `--no-cello`.

## Run the tools

From the repository root, start the **live tutor**:

```powershell
python .\video\code\prototypes\cello_tutor_1.py
```

Press **G** to show/hide the target overlay and **Q** to quit. The webcam view is mirrored. Both ears must be visible to align the target. For body-only tracking, run `python .\video\code\prototypes\cello_tutor_1.py --no-cello`.

To record a session instead, run:

```powershell
python .\video\code\prototypes\cello_tutor_1_session.py
```

Recording starts automatically. Press **G** to toggle the target and **Q** to stop and save. An MP4 and per-frame CSV are written to `video/code/prototypes/recordings/` with timestamped names. The session stops after 10 minutes; if the video encoder cannot open, the script reports that no video was recorded, while the CSV log is still written.

To create a custom reference pose from a clear photo (with both ears visible), use:

```powershell
python .\video\code\prototypes\cello_tutor_1.py --make-reference "C:\path\to\reference.jpg" --bow-side left
```

The generated `reference_posture.json` is saved beside the script and loaded automatically on the next run. `--bow-side` describes which side of the photo shows the reference player's bow hand. Creating it requires a Roboflow key for cello/bow points.

For **webcam image capture**, change to `video` first so the script saves into the submitted dataset folder:

```powershell
Set-Location .\video
python .\code\tools\capture_images.py
```

Press **Space** to save one unmirrored frame, **R** to toggle automatic capture (one image every 0.5 seconds), and **Q** to quit. New numbered JPGs go into `video/dataset/raw/`.

For a **single-image landmarking** run, provide the image path from the repository root:

```powershell
python .\video\code\tools\pose_cello_image.py "C:\path\to\photo.jpg"
```

The tool also opens a file picker when no path is supplied. It saves `<image-name>_landmarked.jpg` next to the source image and displays a preview; press any key to close it. It requires both MediaPipe models and a Roboflow key (one cloud inference per image).

To check the workflow response parser without a key, webcam, or network request:

```powershell
python .\video\code\roboflow_workflow.py --selftest
```

## Scope and caveats

The overlay aligns a 2D reference to the player's face and scales it; it is not 3D motion analysis or a complete measure of cello technique. Landmark quality depends on camera angle, lighting, visibility, and movement. Roboflow detection can be delayed or unavailable, and its image inference uses network access and credits. The current prototype does not provide audio/pitch feedback, sheet-music alignment, user accounts, or a setup interface.

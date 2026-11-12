# Engineering Portfolio: Cello Posture Tutor

## Video processing summary

The Cello Posture Tutor is a computer-vision prototype that gives a cello player a visual reference while they practise.

## Logic - code/video/prototypes/cello_tutor_1

1. The tutor opens the computer's default webcam
2. MediaPipe's Pose Landmarker estimates the player's nose, ears, shoulders, elbows, and wrists
3. When cello tracking is enabled, a background worker sends a resized camera frame to the Roboflow workflow. The response provides keypoints, currently the cello scroll, bridge, and bow tip. The workflow's image coordinates are scaled back to the webcam frame.
4. The target posture—which was established by running the model on an experienced cellist to extract reference keypoints—is anchored at the player's ears and scaled. It is mirrored when needed.
5. The program draws the target posture, detected body skeleton of player (detected posture), and available cello/bow points (cello posture). It compares elbows, wrists, and the scroll with the target and shows hints when their difference exceeds a threshold.
6. In the interactive window, **G** toggles the target overlay and **Q** exits.

## Map

| Path | Purpose |
| --- | --- |
| `code/video/prototypes/cello_tutor_1.py` | Main live tutor prototype: webcam loop, MediaPipe pose, optional cello tracking, target overlay, and hints. |
| `code/video/prototypes/cello_tutor_1_session.py` | Session-recording version. Saves annotated video and frame-by-frame CSV under `code/video/prototypes/recordings/`. |
| `code/video/roboflow_workflow.py` | Roboflow HTTP client, API-key lookup, retry/error handling, and keypoint parser. |
| `code/video/tools/pose_cello_image.py` | Single-image body, hand, cello, and bow landmarking tool. |
| `code/video/tools/capture_images.py` | Webcam image-capture utility. |
| `code/video/main.py` | Minimal MediaPipe import/version check; not the tutor entry point. |
| `code/video/tools/archive tools/` | Archived pose, guitar, and diagnostic prototypes. |
| `dataset/labels.json` | Tracked cello/bow keypoint labels. The image files are not tracked in the current 1.2 branch. |
| `README.md` | Older setup notes; some commands refer to the previous folder layout. |
| `requirements.txt` | Python dependency pins. |
| `.gitignore` | Excludes virtual environments, local model assets, videos, and generated files. |

## Main implementation details

### Body pose and target alignment

The tutor uses MediaPipe's upper-body landmark indices for the nose, ears, shoulders, elbows, and wrists. The ear midpoint provides an anchor shared by the reference and live player; face width provides a scale factor so the reference changes size with the player in frame. A side swap handles mirrored views and different bow sides. Exponential smoothing is applied to scale and anchor between frames. If the ears cannot be detected, the ghost and its hints are withheld rather than aligning an unreliable target.

### Cello/bow detection and data flow

`roboflow_workflow.py` reads `ROBOFLOW_API_KEY` from the process environment rather than storing credentials in source. It JPEG-encodes each BGR frame, sends it to the workflow over HTTPS, and normalizes the workflow response into named keypoints with confidence values and image dimensions. The tutor filters low-confidence keypoints, converts them back to the original frame size, and draws the scroll-to-bridge line and bow-tip connection when enough points are available.

The live tutor caps inference at five requests per second and 3,000 requests per run. Each request uses Roboflow credits and sends an image outside the computer to the service, so cello tracking requires network access, workflow access, and a valid key.

## Misc

For recording, use `cello_tutor_1_session.py` in place of `cello_tutor_1.py`. For a still image, run `python mediapipe\code\tools\pose_cello_image.py path\to\photo.jpg`; it requires the hand model and Roboflow key as well as the pose model. A custom reference can be created with `python mediapipe\code\prototypes\cello_tutor_1.py --make-reference path\to\reference.jpg --bow-side left`.

## Current scope and limitations

- The tutor compares a 2D camera view with a single reference pose; it does not reconstruct 3D motion or account for every valid variation in body shape, cello setup, or playing technique.
- The reference is anchored and scaled using the face or shoulders, not calibrated using camera geometry. Occlusion, lighting, camera placement, motion blur, and mirrored orientation can reduce landmark quality.
- Cello and bow detection is a separate cloud inference path. It may be delayed or unavailable when the network, API key, workflow, or confidence threshold prevents a usable result.
- There is no pitch/note recognition, sheet-music alignment, sound-quality feedback, user account, graphical setup flow, or haptic output in the current implementation.
- The repository includes data, labels, and YOLO experiment results, but the live tutor does not train a model at startup and does not use a local YOLO weights file.

The intended next product step is to validate the visual feedback with cello teachers and students, then decide which additions (for example audio feedback) are supported by evidence and fit the project scope.

## Setup
1. `python -m venv .venv`
2. Activate it
3. `pip install -r requirements.txt`
4. Move into mediapipe directory 
5. `cd mediapipe`
6. Download the pose and hand models:
```bash
   curl -o pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
   curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```
1. `python pose_demo.py`
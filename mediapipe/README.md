## Setup
1. `cd mediapipe`
2. `python -m venv mediapipe_env`
3. Activate it
4. `pip install -r requirements.txt`
5. Download the pose and hand models:
```bash
   curl -o pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
   curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```
6. `python pose_demo.py`
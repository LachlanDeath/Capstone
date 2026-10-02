## Intention
Setup mediapipe working virtual environment, and run test program.
## Setup

### 1. Navigate to the project folder

Make sure you're in the `capstone_repo` directory.

**Mac/Linux:**
```bash
cd path/to/capstone_repo
```

**Windows (PowerShell):**
```powershell
cd C:\path\to\capstone_repo
```

### 2. Create a virtual environment

**Mac/Linux:**
```bash
python3 -m venv .venv
```

**Windows:**
```powershell
python -m venv .venv
```

### 3. Activate it

**Mac/Linux:**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\activate
```

**Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

Your prompt should now start with `(.venv)`.

### 4. Install the required Python libraries

```bash
pip install -r requirements.txt
```

(Same command on all platforms, once the venv is activated.)

### 5. Move into the `mediapipe` directory

```bash
cd video/mediapipe
```

### 6. Download the pose and hand models

**Mac/Linux:**
```bash
curl -o pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

**Windows (PowerShell):**
```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" -OutFile pose_landmarker_lite.task
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -OutFile hand_landmarker.task
```

If neither command works, open each URL directly in a browser, download the file, and place both `.task` files in the `mediapipe` directory.

### 7. Run the test program

```bash
python pose_demo.py
```

A live video window should open showing you with your joints tracked. Press `q` in that window to quit.

---

## Troubleshooting

- **`FileNotFoundError: Unable to open file at pose_landmarker_lite.task`** — you're running the script from the wrong folder. Make sure you're inside `mediapipe/` (step 5) before running `python pose_demo.py`.
- **`RuntimeError: Unable to open zip archive`** — the `.task` file didn't download correctly (likely 0 bytes or an HTML error page). Delete it and redownload using step 6.
- **Camera won't open / `Camera index out of range`** — no webcam detected at index 0. Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in the script.
- **Windows: `running scripts is disabled on this system`** when activating the venv — run this once, then retry step 3:
```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```
- **Requires Python 3.11 or 3.12** — MediaPipe does not yet support Python 3.14. Check your version with `python --version` before creating the venv.
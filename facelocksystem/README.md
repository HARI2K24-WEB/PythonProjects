Face lock system

Usage:
1. Install dependencies:

```powershell
& C:/Users/hari2/AppData/Local/Programs/Python/Python310/python.exe -m pip install -r d:/WEB/pyhton/facelocksystem/requirements.txt
```

2. Run the script:

```powershell
& C:/Users/hari2/AppData/Local/Programs/Python/Python310/python.exe d:/WEB/pyhton/facelocksystem/facelock.py
```

Controls (type at prompt):
- F : Register — opens camera; press 'F' in the camera window to capture and save your photo.
- L : Login — opens camera and compares live faces to the saved photo.
- Q : Quit

Notes:
- This implementation uses OpenCV's LBPH face recognizer (in `opencv-contrib-python`) and Haar cascades. It avoids `dlib` and `face_recognition`, so it's simpler to install on Windows.
- `CONFIDENCE_THRESHOLD` in `facelock.py` controls matching strictness (lower = stricter). Tune it for your environment.

Eye Monitor

This script uses your webcam to monitor whether your eyes are open or closed.

Behavior:
- If eyes are continuously closed for the configured duration (default 2.0s), the script will turn off the monitor.
- When eyes open again, the script will wake the monitor by simulating a small mouse move.

Install requirements:
```powershell
& C:/Users/hari2/AppData/Local/Programs/Python/Python310/python.exe -m pip install -r d:/WEB/pyhton/eye_monitor/requirements.txt
```

Run:
```powershell
& C:/Users/hari2/AppData/Local/Programs/Python/Python310/python.exe d:/WEB/pyhton/eye_monitor/eye_monitor.py
```

Notes and troubleshooting:
- The script uses MediaPipe Face Mesh; ensure `mediapipe` is installed and compatible with your Python version.
- `pyautogui` is used to simulate mouse movement for waking on Windows. It may require additional permissions.
- Turning the monitor off uses Windows WM_SYSCOMMAND; this may behave differently on other OSes. The code includes platform checks.
- If detection is unstable, try adjusting `--threshold` and `--duration` command-line flags.

Security and safety:
- This script controls your display; test carefully and ensure you have a way to recover control (keyboard/mouse) if needed.

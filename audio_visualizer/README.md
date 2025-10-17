Audio Visualizer

This script captures microphone input and displays a colorful real-time visualization of waveform and frequency bands.

Requirements:
- Python 3.8+
- Install dependencies:

```powershell
& C:/Users/hari2/AppData/Local/Programs/Python/Python310/python.exe -m pip install -r d:/WEB/pyhton/audio_visualizer/requirements.txt
```

Run:
```powershell
& C:/Users/hari2/AppData/Local/Programs/Python/Python310/python.exe d:/WEB/pyhton/audio_visualizer/audio_visualizer.py
```

Notes:
- If the visualization is empty or silent, check microphone permissions and that no other app is using the mic.
- You can adjust RATE and CHUNK for latency/accuracy trade-offs.
- The script uses matplotlib for the GUI; it is not optimized for high-performance visuals but is useful for quick experiments.

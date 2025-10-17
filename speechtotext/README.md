Speech-to-text utility

Controls (run from PowerShell/terminal):
- R : Start recording from default microphone
- S : Stop recording and transcribe speech to text (uses Google Web Speech API via speech_recognition)
- T : Translate the last transcription to Tamil (uses deep-translator)
- Q : Quit

Quick install:
Open PowerShell and run:

```powershell
& C:/Users/hari2/AppData/Local/Programs/Python/Python310/python.exe -m pip install -r d:/WEB/pyhton/speechtotext/requirements.txt
```

Notes:
- `speech_recognition` uses Google's Web Speech API by default which requires internet.
- On Windows the script reads single-key input using msvcrt; run from a normal PowerShell window (not some IDE consoles that don't forward key events).
- If you prefer local transcription without Google, consider installing `whisper` and updating `transcribe_wav` to use it.

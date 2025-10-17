import os
import sys
import wave
import time
import tempfile
import threading
from deep_translator import GoogleTranslator
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr

# Windows-friendly single-key input
if os.name == 'nt':
    import msvcrt

RATE = 16000
CHANNELS = 1

class Recorder:
    def __init__(self, rate=RATE, channels=CHANNELS):
        self.rate = rate
        self.channels = channels
        self.recording = False
        self.frames = []
        self._stream = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            print("Recording status:", status)
        # append float32 data
        self.frames.append(indata.copy())

    def start(self):
        if self.recording:
            print("Already recording")
            return
        self.frames = []
        self._stream = sd.InputStream(samplerate=self.rate, channels=self.channels, callback=self._callback)
        self._stream.start()
        self.recording = True
        print("Recording started...")

    def stop(self):
        if not self.recording:
            print("Not recording")
            return None
        self._stream.stop()
        self._stream.close()
        self.recording = False
        # concatenate frames
        import numpy as np
        data = np.concatenate(self.frames, axis=0)
        return data

    def save_wav(self, data, path):
        sf.write(path, data, self.rate)


def transcribe_wav(path):
    r = sr.Recognizer()
    with sr.AudioFile(path) as source:
        audio = r.record(source)
    try:
        print("Transcribing (Google Web Speech API)...")
        text = r.recognize_google(audio)
        print("Transcription:", text)
        return text
    except sr.RequestError as e:
        print("API unavailable or unresponsive; error:", e)
    except sr.UnknownValueError:
        print("Unable to recognize speech")
    return ""


def translate_text(text, target='ta'):
    if not text:
        print("No text to translate")
        return ""
    try:
        translated = GoogleTranslator(source='auto', target=target).translate(text)
        print(f"Translated ({target}):", translated)
        return translated
    except Exception as e:
        print("Translation error:", e)
        return ""

def get_key():
    if os.name == 'nt':
        while True:
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                try:
                    return ch.decode('utf-8')
                except:
                    return ''
            time.sleep(0.01)
    else:
        # Unix fallback
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    print("Controls: R=record start, S=stop and transcribe, T=translate to Tamil, Q=quit")
    recorder = Recorder()
    last_text = ""
    wav_path = None

    while True:
        print('\nPress a key: ', end='', flush=True)
        key = get_key().lower()
        print(key.upper())

        if key == 'r':
            try:
                recorder.start()
            except Exception as e:
                print("Failed to start recording:", e)

        elif key == 's':
            try:
                data = recorder.stop()
                if data is None:
                    continue
                fd, wav_path = tempfile.mkstemp(suffix='.wav')
                os.close(fd)
                recorder.save_wav(data, wav_path)
                print("Saved recording to:", wav_path)
                last_text = transcribe_wav(wav_path)
            except Exception as e:
                print("Error stopping/transcribing:", e)

        elif key == 't':
            if not last_text:
                print("No transcription available to translate. Press S to transcribe first.")
                continue
            translate_text(last_text, target='ta')

        elif key == 'q':
            print("Quitting...")
            break

        else:
            print("Unknown key. Use R, S, T, Q")

    # cleanup
    if wav_path and os.path.isfile(wav_path):
        try:
            os.remove(wav_path)
        except:
            pass

if __name__ == '__main__':
    main()

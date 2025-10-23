import cv2
import os
try:
    import mediapipe as mp
    HAVE_MEDIAPIPE = True
except Exception:
    mp = None
    HAVE_MEDIAPIPE = False
import time
import numpy as np
import platform
import ctypes
import argparse
from ctypes import wintypes
import threading
mp_face_mesh = None
if HAVE_MEDIAPIPE:
    try:
        mp_face_mesh = mp.solutions.face_mesh
    except Exception:
        # if anything goes wrong accessing solutions, disable MediaPipe usage
        mp_face_mesh = None
        HAVE_MEDIAPIPE = False

# Eye landmarks indices for MediaPipe Face Mesh (approximate)
# Using left/right eye outer and inner points to compute simple openness metric
LEFT_EYE_IDX = [33, 133, 159, 145]  # left eye: outer, inner, top, bottom (for MediaPipe)
RIGHT_EYE_IDX = [362, 263, 386, 374]

# Thresholds
EYE_CLOSED_THRESHOLD = 0.20  # normalized ratio below which eye is considered closed
CLOSE_DURATION_SECONDS = 2.0  # required continuous seconds eyes must be closed to turn off

# Monitor power constants for Windows
WM_SYSCOMMAND = 0x0112
SC_MONITORPOWER = 0xF170
HWND_BROADCAST = 0xFFFF


def monitor_off_windows():
    # send WM_SYSCOMMAND SC_MONITORPOWER with param 2 to turn off
    ctypes.windll.user32.PostMessageW(HWND_BROADCAST, WM_SYSCOMMAND, SC_MONITORPOWER, 2)


def wake_monitor_windows():
    # simulate mouse move
    import pyautogui
    x, y = pyautogui.position()
    pyautogui.moveTo(x+1, y)
    pyautogui.moveTo(x, y)


def eye_openness_ratio(landmarks, image_w, image_h, idxs):
    # compute vertical distance / horizontal distance as a simple openness measure
    outer = landmarks[idxs[0]]
    inner = landmarks[idxs[1]]
    top = landmarks[idxs[2]]
    bottom = landmarks[idxs[3]]
    outer_pt = np.array([outer.x * image_w, outer.y * image_h])
    inner_pt = np.array([inner.x * image_w, inner.y * image_h])
    top_pt = np.array([top.x * image_w, top.y * image_h])
    bottom_pt = np.array([bottom.x * image_w, bottom.y * image_h])
    horiz = np.linalg.norm(outer_pt - inner_pt)
    vert = np.linalg.norm(top_pt - bottom_pt)
    if horiz == 0:
        return 0.0
    return vert / horiz


def haar_eye_openness(eye_rect):
    """Compute a simple openness metric from an eye rectangle (x,y,w,h): use h/w."""
    x, y, w, h = eye_rect
    if w == 0:
        return 0.0
    return float(h) / float(w)


# Fullscreen blank overlay (safe alternative to powering off monitor)
class BlankOverlay:
    def __init__(self):
        self.thread = None
        self._stop_event = threading.Event()
        self.root = None

    def _run(self):
        try:
            import tkinter as tk
        except Exception:
            print('tkinter not available; cannot show blank overlay')
            return
        root = tk.Tk()
        self.root = root
        root.overrideredirect(True)
        root.attributes('-fullscreen', True)
        # ensure on top
        root.attributes('-topmost', True)
        root.config(bg='black')
        # periodically check stop event
        def poll():
            if self._stop_event.is_set():
                try:
                    root.destroy()
                except Exception:
                    pass
                return
            root.after(200, poll)
        root.after(200, poll)
        try:
            root.mainloop()
        except Exception:
            pass

    def show(self):
        if self.thread and self.thread.is_alive():
            return
        self._stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def hide(self):
        self._stop_event.set()
        # also try destroying root from main thread if available
        try:
            if self.root:
                self.root.destroy()
        except Exception:
            pass



def check_quit_key():
    try:
        if os.name == 'nt':
            import msvcrt
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch in (b'q', b'Q'):
                    return True
            return False
        else:
            import sys, select
            dr, dw, de = select.select([sys.stdin], [], [], 0)
            if dr:
                ch = sys.stdin.read(1)
                if ch in ('q', 'Q'):
                    return True
            return False
    except Exception:
        return False


def main(args):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Cannot open camera')
        return

    mode = getattr(args, 'mode', 'blank')
    overlay = None
    if mode == 'blank':
        overlay = BlankOverlay()

    closed_since = None
    monitor_off = False

    # Choose method: MediaPipe (preferred) if available, otherwise Haar cascades
    if HAVE_MEDIAPIPE:
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    else:
        # load Haar cascades for face + eyes
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        if face_cascade.empty():
            print('Warning: failed to load face Haar cascade; face detection may not work')
        if eye_cascade.empty():
            print('Warning: failed to load eye Haar cascade; eye detection may not work')

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            eye_open = False
            if HAVE_MEDIAPIPE and mp_face_mesh is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                if results is not None and getattr(results, 'multi_face_landmarks', None):
                    for lm in results.multi_face_landmarks:
                        lmks = lm.landmark
                        left_ratio = eye_openness_ratio(lmks, w, h, LEFT_EYE_IDX)
                        right_ratio = eye_openness_ratio(lmks, w, h, RIGHT_EYE_IDX)
                        ratio = (left_ratio + right_ratio) / 2.0
                        cv2.putText(frame, f"Eye ratio: {ratio:.2f}", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                        if ratio > EYE_CLOSED_THRESHOLD:
                            eye_open = True
                        break
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100,100))
                if len(faces) > 0:
                    x,y,fw,fh = faces[0]
                    roi = gray[y:y+fh, x:x+fw]
                    eyes = eye_cascade.detectMultiScale(roi)
                    if len(eyes) > 0:
                        # take first detected eye rect and compute openness
                        ex,ey,ew,eh = eyes[0]
                        ratio = haar_eye_openness((ex,ey,ew,eh))
                        cv2.putText(frame, f"Eye ratio(haar): {ratio:.2f}", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                        if ratio > EYE_CLOSED_THRESHOLD:
                            eye_open = True

            if eye_open:
                closed_since = None
                if monitor_off:
                    print('Eyes opened — waking monitor')
                    if platform.system() == 'Windows':
                        wake_monitor_windows()
                    monitor_off = False
            else:
                if closed_since is None:
                    closed_since = time.time()
                else:
                    elapsed = time.time() - closed_since
                    cv2.putText(frame, f"Closed for {elapsed:.1f}s", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    if elapsed >= CLOSE_DURATION_SECONDS and not monitor_off:
                        print('Eyes closed — turning monitor off')
                        if platform.system() == 'Windows':
                            monitor_off_windows()
                        monitor_off = True

            # If monitor is currently off, skip showing window and use console key polling
            if monitor_off:
                # still print a small status occasionally
                if int(time.time()) % 5 == 0:
                    print('Monitor is off; watching for eyes to open...')
                if check_quit_key():
                    break
            else:
                cv2.imshow('Eye Monitor', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

    finally:
        # cleanup MediaPipe resources if used
        try:
            if HAVE_MEDIAPIPE and 'face_mesh' in locals():
                face_mesh.close()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=EYE_CLOSED_THRESHOLD)
    parser.add_argument('--duration', type=float, default=CLOSE_DURATION_SECONDS)
    parser.add_argument('--mode', choices=['blank','power'], default='blank', help="'blank' shows a fullscreen black overlay; 'power' sends monitor power off command (Windows)")
    args = parser.parse_args()
    # set globals from args
    EYE_CLOSED_THRESHOLD = args.threshold
    CLOSE_DURATION_SECONDS = args.duration
    try:
        main(args)
    except KeyboardInterrupt:
        pass

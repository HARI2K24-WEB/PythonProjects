import cv2
import os
import time
import numpy as np
import platform
import ctypes
import threading
import logging

# --------------------------
# CONFIG / PATHS
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_IMAGE = os.path.join(BASE_DIR, "user.jpg")
MODEL_FILE = os.path.join(BASE_DIR, "lbph_model.yml")
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

FACE_SIZE = (100, 100)
CONFIDENCE_THRESHOLD = 70.0
FALLBACK_MSE_THRESHOLD = 1500.0

# Eye monitoring thresholds
EYE_CLOSED_THRESHOLD = 0.20
CLOSE_DURATION_SECONDS = 2.0

# MediaPipe setup
try:
    import mediapipe as mp
    HAVE_MEDIAPIPE = True
    mp_face_mesh = mp.solutions.face_mesh
except Exception:
    mp = None
    HAVE_MEDIAPIPE = False
    mp_face_mesh = None

# Eye landmark indices
LEFT_EYE_IDX = [33, 133, 159, 145]
RIGHT_EYE_IDX = [362, 263, 386, 374]

# Windows monitor off constants
WM_SYSCOMMAND = 0x0112
SC_MONITORPOWER = 0xF170
HWND_BROADCAST = 0xFFFF

# --------------------------
# UTILITY FUNCTIONS
# --------------------------

def mse(a, b):
    a = a.astype('float32')
    b = b.astype('float32')
    return np.mean((a - b) ** 2)

def ensure_recognizer():
    if not hasattr(cv2, 'face'):
        return None
    try:
        return cv2.face.LBPHFaceRecognizer_create()
    except Exception:
        logging.exception("LBPH recognizer not available")
        return None

def detect_faces(gray):
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return faces

# --------------------------
# FACE LOGIN SYSTEM
# --------------------------

def register_user():
    print("Register: Press 'F' to capture your face, 'Q' to cancel.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        display = frame.copy()
        cv2.putText(display, "Press F to capture, Q to cancel", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Register - Capture Face", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('f'):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray)
            if len(faces) == 0:
                print("No face detected. Try again.")
                continue
            x, y, w, h = max(faces, key=lambda r: r[2]*r[3])
            face_img = gray[y:y+h, x:x+w]
            try:
                face_resized = cv2.resize(face_img, FACE_SIZE)
            except Exception:
                face_resized = face_img
            recognizer = ensure_recognizer()
            if recognizer is not None:
                try:
                    recognizer.train([face_resized], np.array([1], dtype=np.int32))
                    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
                    recognizer.write(MODEL_FILE)
                    print("Saved successfully (LBPH model).")
                except Exception as e:
                    print(f"Failed to train/save model: {e}")
            else:
                os.makedirs(os.path.dirname(USER_IMAGE), exist_ok=True)
                cv2.imwrite(USER_IMAGE, face_resized)
                print("Saved user face image (fallback).")
            cv2.waitKey(500)
            break
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def login_user():
    recognizer = ensure_recognizer()
    have_model = False
    if recognizer is not None and os.path.isfile(MODEL_FILE):
        try:
            recognizer.read(MODEL_FILE)
            have_model = True
        except Exception:
            have_model = False
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return False
    verified = False
    print("Starting live verification. Press 'Q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray)
        message = "No face detected"
        color = (0, 0, 255)
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            try:
                face_resized = cv2.resize(face_img, FACE_SIZE)
            except Exception:
                face_resized = face_img
            if have_model:
                try:
                    label, confidence = recognizer.predict(face_resized)
                    if confidence < CONFIDENCE_THRESHOLD and label == 1:
                        message = "Verified! Login successful."
                        color = (0, 255, 0)
                        verified = True
                    else:
                        message = "Not a valid user"
                        color = (0, 0, 255)
                except Exception:
                    continue
            else:
                if os.path.isfile(USER_IMAGE):
                    stored = cv2.imread(USER_IMAGE, cv2.IMREAD_GRAYSCALE)
                    if stored is not None:
                        try:
                            stored_resized = cv2.resize(stored, FACE_SIZE)
                        except Exception:
                            stored_resized = stored
                        if mse(stored_resized, face_resized) < FALLBACK_MSE_THRESHOLD:
                            message = "Verified (fallback)"
                            color = (0, 255, 0)
                            verified = True
                        else:
                            message = "Not a valid user (fallback)"
        cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Live Verification", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or verified:
            break
    cap.release()
    cv2.destroyAllWindows()
    return verified

# --------------------------
# EYE MONITOR SYSTEM
# --------------------------

def monitor_off_windows():
    ctypes.windll.user32.PostMessageW(HWND_BROADCAST, WM_SYSCOMMAND, SC_MONITORPOWER, 2)

def wake_monitor_windows():
    import pyautogui
    x, y = pyautogui.position()
    pyautogui.moveTo(x+1, y)
    pyautogui.moveTo(x, y)

def eye_openness_ratio(landmarks, image_w, image_h, idxs):
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
    if horiz == 0: return 0.0
    return vert / horiz

def start_eye_monitor():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    monitor_off = False
    closed_since = None

    if HAVE_MEDIAPIPE:
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    else:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            h, w = frame.shape[:2]
            eye_open = False

            if HAVE_MEDIAPIPE and mp_face_mesh is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                if results.multi_face_landmarks:
                    for lm in results.multi_face_landmarks:
                        left_ratio = eye_openness_ratio(lm.landmark, w, h, LEFT_EYE_IDX)
                        right_ratio = eye_openness_ratio(lm.landmark, w, h, RIGHT_EYE_IDX)
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
                        ex,ey,ew,eh = eyes[0]
                        if ew != 0:
                            ratio = eh / ew
                            cv2.putText(frame, f"Eye ratio(haar): {ratio:.2f}", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                            if ratio > EYE_CLOSED_THRESHOLD:
                                eye_open = True

            if eye_open:
                closed_since = None
                if monitor_off:
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
                        if platform.system() == 'Windows':
                            monitor_off_windows()
                        monitor_off = True

            cv2.imshow('Eye Monitor', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        if HAVE_MEDIAPIPE and 'face_mesh' in locals():
            face_mesh.close()
        cap.release()
        cv2.destroyAllWindows()

# --------------------------
# MAIN MENU
# --------------------------

def main():
    while True:
        print("\nFace Lock + Eye Monitor System")
        print("F - Register Face")
        print("L - Login & Start Eye Monitor")
        print("Q - Quit")
        choice = input("Enter command: ").strip().lower()
        if choice == 'f':
            register_user()
        elif choice == 'l':
            if login_user():
                print("Login verified. Starting eye monitor...")
                start_eye_monitor()
            else:
                print("Login failed. Cannot start monitor.")
        elif choice == 'q':
            print("Exiting...")
            break
        else:
            print("Unknown command")

if __name__ == '__main__':
    main()

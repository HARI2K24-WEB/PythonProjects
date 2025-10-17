import cv2
import os
import time
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_IMAGE = os.path.join(BASE_DIR, "user.jpg")
MODEL_FILE = os.path.join(BASE_DIR, "lbph_model.yml")
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# LBPH confidence threshold: lower is better. Tune this for your environment.
CONFIDENCE_THRESHOLD = 70.0


def ensure_recognizer():
    # Create LBPH recognizer (requires opencv-contrib)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    return recognizer


def detect_faces(gray):
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return faces


def register_user():
    print("Register: Press 'F' in the camera window to capture your face. Press 'Q' to cancel.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        display = frame.copy()
        cv2.putText(display, "Press F to capture, Q to cancel", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Register - Capture Face", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('f'):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray)
            if len(faces) == 0:
                print("No face detected. Try again.")
                continue
            # take the largest face
            x, y, w, h = max(faces, key=lambda r: r[2]*r[3])
            face_img = gray[y:y+h, x:x+w]
            # Save user image
            cv2.imwrite(USER_IMAGE, frame)
            # Train recognizer on this single sample
            recognizer = ensure_recognizer()
            recognizer.train([face_img], np.array([1]))
            recognizer.write(MODEL_FILE)
            print("Saved successfully.")
            cv2.putText(frame, "Saved successfully", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
            cv2.imshow("Register - Capture Face", frame)
            cv2.waitKey(1000)
            break
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def login_user():
    if not os.path.isfile(MODEL_FILE):
        print("No registered user found. Press 'F' to register first.")
        return
    recognizer = ensure_recognizer()
    recognizer.read(MODEL_FILE)
    print("Starting live verification. Press 'Q' in the camera window to stop.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    verified = False
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
            # Resize face to the size used for training if necessary
            try:
                label, confidence = recognizer.predict(face_img)
            except Exception:
                # If prediction fails due to size, resize
                face_small = cv2.resize(face_img, (100, 100))
                label, confidence = recognizer.predict(face_small)
            if confidence < CONFIDENCE_THRESHOLD and label == 1:
                message = "Verified successfully - Login successfully"
                color = (0, 255, 0)
                verified = True
            else:
                message = "Not a valid user"
                color = (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Live Verification", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if verified:
            cv2.waitKey(1000)
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    print("Face lock system controls:")
    print("F - Register (capture photo)")
    print("L - Login (start live face verification)")
    print("Q - Quit")
    while True:
        key = input("Enter command (F/L/Q): ").strip().lower()
        if key == 'f':
            register_user()
        elif key == 'l':
            login_user()
        elif key == 'q':
            print("Quitting")
            break
        else:
            print("Unknown command")


if __name__ == '__main__':
    main()

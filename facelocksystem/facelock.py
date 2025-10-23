import cv2
import os
import time
import numpy as np
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_IMAGE = os.path.join(BASE_DIR, "user.jpg")
MODEL_FILE = os.path.join(BASE_DIR, "lbph_model.yml")
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# LBPH confidence threshold: lower is better. Tune this for your environment.
CONFIDENCE_THRESHOLD = 70.0
# size to which face images are resized for training and prediction
FACE_SIZE = (100, 100)
# Fallback settings when cv2.face (LBPH) is not available
FALLBACK_RECOGNITION = True
# Mean squared error threshold for fallback match (lower = more similar). Tune this.
FALLBACK_MSE_THRESHOLD = 1500.0


def ensure_recognizer():
    # Create LBPH recognizer (requires opencv-contrib). If unavailable, return None
    if not hasattr(cv2, 'face'):
        # Caller can decide to use fallback recognition
        return None
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        return recognizer
    except Exception as e:
        logging.exception('Failed to create LBPH recognizer: %s', e)
        return None


def mse(a, b):
    """Mean squared error between two same-sized images."""
    a = a.astype('float32')
    b = b.astype('float32')
    err = np.mean((a - b) ** 2)
    return err


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
            # Resize to consistent size
            try:
                face_resized = cv2.resize(face_img, FACE_SIZE)
            except Exception:
                face_resized = face_img

            # Save cropped face image (not the full frame) for debugging
            try:
                cv2.imwrite(USER_IMAGE, face_resized)
            except Exception as e:
                print(f"Failed to save user image: {e}")

            # Train recognizer on this single sample, or save face for fallback
            recognizer = ensure_recognizer()
            if recognizer is not None:
                try:
                    # labels must be int32
                    recognizer.train([face_resized], np.array([1], dtype=np.int32))
                    # ensure model dir exists
                    try:
                        os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
                    except Exception:
                        pass
                    recognizer.write(MODEL_FILE)
                    print("Saved successfully (LBPH model).")
                    cv2.putText(frame, "Saved successfully", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
                    cv2.imshow("Register - Capture Face", frame)
                    cv2.waitKey(1000)
                    break
                except Exception as e:
                    print(f"Failed to train or save model: {e}")
                    logging.exception('Train error')
                    continue
            else:
                # fallback: save the face image for simple comparison later
                try:
                    os.makedirs(os.path.dirname(USER_IMAGE), exist_ok=True)
                    cv2.imwrite(USER_IMAGE, face_resized)
                    print("Saved user face image for fallback recognition.")
                    cv2.putText(frame, "Saved successfully", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
                    cv2.imshow("Register - Capture Face", frame)
                    cv2.waitKey(1000)
                    break
                except Exception as e:
                    print(f"Failed to save fallback user image: {e}")
                    logging.exception('Fallback save error')
                    continue
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def login_user():
    if not os.path.isfile(MODEL_FILE):
        print("No registered user found. Press 'F' to register first.")
        return
    recognizer = ensure_recognizer()
    # If recognizer exists and model file exists, try to load it
    have_model = False
    if recognizer is not None and os.path.isfile(MODEL_FILE):
        try:
            recognizer.read(MODEL_FILE)
            have_model = True
        except Exception as e:
            logging.exception('Failed to read LBPH model: %s', e)
            have_model = False
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
                face_resized = cv2.resize(face_img, FACE_SIZE)
            except Exception:
                face_resized = face_img

            if have_model:
                try:
                    label, confidence = recognizer.predict(face_resized)
                except Exception as e:
                    logging.exception('Predict error: %s', e)
                    continue
                if confidence < CONFIDENCE_THRESHOLD and label == 1:
                    message = "Verified successfully - Login successfully"
                    color = (0, 255, 0)
                    verified = True
                else:
                    message = "Not a valid user"
                    color = (0, 0, 255)
            else:
                # Fallback comparison using simple MSE against USER_IMAGE
                if os.path.isfile(USER_IMAGE):
                    try:
                        stored = cv2.imread(USER_IMAGE, cv2.IMREAD_GRAYSCALE)
                        if stored is not None:
                            try:
                                stored_resized = cv2.resize(stored, FACE_SIZE)
                            except Exception:
                                stored_resized = stored
                            err = mse(stored_resized, face_resized)
                            if err < FALLBACK_MSE_THRESHOLD:
                                message = "Verified (fallback) - Login successful"
                                color = (0, 255, 0)
                                verified = True
                            else:
                                message = "Not a valid user (fallback)"
                                color = (0, 0, 255)
                        else:
                            message = 'Stored user image unreadable'
                            color = (0, 0, 255)
                    except Exception as e:
                        logging.exception('Fallback compare error: %s', e)
                        message = 'Fallback compare failed'
                        color = (0, 0, 255)
                else:
                    message = 'No trained model or stored user image found.'
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

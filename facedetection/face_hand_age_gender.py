import cv2
import numpy as np
import mediapipe as mp

# Load models
FACE_PROTO = "deploy.prototxt"
FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
AGE_PROTO = "age_deploy.prototxt"
AGE_MODEL = "age_net.caffemodel"
GENDER_PROTO = "gender_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"

# Load DNNs
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)

# Labels
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# Mediapipe setup for hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,  # allow 2 hands
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Start camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # --- FACE DETECTION ---
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype(int)

            # Crop face
            face = frame[y1:y2, x1:x2].copy()
            if face.size == 0:
                continue

            blob_face = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                              (78.4263377603,
                                               87.7689143744,
                                               114.895847746),
                                              swapRB=False)

            # Gender
            gender_net.setInput(blob_face)
            gender = GENDER_LIST[gender_net.forward()[0].argmax()]

            # Age
            age_net.setInput(blob_face)
            age = AGE_BUCKETS[age_net.forward()[0].argmax()]

            label = f"{gender}, {age}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2)

    # --- HAND DETECTION ---
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks, hand_handedness in zip(
            hand_results.multi_hand_landmarks,
            hand_results.multi_handedness
        ):
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0),
                                       thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            # Label Left/Right
            handedness = hand_handedness.classification[0].label
            x = int(hand_landmarks.landmark[0].x * w)
            y = int(hand_landmarks.landmark[0].y * h) - 10
            cv2.putText(frame, handedness, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 0, 0), 2)

    cv2.imshow("Face + Hand + Age/Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()




# new
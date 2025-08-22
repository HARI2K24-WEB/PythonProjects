import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from collections import deque

# Screen size
screen_w, screen_h = pyautogui.size()

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Queue to smooth mouse movement
smooth_buffer = deque(maxlen=5)  # keep last 5 points

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Index & Thumb
            x_index = int(hand_landmarks.landmark[8].x * w)
            y_index = int(hand_landmarks.landmark[8].y * h)
            x_thumb = int(hand_landmarks.landmark[4].x * w)
            y_thumb = int(hand_landmarks.landmark[4].y * h)

            # Map to screen coords
            screen_x = int((x_index / w) * screen_w)
            screen_y = int((y_index / h) * screen_h)

            # Save coords into buffer
            smooth_buffer.append((screen_x, screen_y))

            # Average position (smoothing)
            avg_x, avg_y = np.mean(smooth_buffer, axis=0).astype(int)

            # Move mouse with smoothing
            pyautogui.moveTo(avg_x, avg_y)

            # Draw marker
            cv2.circle(frame, (x_index, y_index), 10, (0, 255, 255), -1)

            # Click when thumb + index close
            if abs(x_index - x_thumb) < 40 and abs(y_index - y_thumb) < 40:
                cv2.putText(frame, "Click", (x_index, y_index - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                pyautogui.click()

    cv2.imshow("Finger Mouse Control (Stable)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

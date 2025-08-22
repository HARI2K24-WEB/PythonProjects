import cv2
import mediapipe as mp

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Finger tip landmark IDs
finger_tips = [4, 8, 12, 16, 20]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror view
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    left_count, right_count = 0, 0

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = hand_handedness.classification[0].label  # 'Left' or 'Right'
            lm_list = []
            h, w, c = frame.shape
            for id, lm in enumerate(hand_landmarks.landmark):
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            # Count fingers
            fingers = 0
            if lm_list:
                # Thumb
                if label == "Right":
                    if lm_list[finger_tips[0]][0] < lm_list[finger_tips[0] - 1][0]:
                        fingers += 1
                else:
                    if lm_list[finger_tips[0]][0] > lm_list[finger_tips[0] - 1][0]:
                        fingers += 1

                # Other fingers
                for tip in finger_tips[1:]:
                    if lm_list[tip][1] < lm_list[tip - 2][1]:
                        fingers += 1

            if label == "Left":
                left_count = fingers
            else:
                right_count = fingers

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show counts and addition
    total = left_count + right_count
    cv2.putText(frame, f"Left: {left_count}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Right: {right_count}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Total: {total}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Finger Counter", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

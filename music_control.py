
from __future__ import annotations
import argparse
import time
from collections import deque
from math import atan2, degrees

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
HAND = mp_hands.HandLandmark


def landmark_to_np(landmarks, image_shape):
    h, w = image_shape[:2]
    pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks.landmark], dtype=np.float32)
    return pts


def fingers_extended(pts: np.ndarray) -> int:
    WRIST = HAND.WRIST
    TIP_IDS = [HAND.THUMB_TIP, HAND.INDEX_FINGER_TIP, HAND.MIDDLE_FINGER_TIP, HAND.RING_FINGER_TIP, HAND.PINKY_TIP]
    PIP_IDS = [HAND.THUMB_IP, HAND.INDEX_FINGER_PIP, HAND.MIDDLE_FINGER_PIP, HAND.RING_FINGER_PIP, HAND.PINKY_PIP]
    wrist = pts[WRIST]
    count = 0
    for tip_id, pip_id in zip(TIP_IDS, PIP_IDS):
        tip, pip = pts[tip_id], pts[pip_id]
        if np.linalg.norm(tip - wrist) > np.linalg.norm(pip - wrist) + 12:
            count += 1
    return count


class CircleAccumulator:
    def __init__(self, maxlen=30):
        self.points = deque(maxlen=maxlen)
        self.total_angle = 0.0

    def update(self, pt):
        self.points.append(pt)
        if len(self.points) >= 2:
            c = np.mean(self.points, axis=0)
            p1, p2 = self.points[-2] - c, self.points[-1] - c
            a1 = degrees(atan2(p1[1], p1[0]))
            a2 = degrees(atan2(p2[1], p2[0]))
            da = a2 - a1
            if da > 180: da -= 360
            if da < -180: da += 360
            self.total_angle += da

    def consume_steps(self, step_deg=300):
        steps = int(self.total_angle // step_deg)
        if steps != 0:
            self.total_angle -= steps * step_deg
        return steps


class SwipeDetector:
    def __init__(self, maxlen=8, min_speed=12):
        self.points = deque(maxlen=maxlen)
        self.min_speed = min_speed

    def update(self, pt):
        self.points.append(pt)

    def swipe(self):
        if len(self.points) < 2:
            return None
        dx = self.points[-1][0] - self.points[0][0]
        dy = self.points[-1][1] - self.points[0][1]
        dt = len(self.points)
        vx, vy = dx / dt, dy / dt
        if abs(vx) > abs(vy) and abs(vx) > self.min_speed:
            return "right" if vx > 0 else "left"
        return None


def draw_hud(frame, headline: str, sub: str = "", color=(0, 255, 0)):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = 10, 10, int(w*0.6), 90
    cv2.rectangle(frame, (x0, y0), (x1, y1), (45, 45, 45), thickness=-1)
    cv2.putText(frame, headline, (x0+12, y0+34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    if sub:
        cv2.putText(frame, sub, (x0+12, y0+66), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--mirror", action="store_true")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")

    hands = mp_hands.Hands(model_complexity=1, max_num_hands=1,
                           min_detection_confidence=0.6, min_tracking_confidence=0.5)

    circle = CircleAccumulator(maxlen=32)
    swipe = SwipeDetector(maxlen=8, min_speed=10)

    last_fist_time = 0.0
    toggle_cooldown = 0.8
    palm_hold_start = None

    status = "Show your hand"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                pts = landmark_to_np(hand, frame.shape)
                mp_drawing.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

                ext = fingers_extended(pts)
                centroid = np.mean(pts, axis=0)
                index_tip = pts[HAND.INDEX_FINGER_TIP]

                circle.update(index_tip)
                swipe.update(centroid)

                now = time.time()

                # Fist detect (debounced)
                if ext == 0 and (now - last_fist_time) > toggle_cooldown:
                    last_fist_time = now
                    status = "FIST"
                    print("Gesture: FIST")

                # Volume-like circle gesture
                steps = circle.consume_steps(step_deg=300)
                if steps != 0:
                    if steps > 0:
                        status = f"CIRCLE Clockwise (+{steps})"
                        print("Gesture: CIRCLE clockwise")
                    else:
                        status = f"CIRCLE Counter (-{-steps})"
                        print("Gesture: CIRCLE counterclockwise")

                # Swipe with open palm
                sw = swipe.swipe()
                if ext >= 4 and sw is not None:
                    status = f"SWIPE {sw.upper()}"
                    print(f"Gesture: SWIPE {sw}")

                # Open-palm hold (stillness)
                if ext >= 4:
                    if len(swipe.points) >= 6:
                        pts_arr = np.array(swipe.points)
                        motion = np.mean(np.linalg.norm(np.diff(pts_arr, axis=0), axis=1))
                    else:
                        motion = 999
                    if motion < 1.5:
                        palm_hold_start = palm_hold_start or now
                        if now - palm_hold_start > 2.0:
                            status = "PALM HOLD (~2s)"
                            print("Gesture: PALM HOLD")
                    else:
                        palm_hold_start = None
                else:
                    palm_hold_start = None

                draw_hud(frame, status, "q: quit | mirror: on" if args.mirror else "q: quit")
            else:
                draw_hud(frame, status, "q: quit")

            cv2.imshow("Gesture Viewer", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            hands.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

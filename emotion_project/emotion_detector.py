import time
import cv2
from deepface import DeepFace
import pygame
import sys
from datetime import datetime

MUSIC = {
    "happy": "happy.mp3",
    "sad":   "sad.mp3",
    "cry":   "cry.mp3",
}

FRAMES_BETWEEN_ANALYZE = 8
CRY_SAD_THRESHOLD = 60.0
WIN_NAME = "Emotion Detector"

def play_audio_blocking(path):
    pygame.mixer.init()
    try:
        pygame.mixer.music.load(path)
    except Exception as e:
        print("Audio load error:", e)
        return
    pygame.mixer.music.play()
    print(f"[{datetime.now().isoformat()}] Playing: {path}")
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

def choose_label_from_deepface_result(res):
    emotions = res.get("emotion", {})
    dom = res.get("dominant_emotion", None)
    detail = ", ".join([f"{k}:{v:.1f}" for k, v in emotions.items()])
    if dom == "happy":
        return "happy", detail
    if dom == "sad":
        sad_val = emotions.get("sad", 0.0)
        if sad_val >= CRY_SAD_THRESHOLD:
            return "cry", detail
        else:
            return "sad", detail
    if dom in ("angry", "fear", "disgust"):
        strongest = max(emotions.get("angry",0), emotions.get("fear",0), emotions.get("disgust",0))
        if strongest >= CRY_SAD_THRESHOLD:
            return "cry", detail
        else:
            return "sad", detail
    return None, detail

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        sys.exit(1)
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    frame_count = 0
    detected_label = None
    detected_info = None
    print("Camera started. Waiting for expression. Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame.")
                break
            frame_count += 1
            cv2.imshow(WIN_NAME, frame)
            if frame_count % FRAMES_BETWEEN_ANALYZE == 0:
                try:
                    result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
                    if isinstance(result, list) and len(result) > 0:
                        res = result[0]
                    else:
                        res = result
                    dom = res.get("dominant_emotion", None)
                    emotions = res.get("emotion", {})
                    ts = datetime.now().isoformat()
                    print(f"[{ts}] Dominant: {dom} | {', '.join([f'{k}:{v:.1f}' for k,v in emotions.items()])}")
                    label, detail = choose_label_from_deepface_result(res)
                    if label:
                        detected_label = label
                        detected_info = {"dominant": dom, "scores": emotions, "detail": detail, "time": ts}
                        break
                except Exception as e:
                    print("DeepFace error:", str(e))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Manual quit.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    if detected_label:
        print("\n=== DETECTION ===")
        print("Label:", detected_label)
        print("Time:", detected_info["time"])
        print("Dominant:", detected_info["dominant"])
        print("Scores:", detected_info["scores"])
        print("Detail:", detected_info["detail"])
        print("=================\n")
        audio_file = MUSIC.get(detected_label)
        if audio_file:
            play_audio_blocking(audio_file)
        else:
            print("No audio for", detected_label)
    else:
        print("No matching emotion detected.")

if __name__ == "__main__":
    main()

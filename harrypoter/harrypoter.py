import cv2
import numpy as np
import argparse
from collections import deque

def parse_args():
    p = argparse.ArgumentParser(description="Real-time Invisible Cloak using OpenCV")
    p.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    p.add_argument("--color", type=str, default="red", choices=["red", "green", "blue"],
                   help="Cloak color to make invisible")
    p.add_argument("--width", type=int, default=640, help="Capture width")
    p.add_argument("--height", type=int, default=480, help="Capture height")
    p.add_argument("--warmup-frames", type=int, default=60, help="Frames to build background")
    p.add_argument("--denoise", action="store_true", help="Extra denoising for noisy lighting")
    return p.parse_args()

def get_hsv_ranges(color):
    """
    Returns list of (lower, upper) HSV tuples for the chosen color.
    H in [0,179], S,V in [0,255]. Values tuned for typical indoor lighting.
    You may tweak if your lights or fabric differ.
    """
    if color == "red":
        # Red wraps around 0°, so we use two ranges.
        ranges = [
            (np.array([0, 120, 70]),   np.array([10, 255, 255])),
            (np.array([170, 120, 70]), np.array([180, 255, 255]))
        ]
    elif color == "green":
        ranges = [
            (np.array([35, 80, 70]), np.array([85, 255, 255]))
        ]
    else:  # blue
        ranges = [
            (np.array([90, 80, 70]), np.array([130, 255, 255]))
        ]
    return ranges

def build_background(cap, frames=60, use_median=True):
    """
    Capture several frames with no subject and combine them to a stable background.
    Median is robust to flicker; mean is fine too.
    """
    grabbed_frames = []
    for _ in range(frames):
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        grabbed_frames.append(frame)
    if not grabbed_frames:
        return None
    if use_median:
        bg = np.median(np.stack(grabbed_frames, axis=0), axis=0).astype(np.uint8)
    else:
        bg = np.mean(np.stack(grabbed_frames, axis=0), axis=0).astype(np.uint8)
    return bg

def refine_mask(mask, kernel_sz=3, iterations=2, extra_denoise=False):
    """
    Morphological cleanup to remove speckles and fill small holes.
    """
    kernel = np.ones((kernel_sz, kernel_sz), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    mask = cv2.dilate(mask, kernel, iterations=iterations)
    if extra_denoise:
        # Optional: bilateral filter on mask edges to stabilize shimmering
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
    return mask

def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print("ERROR: Could not open camera. Try a different --camera index.")
        return

    hsv_ranges = get_hsv_ranges(args.color)

    # Warm up and capture initial background
    print("[i] Warming up camera & capturing background...")
    background = build_background(cap, frames=args.warmup_frames, use_median=True)
    if background is None:
        print("ERROR: Failed to capture background.")
        cap.release()
        return
    print("[i] Background captured. Press 'b' to recapture anytime, 'q' to quit.")

    # A tiny history of masks to stabilize (optional)
    mask_history = deque(maxlen=5)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Mirror for a selfie-view feel
        frame = cv2.flip(frame, 1)

        # Safety: resize to requested dims (some cams ignore set)
        frame = cv2.resize(frame, (args.width, args.height))
        bg = cv2.resize(background, (args.width, args.height))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Build mask for selected color
        mask_total = None
        for (lower, upper) in hsv_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            mask_total = mask if mask_total is None else (mask_total | mask)

        # Optional extra threshold tweak if your fabric is shiny
        # mask_total = cv2.medianBlur(mask_total, 3)

        # Refine
        mask_refined = refine_mask(mask_total, kernel_sz=3, iterations=2, extra_denoise=args.denoise)

        # Temporal smoothing (reduces flicker)
        mask_history.append(mask_refined)
        mask_avg = np.mean(mask_history, axis=0).astype(np.uint8)
        _, mask_binary = cv2.threshold(mask_avg, 127, 255, cv2.THRESH_BINARY)

        # Inverse mask for the normal foreground
        inv_mask = cv2.bitwise_not(mask_binary)

        # Extract background where cloak is present
        cloak_area = cv2.bitwise_and(bg, bg, mask=mask_binary)
        # Keep current frame where cloak is NOT present
        non_cloak_area = cv2.bitwise_and(frame, frame, mask=inv_mask)

        # Composite
        final = cv2.addWeighted(cloak_area, 1, non_cloak_area, 1, 0)

        # Small HUD
        hud = frame.copy()
        cv2.putText(final, f"Cloak: {args.color} | 'b' recapture bg | 'q' quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("Invisible Cloak - Output", final)
        cv2.imshow("Mask (debug)", mask_binary)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            # Give a second to move out of frame
            print("[i] Recapturing background in 1 sec. Step out of frame...")
            for _ in range(30):
                cap.read()
                cv2.waitKey(1)
            background = build_background(cap, frames=args.warmup_frames, use_median=True)
            if background is not None:
                print("[i] Background updated.")
            else:
                print("[!] Background recapture failed; keeping previous.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

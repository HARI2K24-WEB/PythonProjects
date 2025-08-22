import cv2

# Start video capture
cap = cv2.VideoCapture(0)

# First frame (for comparison)
first_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for speed + convert to grayscale
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Save first frame as baseline
    if first_frame is None:
        first_frame = gray
        continue

    # Calculate difference between current and first frame
    frame_delta = cv2.absdiff(first_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

    # Dilation to fill holes
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours (moving areas)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 500:  # ignore small movements
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Movement Detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show result
    cv2.imshow("Motion Detection", frame)
    cv2.imshow("Threshold", thresh)
    cv2.imshow("Frame Delta", frame_delta)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

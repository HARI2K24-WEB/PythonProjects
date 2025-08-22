import cv2

def thermal_camera():
    # Start webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror view
        frame = cv2.flip(frame, 1)

        # Convert to grayscale first
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply thermal colormap (JET)
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

        # Show both original and thermal
        cv2.imshow("Original", frame)
        cv2.imshow("Thermal Effect", thermal)

        # ESC key to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    thermal_camera()

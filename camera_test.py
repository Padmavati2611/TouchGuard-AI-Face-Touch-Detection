import cv2

cap = cv2.VideoCapture(0)

print("Press Q to close the camera.")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Could not read camera frame.")
        break

    cv2.imshow("TouchGuard Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

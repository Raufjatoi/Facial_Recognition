import torch
import cv2

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x')  # Choose 'yolov5s', 'yolov5m', 'yolov5l', or 'yolov5x' for different sizes

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv5 model
    results = model(frame)

    # Render and display results
    results.render()  # Renders detection on the frame
    cv2.imshow("YOLOv5 Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

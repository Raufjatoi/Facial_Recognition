import torch
import cv2

# Load YOLOv5 pre-trained model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # YOLOv5 small model

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam, you can change it if needed

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Perform detection on the frame
    results = model(frame)

    # Render the results on the frame
    results.render()  # Draw bounding boxes on the frame

    # Display the frame with detected objects
    cv2.imshow('Live Detection', frame)

    # Check for a keypress to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()

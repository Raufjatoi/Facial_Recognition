import torch
import cv2 as c

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

cap = c.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("failed to grab frames")
        break
    
    results = model(frame)

    results.render()

    c.imshow('Live Detection', frame)

    if c.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
c.destroyAllWindows()
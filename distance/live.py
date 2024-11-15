import torch
import cv2
import numpy as np
from scipy.spatial import distance

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')

MIN_DISTANCE = 100

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    detections = results.pandas().xyxy[0]
    people = detections[detections['name'] == 'person']

    centroids = []
    for i, row in people.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        centroid = ((x1 + x2) // 2, (y1 + y2) // 2) 
        centroids.append(centroid)

        color = (0, 255, 0) 
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, centroid, 5, color, -1) 

    for i, (c1) in enumerate(centroids):
        for j, (c2) in enumerate(centroids[i + 1:], start=i + 1):
            dist = distance.euclidean(c1, c2)
            if dist < MIN_DISTANCE:
                cv2.line(frame, c1, c2, (0, 0, 255), 2)
                cv2.circle(frame, c1, 5, (0, 0, 255), -1)
                cv2.circle(frame, c2, 5, (0, 0, 255), -1)

    cv2.imshow('Social Distancing Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import torch
import cv2
import numpy as np
from scipy.spatial import distance

# Load YOLOv5 model (ensure yolov5s.pt is available in the working directory or provide its full path)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')

# Minimum distance in pixels to define safe distancing
MIN_DISTANCE = 100

# Load the image (provide the path to your image file)
image_path = 'test.png'
image = cv2.imread(image_path)

# Run YOLOv5 inference on the image
results = model(image)

# Filter out only detections for people
detections = results.pandas().xyxy[0]
people = detections[detections['name'] == 'person']

# Calculate centroids of detected people
centroids = []
for i, row in people.iterrows():
    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
    centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
    centroids.append(centroid)

    # Draw bounding boxes around each detected person
    color = (0, 255, 0)  # Green for initial safe distance
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.circle(image, centroid, 5, color, -1)

# Calculate distances between each pair of centroids
for i, c1 in enumerate(centroids):
    for j, c2 in enumerate(centroids[i + 1:], start=i + 1):
        dist = distance.euclidean(c1, c2)
        if dist < MIN_DISTANCE:
            # Draw a red line and update the colors for violating social distancing
            cv2.line(image, c1, c2, (0, 0, 255), 2)
            cv2.circle(image, c1, 5, (0, 0, 255), -1)
            cv2.circle(image, c2, 5, (0, 0, 255), -1)

# Show the final image with social distancing alerts
cv2.imshow('Social Distancing Detector - Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  

cap = cv2.VideoCapture(0)

class_names = model.names

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    results = model(frame)

    results.render()

    labels = results.xywh[0][:, -1].cpu().numpy()
    counts = {class_names[int(label)]: 0 for label in labels} 

    for label in labels:
        counts[class_names[int(label)]] += 1

    y_offset = 30 
    for obj, count in counts.items():
        cv2.putText(frame, f"{obj}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        y_offset += 40

    cv2.imshow("Object Counting Tool", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
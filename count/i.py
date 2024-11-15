import cv2
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  

image_path = 'test.png' 
img = cv2.imread(image_path)

results = model(img)

results.render()

labels = results.xywh[0][:, -1].cpu().numpy() 
class_names = model.names  
counts = {class_names[int(label)]: 0 for label in labels}  

for label in labels:
    counts[class_names[int(label)]] += 1

y_offset = 30 
for obj, count in counts.items():
    cv2.putText(img, f"{obj}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    y_offset += 40  

cv2.imshow("Object Detection", img)

cv2.imwrite('result.png', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
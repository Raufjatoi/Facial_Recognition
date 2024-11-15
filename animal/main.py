import torch
import cv2
import matplotlib.pyplot as plt

model_s = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', force_reload=True)   
#model_m = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5m.pt', force_reload=True)
#model_l = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5l.pt', force_reload=True) 
model_x = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5x.pt', force_reload=True) 

def detect_animals(image_path, model):
    img = cv2.imread(image_path)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model(img_rgb)

    results.render() 

    img_bgr = cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR)

    cv2.imshow('Detection', img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('res2.png', img_bgr)

detect_animals('test3.png', model_x)
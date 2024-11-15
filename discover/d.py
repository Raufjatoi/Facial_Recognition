import cv2
import os
import json

# Initialize some global variables
drawing = False  # Whether the mouse is being clicked
ix, iy = -1, -1  # Starting coordinates for the rectangle
annotations = []  # List to store the annotations

# Define the object class
classes = ['class1', 'class2', 'class3']  # Customize this with your object classes

# Function to draw the bounding box
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img, annotations, current_class

    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button press
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:  # Mouse movement
        if drawing:
            img_copy = img.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Image Annotation", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:  # Left mouse button release
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow("Image Annotation", img)
        # Save annotation as a relative bounding box (normalized)
        h, w, _ = img.shape
        x_center = (ix + x) / 2 / w
        y_center = (iy + y) / 2 / h
        width = abs(x - ix) / w
        height = abs(y - iy) / h
        annotations.append({
            "class": current_class,
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height
        })

# Function to save annotations
def save_annotations(image_name, annotations):
    annotation_file = os.path.join('annotations', image_name.split('.')[0] + '.json')  # Save as JSON file
    with open(annotation_file, 'w') as f:
        json.dump(annotations, f)

    # Save the annotated image
    cv2.imwrite(f'annotated_{image_name}', img)

# Load the image
image_path = 'test.jpeg'  # Replace with the path to your image
img = cv2.imread(image_path)
img_copy = img.copy()

# Create an output directory for annotations
if not os.path.exists('annotations'):
    os.makedirs('annotations')

# Start the OpenCV window
cv2.imshow("Image Annotation", img)

# Set mouse callback for drawing bounding boxes
cv2.setMouseCallback("Image Annotation", draw_rectangle)

# Ask the user to select the class and annotate multiple objects
current_class = 0  # Start with the first class
while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # Press 'q' to quit
        break

    elif key == ord('n'):  # Press 'n' to go to the next object class
        current_class = (current_class + 1) % len(classes)
        print(f"Current Class: {classes[current_class]}")

    elif key == ord('s'):  # Press 's' to save the annotations
        save_annotations(image_path, annotations)
        print(f"Annotations saved for {image_path}")

    elif key == ord('r'):  # Press 'r' to reset the image and annotations
        img = img_copy.copy()
        annotations = []
        print("Annotations reset.")

# Close the OpenCV window
cv2.destroyAllWindows()

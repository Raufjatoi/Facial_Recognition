import os
import cv2
import face_recognition
import numpy as np

# Path to the folder containing known faces
known_faces_dir = "known_faces/"

# Initialize arrays to hold known face encodings and their corresponding names
known_face_encodings = []
known_face_names = []

# Load known faces and their names from the directory structure
def load_known_faces():
    for person_name in os.listdir(known_faces_dir):
        person_folder = os.path.join(known_faces_dir, person_name)
        if os.path.isdir(person_folder):
            for image_file in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_file)
                
                # Load the image file and convert it to RGB (required by face_recognition)
                image = cv2.imread(image_path)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Get face encodings for the image
                face_encodings = face_recognition.face_encodings(rgb_image)
                
                # If faces are found in the image, we store the first face encoding
                if face_encodings:
                    known_face_encodings.append(face_encodings[0])
                    known_face_names.append(person_name)

# Initialize the webcam for real-time face recognition
def recognize_faces_in_image(image):
    # Convert the image to RGB (required by face_recognition)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Find all face locations and encodings in the image
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    # List of names of recognized faces
    face_names = []

    for face_encoding in face_encodings:
        # Compare each face encoding with known face encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"  # Default name if no match found
        
        # If a match is found, get the corresponding name
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        
        face_names.append(name)

    return face_locations, face_names

# Load known faces from the directory
load_known_faces()

# Open webcam for live video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Detect and recognize faces in the frame
    face_locations, face_names = recognize_faces_in_image(frame)

    # Draw rectangles around faces and label them with names
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame with rectangles and labels
    cv2.imshow("Live Face Recognition", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

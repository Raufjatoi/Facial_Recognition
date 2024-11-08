# create_model.py
import os
import cv2
import face_recognition
import pickle

# Path to the folder containing known faces
known_faces_dir = "known_faces/"
model_path = "updated_model.pkl"

# Initialize arrays to hold known face encodings and their corresponding names
known_face_encodings = []
known_face_names = []

# Load known faces and their names from the directory structure
for person_name in os.listdir(known_faces_dir):
    person_folder = os.path.join(known_faces_dir, person_name)
    if os.path.isdir(person_folder):
        for image_file in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_file)
            image = cv2.imread(image_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_image)
            if face_encodings:
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(person_name)

# Save encodings and names to a pickle file
with open(model_path, "wb") as model_file:
    pickle.dump({"encodings": known_face_encodings, "names": known_face_names}, model_file)

print("Model saved to", model_path)

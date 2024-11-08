import face_recognition
import cv2
import numpy as np
import os

def load_known_faces(known_faces_dir="known_faces"):
    known_face_encodings = []
    known_face_names = []

    # Loop through each image file in the known_faces directory
    for filename in os.listdir(known_faces_dir):
        # Ensure we're only processing images
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)

            # Make sure there's at least one face encoding
            if encoding:
                known_face_encodings.append(encoding[0])
                known_face_names.append(os.path.splitext(filename)[0])  # Use filename as the name

    return known_face_encodings, known_face_names

def recognize_faces(frame, known_face_encodings, known_face_names):
    # Resize frame to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]  # Convert to RGB

    # Detect face locations
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = []

    # Only encode faces if face locations are found
    if face_locations:
        try:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        except TypeError as e:
            print("Error with face encoding:", e)
            face_encodings = []  # Skip this frame if there's an error with encoding

    face_names = []
    for face_encoding in face_encodings:
        # Compare each face encoding with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Use the known face with the smallest distance to the new face
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(name)

    return face_locations, face_names
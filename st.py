import os
import cv2
import face_recognition
import numpy as np
import streamlit as st
from PIL import Image

# Path to the folder containing known faces
known_faces_dir = "known_faces/"
known_face_encodings = []
known_face_names = []

# Function to load known faces
def load_known_faces():
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

# Recognize faces in the uploaded image
def recognize_faces_in_image(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    face_names = []
    face_colors = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        color = (0, 0, 255)  # Red for unknown

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            color = (0, 255, 0)  # Green for known

        face_names.append(name)
        face_colors.append(color)

    return face_locations, face_names, face_colors

# Streamlit App Configuration
st.title("Custom Face Recognition App")
st.write("Upload an image and see if faces are recognized!")

# Upload Image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Convert the uploaded file to a format OpenCV can work with
    image = np.array(Image.open(uploaded_image))
    face_locations, face_names, face_colors = recognize_faces_in_image(image)

    # Draw rectangles and labels on the image
    for (top, right, bottom, left), name, color in zip(face_locations, face_names, face_colors):
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # Display the processed image in Streamlit
    st.image(image, caption="Processed Image", use_column_width=True)
    st.write("Faces Detected:", face_names)

# Load known faces at app start
load_known_faces()

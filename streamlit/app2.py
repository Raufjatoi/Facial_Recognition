# app.py
import cv2
import face_recognition
import streamlit as st
from PIL import Image
import numpy as np
import pickle
import os
import tempfile

# Default model path
default_model_path = "face_encodings_model.pkl"

# Load the model from a given path
def load_model(model_path):
    with open(model_path, "rb") as model_file:
        data = pickle.load(model_file)
        return data["encodings"], data["names"]

# Recognize faces in an image
def recognize_faces_in_image(image, known_face_encodings, known_face_names):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    face_names = []
    face_colors = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        color = (0, 102, 255)  # Blue for unknown

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            color = (0, 255, 0)  # Green for known

        face_names.append(name)
        face_colors.append(color)

    return face_locations, face_names, face_colors

# Streamlit App Configuration
st.set_page_config(page_title="Face Recognition App", page_icon=":smiley:", layout="centered")

# App Title and Instructions
st.title("🔍 Custom Face Recognition App")
st.markdown("Upload an image or activate live detection to identify known faces. Known faces will be highlighted in **green**, and unknown faces in **blue**.")

# Sidebar - Model Selection
st.sidebar.header("🔧 Model Settings")
use_custom_model = st.sidebar.checkbox("Use a custom model")

if use_custom_model:
    custom_model = st.sidebar.file_uploader("Upload your custom model (.pkl)", type=["pkl"])
    if custom_model:
        known_face_encodings, known_face_names = load_model(custom_model)
        st.sidebar.success("Custom model loaded successfully!")
    else:
        st.sidebar.warning("Please upload a custom model to use.")
else:
    known_face_encodings, known_face_names = load_model(default_model_path)
    st.sidebar.info("Using default model.")

# Mode Selection
st.write("## Choose Mode")
mode = st.radio("Select a mode:", ("Image Upload", "Live Detection"))

# Image Upload Mode
if mode == "Image Upload":
    uploaded_image = st.file_uploader("Choose an image to analyze...", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        image = np.array(Image.open(uploaded_image))
        face_locations, face_names, face_colors = recognize_faces_in_image(image, known_face_encodings, known_face_names)

        # Draw rectangles and labels on the image
        for (top, right, bottom, left), name, color in zip(face_locations, face_names, face_colors):
            cv2.rectangle(image, (left, top), (right, bottom), color, 2)
            cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        # Display the processed image with labels
        st.image(image, caption="Processed Image with Recognized Faces", use_column_width=True)
        st.write("**Detected Faces**:", ", ".join(face_names))

# Live Detection Mode
elif mode == "Live Detection":
    if st.button("Start Live Detection"):
        # Open a temporary window for live camera feed
        temp_file = tempfile.NamedTemporaryFile(delete=False)

        # Webcam capture
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Could not open webcam.")
        else:
            st.write("**Press 'Q' to quit live detection.**")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to grab frame.")
                    break

                # Perform face recognition on the frame
                face_locations, face_names, face_colors = recognize_faces_in_image(frame, known_face_encodings, known_face_names)

                # Draw rectangles and labels on the frame
                for (top, right, bottom, left), name, color in zip(face_locations, face_names, face_colors):
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

                # Display the frame in Streamlit
                cv2.imwrite(temp_file.name, frame)
                st.image(temp_file.name)

                # Break loop if 'Q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Release the capture and close windows
            cap.release()
            cv2.destroyAllWindows()

# Footer with Credit
st.markdown("""
<hr>
<p style='text-align: center;'>
    Created by <a href='https://www.linkedin.com/in/rauf' target='_blank'>Rauf</a>
</p>
""", unsafe_allow_html=True)

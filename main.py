import os as o
import cv2 as c
import face_recognition as f
import numpy as n

known_faces_dir = "known_faces/"

known_face_encodings = []
known_face_names = []

def load_known_faces():
    for person_name in o.listdir(known_faces_dir):
        person_folder = o.path.join(known_faces_dir, person_name)
        if o.path.isdir(person_folder):
            for image_file in o.listdir(person_folder):
                image_path = o.path.join(person_folder, image_file)
                
                image = c.imread(image_path)
                rgb_image = c.cvtColor(image, c.COLOR_BGR2RGB)
                
                face_encodings = f.face_encodings(rgb_image)
                
                if face_encodings:
                    known_face_encodings.append(face_encodings[0])
                    known_face_names.append(person_name)

def recognize_faces_in_image(image):
    rgb_image = c.cvtColor(image, c.COLOR_BGR2RGB)
    
    face_locations = f.face_locations(rgb_image)
    face_encodings = f.face_encodings(rgb_image, face_locations)
    
    face_names = []

    for face_encoding in face_encodings:
        matches = f.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        
        face_names.append(name)

    return face_locations, face_names

load_known_faces()


image_path = 'test1.PNG'
image = c.imread(image_path)

face_locations, face_names = recognize_faces_in_image(image)

for (top, right, bottom, left), name in zip(face_locations, face_names):
    c.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    c.putText(image, name, (left, top - 10), c.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

c.imshow("Face Recognition", image)
c.waitKey()
c.destroyAllWindows()

c.imwrite("recognized_faces.jpg", image)

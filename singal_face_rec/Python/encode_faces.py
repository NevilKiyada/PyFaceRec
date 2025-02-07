import os
import face_recognition
import pickle

image_directory = 'captured_faces'
face_encodings = {}

for person_name in os.listdir(image_directory):
    person_dir = os.path.join(image_directory, person_name)
    if os.path.isdir(person_dir):
        all_encodings = []

        for image_file in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_file)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                all_encodings.append(encodings[0])

        if all_encodings:
            average_encoding = sum(all_encodings) / len(all_encodings)
            face_encodings[person_name] = average_encoding

with open("face_encodings.pkl", "wb") as f:
    pickle.dump(face_encodings, f)

print("Face encodings saved successfully.")

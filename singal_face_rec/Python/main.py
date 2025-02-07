import cv2
import face_recognition
import pickle
import numpy as np

# Load face encodings
with open("face_encodings.pkl", "rb") as f:
    known_faces = pickle.load(f)

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (face_location, face_encoding) in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(list(known_faces.values()), face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = list(known_faces.keys())[first_match_index]

        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

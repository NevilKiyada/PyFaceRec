import cv2
import face_recognition
import numpy as np
import os

# Path to stored face images
FACE_DIR = "captured_faces"

# Load known faces and their names
known_face_encodings = []
known_face_names = []

# Dynamically load all images in the "captured_faces" directory
for person_name in os.listdir(FACE_DIR):
    person_folder = os.path.join(FACE_DIR, person_name)
    if os.path.isdir(person_folder):  # Ensure it's a folder
        for filename in os.listdir(person_folder):
            img_path = os.path.join(person_folder, filename)
            try:
                img = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(img)
                if encodings:  # Ensure at least one face is detected
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(person_name)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

print(f"Loaded {len(known_face_encodings)} known faces.")

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame, model="cnn")  # Use "cnn" for better accuracy
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Match detected face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
        name = "Unknown"

        # Find best match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Scale face locations back to original size
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # Draw bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show video frame
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
video_capture.release()
cv2.destroyAllWindows()

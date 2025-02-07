import os
import cv2
import face_recognition

# Number of images to capture
num_photos = 40

# Get person’s name
person_name = input("Enter the name of the person: ")
save_path = os.path.join("captured_faces", person_name)
os.makedirs(save_path, exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()


capture_count = 0

while capture_count < num_photos:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)

    if face_locations:
        img_path = os.path.join(save_path, f"{person_name}_{capture_count + 1}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Captured and saved {img_path}")
        capture_count += 1

    cv2.imshow("Capturing Face", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Finished capturing {capture_count} images for {person_name}.")

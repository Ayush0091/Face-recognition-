import cv2
import numpy as np
import face_recognition
import pickle

# Load or initialize face data
try:
    with open("face_data.pkl", "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
except FileNotFoundError:
    known_face_encodings = []
    known_face_names = []

# Start webcam
video_capture = cv2.VideoCapture(0)

# This will control whether we prompt the user for updates or deletions
prompted_for_update_or_delete = False

# Define a threshold for face recognition accuracy
FACE_RECOGNITION_THRESHOLD = 0.6  # Lower values will allow closer matches

while True:
    ret, frame = video_capture.read()
    if not ret:
        continue

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces & encode them
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2  # Resize back

        # Check if the face is recognized (use face distance for accuracy)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        name = "Unknown"
        accuracy = 100 - min(face_distances) * 100  # Accuracy based on face distance

        # Only accept matches with a distance below the threshold
        if True in matches and min(face_distances) < FACE_RECOGNITION_THRESHOLD:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            accuracy = 100 - min(face_distances) * 100  # Update accuracy for known faces

        # Draw rectangle and display name with accuracy
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({accuracy:.2f}%)", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # If face is recognized, ask if the user wants to update or delete the identity (only once per session)
        if name != "Unknown" and not prompted_for_update_or_delete:
            prompted_for_update_or_delete = True
            print(f"\nKnown face detected: {name}. Do you want to update the identity, delete the identity, or do nothing?")
            action = input("Enter 'update' to change name, 'delete' to remove the person, or 'no' to skip: ").strip().lower()

            if action == 'update':
                print(f"Enter the new name for {name}: ")
                new_name = input().strip()

                # Update the name in the list
                first_match_index = matches.index(True)
                known_face_names[first_match_index] = new_name

                # Save updated data to file
                with open("face_data.pkl", "wb") as f:
                    pickle.dump((known_face_encodings, known_face_names), f)

                print(f"Face identity updated to {new_name}!")
                break  # Exit the loop after updating

            elif action == 'delete':
                print(f"Deleting {name} from known faces...")
                first_match_index = matches.index(True)
                del known_face_encodings[first_match_index]
                del known_face_names[first_match_index]

                # Save updated data to file
                with open("face_data.pkl", "wb") as f:
                    pickle.dump((known_face_encodings, known_face_names), f)

                print(f"Face identity for {name} deleted!")
                break  # Exit the loop after deletion

        # If face is unknown, ask to save or update
        if name == "Unknown" and not prompted_for_update_or_delete:
            prompted_for_update_or_delete = True
            print("\nNew face detected! Do you want to save this person? (y/n): ")
            response = input().strip().lower()
            if response == 'y':
                print("Enter name for this person: ")
                person_name = input().strip()

                known_face_encodings.append(face_encoding)
                known_face_names.append(person_name)

                # Save to file
                with open("face_data.pkl", "wb") as f:
                    pickle.dump((known_face_encodings, known_face_names), f)

                print(f"Face saved as {person_name}!")
                break  # Exit the loop after saving

    # Display output
    cv2.imshow('Face Recognition', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

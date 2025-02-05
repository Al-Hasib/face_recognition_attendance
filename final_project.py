import cv2
import numpy as np
import face_recognition
import os
import openpyxl
from datetime import datetime

# Load known face images from the "images" folder
image_folder = r"F:\Masud vai\masud_vai_dataset"
known_face_encodings = []
known_face_names = []

for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img = face_recognition.load_image_file(os.path.join(image_folder, filename))

        # Resize the loaded image
        small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

        face_encoding = face_recognition.face_encodings(small_frame)[0]  # Assuming there's only one face
        known_face_encodings.append(face_encoding)
        known_face_names.append(os.path.splitext(filename)[0])

# Create or load Excel file for attendance
excel_file = r"F:\Masud vai\attendance_for_project.xlsx"

if os.path.isfile(excel_file):
    workbook = openpyxl.load_workbook(excel_file)
    sheet = workbook.active
    sheet["A1"] = "Name"
    sheet["B1"] = "Time"
    sheet["C1"] = "Status"  # Add a column for Status
    sheet["D1"] = "Remarks"
else:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet["A1"] = "Name"
    sheet["B1"] = "Time"
    sheet["C1"] = "Status"  # Add a column for Status
    sheet["D1"] = "Remarks"  # Add a column for Remarks
    workbook.save(excel_file)

# Initialize variables
known_names_recorded = set()  # Keep track of known names already recorded
process_this_frame = True

# Open the webcam (you can change the argument to the camera index if you have multiple cameras)
cap = cv2.VideoCapture(0)

# Define face_locations and face_names outside the loop
face_locations = []
face_names = []

while True:
    ret, frame = cap.read()

    # Resize frame to speed up face recognition
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if process_this_frame:
        # Find all face locations and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Calculate face distances to all known faces
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            # Find the index of the face with the smallest distance (best match)
            best_match_index = np.argmin(face_distances)

            # If the smallest distance is below a certain threshold, consider it a match
            if face_distances[best_match_index] < 0.6:
                best_match_name = known_face_names[best_match_index]

                # Record attendance in Excel only if the face is known and not already recorded
                if best_match_name not in known_names_recorded:
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    sheet.append([best_match_name, current_time, "Present", ""])
                    known_names_recorded.add(best_match_name)
            else:
                best_match_name = "Unknown"

            face_names.append(best_match_name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw the name below the face
        cv2.putText(frame, name, (left + 6, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the resulting image
    cv2.imshow('Face Recognition Attendance System', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the workbook and release the webcam
workbook.save(excel_file)
cap.release()
cv2.destroyAllWindows()
import cv2
import face_recognition
import numpy as np

# Helper function to load and process reference images
def load_and_encode_image(image_path):
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)
    if encoding:
        return encoding[0]
    return None

# Load and encode reference images
reference_face_encodings = [
    load_and_encode_image("biden.png"),
    load_and_encode_image("obama.png"),
    load_and_encode_image("MESSI.jpg"),
    load_and_encode_image("arm.jpg")
]

# Filter out None values
reference_face_encodings = [encoding for encoding in reference_face_encodings if encoding is not None]
reference_labels = ["Biden", "Obama", "Messi", "Arm"]

# Open video capture
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FPS, 15)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Reduced resolution
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Reduce the frame size to speed up processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    
    # Convert the frame from BGR to RGB
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2 
        
        # Compare face encoding with reference encodings
        matches = face_recognition.compare_faces(reference_face_encodings, face_encoding)
        
        if True in matches:
            match_index = matches.index(True)
            label = reference_labels[match_index]
            color = [(0, 255, 0), (0, 255, 255), (255, 255, 0), (0, 0, 0)][match_index]
        else:
            label = "Unknown"
            color = (0, 0, 255)
        
        # Draw rectangle and label
        cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow('Video', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy all windows
video_capture.release()
cv2.destroyAllWindows()

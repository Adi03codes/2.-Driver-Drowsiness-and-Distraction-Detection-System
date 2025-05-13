# 2.-Driver-Drowsiness-and-Distraction-Detection-System
##This Driver Drowsiness and Distraction Detection System is a real-time AI-powered solution designed to improve road safety by monitoring and detecting signs of driver fatigue and distraction. 


import cv2
import dlib
from scipy.spatial import distance
import time

# Load pre-trained face and eye detector models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load dlib's facial landmark predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Function to calculate EAR (Eye Aspect Ratio)
def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the vertical eye landmarks
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    # Compute the euclidean distance between the horizontal eye landmarks
    C = distance.euclidean(eye[0], eye[3])
    # Return the EAR (Eye Aspect Ratio)
    ear = (A + B) / (2.0 * C)
    return ear

# Define EAR threshold for drowsiness detection
EAR_THRESHOLD = 0.2
EYE_AR_CONSEC_FRAMES = 30

# Initialize drowsiness detection variables
COUNTER = 0
ALERT = False

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop over the faces found
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Detect facial landmarks using dlib
        rect = dlib.rectangle(x, y, x + w, y + h)
        landmarks = predictor(gray, rect)
        
        # Get the coordinates for the left and right eye
        left_eye = []
        right_eye = []
        
        for i in range(36, 42):  # Left eye landmarks
            left_eye.append((landmarks.part(i).x, landmarks.part(i).y))
        
        for i in range(42, 48):  # Right eye landmarks
            right_eye.append((landmarks.part(i).x, landmarks.part(i).y))
        
        # Convert coordinates to numpy array
        left_eye = numpy.array(left_eye)
        right_eye = numpy.array(right_eye)
        
        # Calculate the Eye Aspect Ratio for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        
        # Average EAR for both eyes
        ear = (left_ear + right_ear) / 2.0
        
        # Drowsiness detection logic
        if ear < EAR_THRESHOLD:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALERT:
                    ALERT = True
                    print("Drowsiness Alert!")
                    # Trigger alert here (sound, message, etc.)
        else:
            COUNTER = 0
            ALERT = False

        # Draw the EAR on the frame
        cv2.putText(frame, f"EAR: {ear:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow("Driver Monitoring System", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

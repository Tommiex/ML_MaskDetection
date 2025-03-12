import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("mask_detection_model.h5")
print("Model loaded successfully!")

# Load the face detection model (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def prepare_image(face, img_size=(224, 224)):
    """ Prepares the detected face image for model prediction. """
    face = cv2.resize(face, img_size)
    face = face / 255.0  # Normalize
    face = np.expand_dims(face, axis=0)  # Add batch dimension
    return face

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]  # Extract face region
        prepared_face = prepare_image(face)
        
        # Predict mask or no mask
        prediction = model.predict(prepared_face)[0][0]
        
        # Determine label and color
        if prediction > 0.4:
            label = "No Mask"
            color = (0, 0, 255)  # Red
        else:
            label = "Mask"
            color = (0, 255, 0)  # Green
        
        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Show frame
    cv2.imshow("Mask Detection", frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.getWindowProperty("Mask Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

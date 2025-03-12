import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("mask_detection_model.h5")
print("Model loaded successfully!")

# Load the DNN Face Detector
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

def prepare_image(face, img_size=(224, 224)):
    """ Prepares the detected face image for model prediction. """
    face = cv2.resize(face, img_size)
    face = face / 255.0  # Normalize
    face = np.expand_dims(face, axis=0)  # Add batch dimension
    return face

def find_working_camera():
    """ Checks available camera indexes and returns the first working one. """
    for i in range(5):  # Check first 5 camera indexes
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            cap.release()
            return i
    return -1

# Open a working webcam
camera_index = find_working_camera()
if camera_index == -1:
    print("❌ No working camera found!")
    exit()

cap = cv2.VideoCapture(camera_index)
print(f"🎥 Using camera index: {camera_index}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to blob for DNN model
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.3:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x_max, y_max) = box.astype("int")

            face = frame[y:y_max, x:x_max]  # Extract face region
            if face.shape[0] > 50 and face.shape[1] > 50:
                prepared_face = prepare_image(face)
                prediction = model.predict(prepared_face)[0][0]

                # Determine label and color
                label = "No Mask" if prediction > 0.4 else "Mask"
                color = (0, 0, 255) if prediction > 0.5 else (0, 255, 0)

                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x_max, y_max), color, 2)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show frame
    cv2.imshow("Mask Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

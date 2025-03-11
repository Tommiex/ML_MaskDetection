import cv2
import numpy as np
from tensorflow.keras.models import load_model

def prepare_image(image_path, img_size=(224, 224)):
    # Read image from file
    img = cv2.imread(image_path)

#Resize image to 224x224
    img = cv2.resize(img, img_size)

#Normalize image
    img = (img / 127.5) - 1

#Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img

#Load the trained model
model = load_model("mask_detection_model.h5")
print("Model loaded successfully!")

#Test with an image
image_path = "archive\data\without_mask\without_mask_5.jpg"  # Set your image path here
img = prepare_image(image_path)

#Predict using the model
prediction = model.predict(img)
print(f"Prediction: {prediction}")

if prediction >= 0.5:
    print("Predicted: With Mask")
else:
    print("Predicted: Without Mask")
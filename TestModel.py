import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import random


# Prepare image function
def prepare_image(image_path, img_size=(224, 224)):
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: {image_path} does not exist.")
        return None

    # Read image from file
    img = cv2.imread(image_path)

    # Resize image to 224x224
    img = cv2.resize(img, img_size)

    # Normalize image (use the same normalization as in training)
    img = img / 255.0  # Normalize the image to [0, 1]

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img


# Load the trained model
model = load_model("mask_detection_model.h5")
print("Model loaded successfully!")

# Define dataset path
with_mask_path = "archive/data/with_mask"
without_mask_path = "archive/data/without_mask"

# Get all image filenames from both categories
with_mask_files = os.listdir(with_mask_path)
without_mask_files = os.listdir(without_mask_path)

# Shuffle and select 50 random images from each category
with_mask_selected = random.sample(with_mask_files, 50)
without_mask_selected = random.sample(without_mask_files, 50)

# Combine both selected lists
selected_images = with_mask_selected + without_mask_selected
random.shuffle(selected_images)  # Shuffle the selected images to randomize the output

# Iterate over selected images and make predictions
for image_name in selected_images:
    # Create the full path to the image
    if image_name in with_mask_selected:
        image_path = os.path.join(with_mask_path, image_name)
        label = "🟢"  # with_mask category symbol
    else:
        image_path = os.path.join(without_mask_path, image_name)
        label = "🔴"  # without_mask category symbol

    # Prepare image for prediction
    img = prepare_image(image_path)

    if img is not None:
        # Make a prediction
        prediction = model.predict(img)

        # Check if the prediction is greater than or equal to 0.5
        if prediction >= 0.5:
            print(
                f"{image_name} {label} - Predicted: Without Mask 🔴"
            )  # Predicted as "Without Mask"
        else:
            print(
                f"{image_name} {label} - Predicted: With Mask 🟢"
            )  # Predicted as "With Mask"

import os
import cv2
import numpy as np

dataset_path = "archive"
categories = ["with_mask", "without_mask"]  # Define labels
data = []
labels = []

for category in categories:
    folder_path = os.path.join(dataset_path, category)
    label = categories.index(category)  # Assign label (0 or 1)

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))  # Resize for consistency
        data.append(img)
        labels.append(label)

# Convert to NumPy arrays
data = np.array(data) / 255.0  # Normalize
labels = np.array(labels)

# Shuffle data
from sklearn.utils import shuffle
data, labels = shuffle(data, labels, random_state=42)
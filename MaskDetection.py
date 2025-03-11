import os
import cv2
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

DATASET_PATH = "archive\data"
CATEGORIES = ["with_mask", "without_mask"]  # Define labels


def load_dataset(dataset_path, categories, img_size=(224, 224)):

    data, labels = [], []

    for category in categories:
        folder_path = os.path.join(dataset_path, category)
        label = categories.index(category)  # Assign label (0 or 1)
        num_files = len(os.listdir(folder_path))
        print(f"Number of images in {category}: {num_files}")

        # วนลูปเพื่อดึงชื่อไฟล์รูปภาพจากโฟลเดอร์
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)

            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, img_size)  # ปรับขนาดรูปภาพให้เหมาะกับโมเดล
                data.append(img)
                labels.append(label)

            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    # Convert to NumPy arrays
    data = np.array(data) / 255.0  # Normalize
    labels = np.array(labels)

    return data, labels

data, labels = load_dataset(DATASET_PATH, CATEGORIES)
data, labels = shuffle(data, labels, random_state=42) # สุ่มลำดับของข้อมูล random_state=42 ใช้กำหนดค่าการสุ่มให้ได้ผลลัพธ์เดิมทุกครั้ง
print(np.array(data).shape)  # ✅ Correct, prints dataset shape

split_index = int(0.7 * len(data)) # split_index = 70% ของข้อมูลทั้งหมด ตามหลัก 70 เทรน / 30 ทดสอบ
data_train, data_test = data[:split_index], data[split_index:]
labels_train, labels_test = labels[:split_index], labels[split_index:]

print(f"Train data: {data_train.shape}, Train labels: {labels_train.shape}")
print(f"Test data: {data_test.shape}, Test labels: {labels_test.shape}")


print(f"Total images: {len(data)}")
print(f"Training set: {len(data_train)}")
print(f"Testing set: {len(data_test)}")

# Define the CNN architecture
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Regularization to prevent overfitting
    layers.Dense(1, activation='sigmoid')  # Binary classification (Mask vs. No Mask)
])

model.summary()  # Print model architecture

# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model using NumPy arrays
history = model.fit(data_train, labels_train, epochs=50, batch_size=32, validation_data=(data_test, labels_test))

model.save("mask_detection_model.h5")
print("save successfully")

model = load_model("mask_detection_model.h5")
print("loaded successfully!")

test_loss, test_acc = model.evaluate(data_test, labels_test)
print(f"Test Accuracy: {test_acc:.2%}")


import random
import matplotlib.pyplot as plt

# Select 10 random test images
random_indices = random.sample(range(len(data_test)), 10)

plt.figure(figsize=(10, 5))
for i, idx in enumerate(random_indices):
    sample_img = np.expand_dims(data_test[idx], axis=0)  # Add batch dimension
    prediction = model.predict(sample_img)[0][0]  # Get prediction score

    pred_label = "With Mask" if prediction < 0.5 else "Without Mask"
    
    # Show image
    plt.subplot(2, 5, i + 1)
    plt.imshow(data_test[idx])
    plt.title(pred_label)
    plt.axis("off")

plt.tight_layout()
plt.show()

print("asdasd",model.history.history.keys())

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()


from sklearn.metrics import classification_report

preds = (model.predict(data_test) > 0.5).astype(int)  # Convert probabilities to 0/1
print(classification_report(labels_test, preds, target_names=["With Mask", "Without Mask"]))



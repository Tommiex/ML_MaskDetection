import os
import cv2
import numpy as np
from sklearn.utils import shuffle

DATASET_PATH = "archive\data"
CATEGORIES = ["with_mask", "without_mask"]  # Define labels


def load_dataset(dataset_path, categories, img_size=(224, 224)):

    data, labels = [], []

    for category in categories:
        folder_path = os.path.join(dataset_path, category)
        label = categories.index(category) # Assign label (0 or 1)
    
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

split_index = int(0.7 * len(data)) # split_index = 70% ของข้อมูลทั้งหมด ตามหลัก 70 เทรน / 30 ทดสอบ
data_train, data_test = data[:split_index], data[split_index:]
labels_train, labels_test = labels[:split_index], labels[split_index:]

print(f"Train data: {data_train.shape}, Train labels: {labels_train.shape}")
print(f"Test data: {data_test.shape}, Test labels: {labels_test.shape}")
            
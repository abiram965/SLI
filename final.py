import os
import cv2
import math
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from cvzone.HandTrackingModule import HandDetector

def load_image_data(directory):
    images = []
    labels = []
    label_map = {}
    label_counter = 0
    for label in sorted(os.listdir(directory)):
        label_path = os.path.join(directory, label)
        if not os.path.isdir(label_path):
            continue
        if label not in label_map:
            label_map[label] = label_counter
            label_counter += 1
        for image in os.listdir(label_path):
            image_path = os.path.join(label_path, image)
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Unable to read image {image_path}")
                continue
            img = cv2.resize(img, (224, 224))
            images.append(img)
            labels.append(label_map[label])
    images = np.array(images)
    labels = np.array(labels)
    return images, labels, label_map

# Load data
data_path = r'C:\\Users\\Abiram\\Desktop\\SLI\\data'
X, y, label_map = load_image_data(data_path)
y = to_categorical(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test = X_train / 255.0, X_test / 255.0

# Define model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(label_map), activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save model with .keras extension (recommended)
model_path = r'C:\\Users\\Abiram\\Desktop\\SLI\\model.keras'
model.save(model_path)

# Load model
model = keras.models.load_model(model_path)
labels = list(label_map.keys())

# Initialize webcam and HandDetector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset, imgSize = 20, 224

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        # Ensure cropping stays within bounds
        h_img, w_img, _ = img.shape
        y1, y2 = max(0, y - offset), min(h_img, y + h + offset)
        x1, x2 = max(0, x - offset), min(w_img, x + w + offset)
        imgCrop = img[y1:y2, x1:x2]
        
        aspectRatio = h / w
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + hCal, :] = imgResize
        
        # Prepare for prediction
        imgWhite1 = np.expand_dims(imgWhite, axis=0)
        prediction = model.predict(imgWhite1)
        index = np.argmax(prediction)
        
        # Display result
        cv2.rectangle(imgOutput, (x1, y1 - 50), (x1 + 90, y1), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x1, y1), (x2, y2), (255, 0, 255), 4)
        
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    
    cv2.imshow("Image", imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
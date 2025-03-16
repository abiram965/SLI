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
import pyttsx3
import time
import threading
from googletrans import Translator
from deep_translator import GoogleTranslator

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

# Language support setup
LANGUAGES = {
    'English': 'en',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Italian': 'it',
    'Hindi': 'hi',
    'Chinese': 'zh-CN',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Arabic': 'ar'
}

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech

# Initialize translator
translator = GoogleTranslator()

# Class for handling voiceover in a separate thread
class VoiceoverManager:
    def __init__(self):
        self.current_text = ""
        self.last_spoken_text = ""
        self.last_spoken_time = 0
        self.cooldown = 2.0  # seconds between repeated voiceovers
        self.is_speaking = False
        self.thread = None
        self.language = 'en'
        
    def set_language(self, lang_code):
        self.language = lang_code
        
    def update_text(self, text):
        self.current_text = text
        
    def should_speak(self):
        current_time = time.time()
        if (self.current_text != self.last_spoken_text or 
            current_time - self.last_spoken_time > self.cooldown):
            return True
        return False
        
    def speak(self):
        if not self.is_speaking and self.current_text and self.should_speak():
            self.is_speaking = True
            self.last_spoken_text = self.current_text
            self.last_spoken_time = time.time()
            
            # Translate if not English
            text_to_speak = self.current_text
            if self.language != 'en':
                try:
                    text_to_speak = GoogleTranslator(source='en', target=self.language).translate(text=self.current_text)
                except Exception as e:
                    print(f"Translation error: {e}")
            
            # Start speaking in a separate thread
            self.thread = threading.Thread(target=self._speak_thread, args=(text_to_speak,))
            self.thread.daemon = True
            self.thread.start()
    
    def _speak_thread(self, text):
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {e}")
        self.is_speaking = False

# UI Elements
def create_language_selector(img, selected_index=0):
    height, width = img.shape[:2]
    lang_names = list(LANGUAGES.keys())
    
    # Draw dropdown background
    dropdown_width = 150
    dropdown_height = 30
    dropdown_x = width - dropdown_width - 10
    dropdown_y = 10
    
    cv2.rectangle(img, (dropdown_x, dropdown_y), 
                 (dropdown_x + dropdown_width, dropdown_y + dropdown_height), 
                 (200, 200, 200), cv2.FILLED)
    
    # Draw selected language
    cv2.putText(img, lang_names[selected_index], 
                (dropdown_x + 5, dropdown_y + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Draw dropdown arrow
    cv2.rectangle(img, (dropdown_x + dropdown_width - 20, dropdown_y), 
                 (dropdown_x + dropdown_width, dropdown_y + dropdown_height), 
                 (150, 150, 150), cv2.FILLED)
    cv2.putText(img, "▼", 
                (dropdown_x + dropdown_width - 15, dropdown_y + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return dropdown_x, dropdown_y, dropdown_width, dropdown_height

def handle_language_selection(event, x, y, flags, param):
    global current_lang_index, is_dropdown_open
    dropdown_x, dropdown_y, dropdown_width, dropdown_height = param['dropdown_coords']
    lang_names = list(LANGUAGES.keys())
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click is on the dropdown
        if (dropdown_x <= x <= dropdown_x + dropdown_width and 
            dropdown_y <= y <= dropdown_y + dropdown_height):
            is_dropdown_open = not is_dropdown_open
        # If dropdown is open, check if user selected a language
        elif is_dropdown_open:
            for i, lang in enumerate(lang_names):
                lang_y = dropdown_y + dropdown_height + (i * 30)
                if (dropdown_x <= x <= dropdown_x + dropdown_width and 
                    lang_y <= y <= lang_y + 30):
                    current_lang_index = i
                    voiceover_manager.set_language(list(LANGUAGES.values())[current_lang_index])
                    is_dropdown_open = False
                    break

def draw_dropdown_menu(img, selected_index):
    dropdown_x, dropdown_y, dropdown_width, dropdown_height = create_language_selector(img, selected_index)
    
    if is_dropdown_open:
        lang_names = list(LANGUAGES.keys())
        for i, lang in enumerate(lang_names):
            lang_y = dropdown_y + dropdown_height + (i * 30)
            # Draw background for each option
            cv2.rectangle(img, (dropdown_x, lang_y), 
                         (dropdown_x + dropdown_width, lang_y + 30), 
                         (220, 220, 220) if i == selected_index else (200, 200, 200), 
                         cv2.FILLED)
            # Draw language name
            cv2.putText(img, lang, 
                        (dropdown_x + 5, lang_y + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return (dropdown_x, dropdown_y, dropdown_width, dropdown_height)

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

# Initialize voiceover manager
voiceover_manager = VoiceoverManager()

# Initialize webcam and HandDetector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset, imgSize = 20, 224

# Initialize language selector variables
current_lang_index = 0  # Default to English
is_dropdown_open = False
dropdown_coords = (0, 0, 0, 0)

# Set up mouse callback
cv2.namedWindow("Image")
param = {'dropdown_coords': dropdown_coords}
cv2.setMouseCallback("Image", handle_language_selection, param)

# For sentence building
current_word = ""
sentence = ""
word_detected_time = 0
word_confidence_count = 0
last_prediction = ""
confidence_threshold = 3  # Number of consecutive detections needed

# Main loop
while True:
    success, img = cap.read()
    if not success:
        print("Failed to get frame from camera")
        break
        
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    
    # Update dropdown coordinates and draw UI
    dropdown_coords = draw_dropdown_menu(imgOutput, current_lang_index)
    param['dropdown_coords'] = dropdown_coords
    
    # Draw current sentence at the bottom
    translated_sentence = sentence
    if sentence and current_lang_index > 0:  # If not English
        try:
            target_lang = list(LANGUAGES.values())[current_lang_index]
            translated_sentence = GoogleTranslator(source='en', target=target_lang).translate(text=sentence)
        except Exception as e:
            print(f"Translation error: {e}")
    
    # Display sentence with background
    if translated_sentence:
        # Create background for subtitle
        text_size = cv2.getTextSize(translated_sentence, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        h_img, w_img, _ = imgOutput.shape
        text_x = (w_img - text_size[0]) // 2
        text_y = h_img - 30
        
        # Background rectangle
        cv2.rectangle(imgOutput, 
                     (text_x - 10, text_y - text_size[1] - 10), 
                     (text_x + text_size[0] + 10, text_y + 10), 
                     (0, 0, 0), cv2.FILLED)
        
        # Text
        cv2.putText(imgOutput, translated_sentence, 
                   (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        # Ensure cropping stays within bounds
        h_img, w_img, _ = img.shape
        y1, y2 = max(0, y - offset), min(h_img, y + h + offset)
        x1, x2 = max(0, x - offset), min(w_img, x + w + offset)
        
        # Check if the hand is fully visible
        if y1 < y2 and x1 < x2:
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
            imgWhite1 = imgWhite1 / 255.0  # Normalize
            prediction = model.predict(imgWhite1, verbose=0)
            index = np.argmax(prediction)
            confidence = prediction[0][index]
            
            # Process prediction
            predicted_label = labels[index]
            
            # Handle consecutive detections for confidence
            current_time = time.time()
            if predicted_label == last_prediction and confidence > 0.7:
                word_confidence_count += 1
            else:
                word_confidence_count = 0
                last_prediction = predicted_label
            
            # If we have enough confidence, update the current word
            if word_confidence_count >= confidence_threshold:
                if current_word != predicted_label:
                    current_word = predicted_label
                    word_detected_time = current_time
                    
                    # Append to sentence with space
                    if sentence:
                        sentence += " "
                    sentence += current_word
                    
                    # Set for voiceover
                    voiceover_manager.update_text(current_word)
                    voiceover_manager.speak()
                    
                    # Reset confidence to avoid multiple additions
                    word_confidence_count = 0
            
            # Display prediction in the original language
            cv2.rectangle(imgOutput, (x1, y1 - 50), (x1 + 200, y1), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, predicted_label, 
                       (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x1, y1), (x2, y2), (255, 0, 255), 4)
            
            # Show confidence
            confidence_text = f"Conf: {confidence:.2f}"
            cv2.putText(imgOutput, confidence_text, 
                       (x1, y1 + h + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)
    
    # Add buttons for controlling the sentence
    button_width = 150
    button_height = 40
    button_gap = 20
    button_start_x = 10
    button_y = 10
    
    # Clear button
    cv2.rectangle(imgOutput, 
                 (button_start_x, button_y), 
                 (button_start_x + button_width, button_y + button_height), 
                 (0, 0, 255), cv2.FILLED)
    cv2.putText(imgOutput, "Clear Sentence", 
               (button_start_x + 10, button_y + 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Speak full sentence button
    cv2.rectangle(imgOutput, 
                 (button_start_x + button_width + button_gap, button_y), 
                 (button_start_x + 2*button_width + button_gap, button_y + button_height), 
                 (0, 255, 0), cv2.FILLED)
    cv2.putText(imgOutput, "Speak Sentence", 
               (button_start_x + button_width + button_gap + 10, button_y + 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Check for button clicks
    def check_button_click(event, x, y, flags, param):
        global sentence
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Clear button
            if (button_start_x <= x <= button_start_x + button_width and 
                button_y <= y <= button_y + button_height):
                sentence = ""
            
            # Speak full sentence button
            elif (button_start_x + button_width + button_gap <= x <= button_start_x + 2*button_width + button_gap and 
                  button_y <= y <= button_y + button_height):
                if sentence:
                    # Translate if necessary
                    text_to_speak = sentence
                    if current_lang_index > 0:  # If not English
                        try:
                            target_lang = list(LANGUAGES.values())[current_lang_index]
                            text_to_speak = GoogleTranslator(source='en', target=target_lang).translate(text=sentence)
                        except Exception as e:
                            print(f"Translation error: {e}")
                    
                    # Update voiceover manager to speak the full sentence
                    voiceover_manager.update_text(text_to_speak)
                    voiceover_manager.speak()
    
    # Update mouse callback to handle both dropdown and buttons
    def combined_mouse_callback(event, x, y, flags, param):
        # Handle language selection
        handle_language_selection(event, x, y, flags, param)
        
        # Handle button clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            # Clear button
            if (button_start_x <= x <= button_start_x + button_width and 
                button_y <= y <= button_y + button_height):
                global sentence
                sentence = ""
            
            # Speak full sentence button
            elif (button_start_x + button_width + button_gap <= x <= button_start_x + 2*button_width + button_gap and 
                  button_y <= y <= button_y + button_height):
                if sentence:
                    # Translate if necessary
                    text_to_speak = sentence
                    if current_lang_index > 0:  # If not English
                        try:
                            target_lang = list(LANGUAGES.values())[current_lang_index]
                            text_to_speak = GoogleTranslator(source='en', target=target_lang).translate(text=sentence)
                        except Exception as e:
                            print(f"Translation error: {e}")
                    
                    # Update voiceover manager to speak the full sentence
                    voiceover_manager.update_text(text_to_speak)
                    voiceover_manager.speak()
    
    # Set the combined callback
    cv2.setMouseCallback("Image", combined_mouse_callback, param)
    
    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
    
    # Quit on 'q' press
    if key & 0xFF == ord('q'):
        break
    # Clear sentence on 'c' press
    elif key & 0xFF == ord('c'):
        sentence = ""
    # Space to speak the full sentence
    elif key & 0xFF == ord(' '):
        if sentence:
            # Translate if necessary
            text_to_speak = sentence
            if current_lang_index > 0:  # If not English
                try:
                    target_lang = list(LANGUAGES.values())[current_lang_index]
                    text_to_speak = GoogleTranslator(source='en', target=target_lang).translate(text=sentence)
                except Exception as e:
                    print(f"Translation error: {e}")
            
            # Update voiceover manager to speak the full sentence
            voiceover_manager.update_text(text_to_speak)
            voiceover_manager.speak()

cap.release()
cv2.destroyAllWindows()
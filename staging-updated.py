import os
import cv2
import math
import numpy as np
import tkinter as tk
from tkinter import Scrollbar, ttk, messagebox
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from cvzone.HandTrackingModule import HandDetector
import pyttsx3
import threading
import time
from pathlib import Path
from PIL import Image, ImageTk
import shutil
from deep_translator import GoogleTranslator
import google.generativeai as genai
from typing import List, Tuple
from dotenv import load_dotenv

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Interpreter")
        self.root.state('zoomed')  # Maximize window
        self.root.configure(bg="#f0f2f5")

        # Gemini API setup
        load_dotenv()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")  # Set this in your environment
        if not self.gemini_api_key:
            messagebox.showwarning("API Key Missing", "GOOGLE_API_KEY not set. Chatbot will not function.")
        else:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction="You are a sign language expert. Only answer questions related to sign language, such as its history, techniques, variations (e.g., ASL, BSL), learning tips, or this app's features. Politely decline to answer unrelated questions."
            )
        self.chat_history: List[Tuple[str, str]] = []
        
        # Chatbot UI variables
        self.chatbot_visible = False
        self.chatbot_frame = None

        # Language support
        self.languages = {
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
        self.current_language = 'English'
        
        # Initialize TTS engine with voice settings
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        # Set up voice properties for different languages
        self.voices = self.engine.getProperty('voices')
        self.voice_map = {}
        for voice in self.voices:
            for lang_name, lang_code in self.languages.items():
                if lang_code.lower() in voice.id.lower() or lang_code.split('-')[0].lower() in voice.id.lower():
                    self.voice_map[lang_code] = voice.id
                    break
        
        # Initialize variables
        self.cap = None
        self.detector = None
        self.model = None
        self.labels = []
        self.data_dir = os.path.join(os.path.expanduser("~"), "Desktop", "SLI", "data")
        self.model_path = os.path.join(os.path.expanduser("~"), "Desktop", "SLI", "model.keras")
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # For real-time detection
        self.is_detecting = False
        self.detection_thread = None
        self.current_word = ""
        self.sentence = ""
        self.word_confidence_count = 0
        self.last_prediction = ""
        self.confidence_threshold = 3
        
        # Create UI
        self.create_landing_page()

    def create_chatbot_icon(self):
        # Remove existing icon if any
        if hasattr(self, 'chatbot_icon') and self.chatbot_icon.winfo_exists():
            self.chatbot_icon.destroy()
        # Chatbot toggle button in bottom-left corner
        self.chatbot_icon = tk.Button(
            self.root,
            text="💬",
            font=("Arial", 16),
            bg="#3498db",
            fg="white",
            command=self.toggle_chatbot,
            relief=tk.FLAT
        )
        self.chatbot_icon.place(relx=0.01, rely=0.95, anchor=tk.SW)

    def toggle_chatbot(self):
        if self.chatbot_visible:
            self.chatbot_frame.destroy()
            self.chatbot_visible = False
        else:
            self.create_chatbot_window()
            self.chatbot_visible = True

    def create_chatbot_window(self):
        self.chatbot_frame = tk.Frame(self.root, bg="white", relief=tk.RIDGE, bd=2)
        self.chatbot_frame.place(
            relx=0.01,
            rely=0.95,
            anchor=tk.SW,
            width=400,
            height=300
        )

        chat_display_frame = tk.Frame(self.chatbot_frame, bg="white")
        chat_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))

        self.chat_display = tk.Text(
            chat_display_frame,
            height=10,
            width=50,
            wrap=tk.WORD,
            state='disabled',
            bg="#f8f9fa",
            font=("Arial", 10)
        )
        scrollbar = Scrollbar(chat_display_frame, command=self.chat_display.yview)
        self.chat_display.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        input_frame = tk.Frame(self.chatbot_frame, bg="white")
        input_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.chat_input = tk.Entry(input_frame, width=40, font=("Arial", 10))
        self.chat_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.chat_input.bind("<Return>", lambda event: self.send_chat_message())

        send_btn = tk.Button(
            input_frame,
            text="Send",
            bg="#3498db",
            fg="white",
            command=self.send_chat_message
        )
        send_btn.pack(side=tk.RIGHT)

        welcome_message = "Hello! I'm your Sign Language Assistant. Ask me anything about sign language, like how to learn ASL, the history of BSL, or tips for using this app!"
        self.chat_history.append(("Bot", welcome_message))
        self.update_chat_display()

    def update_chat_display(self):
        self.chat_display.config(state='normal')
        self.chat_display.delete(1.0, tk.END)
        for sender, message in self.chat_history:
            tag = "user" if sender == "User" else "bot"
            self.chat_display.insert(tk.END, f"{sender}: {message}\n", tag)
        self.chat_display.config(state='disabled')
        self.chat_display.yview(tk.END)
        self.chat_display.tag_configure("user", foreground="#2c3e50", font=("Arial", 10, "bold"))
        self.chat_display.tag_configure("bot", foreground="#3498db", font=("Arial", 10))

    def send_chat_message(self):
        user_input = self.chat_input.get().strip()
        if not user_input:
            return

        self.chat_input.delete(0, tk.END)
        self.chat_history.append(("User", user_input))
        self.update_chat_display()

        if self.gemini_api_key:
            threading.Thread(
                target=self.get_bot_response,
                args=(user_input,),
                daemon=True
            ).start()
        else:
            self.chat_history.append(("Bot", "Chatbot is disabled due to missing API key."))
            self.update_chat_display()

    def get_bot_response(self, user_input):
        try:
            response = self.gemini_model.generate_content(user_input)
            bot_response = response.text.strip()
            self.chat_history.append(("Bot", bot_response))
            self.root.after(0, self.update_chat_display)
        except Exception as e:
            error_msg = "Sorry, I couldn't process that. Please try again."
            self.chat_history.append(("Bot", error_msg))
            self.root.after(0, self.update_chat_display)
            print(f"Gemini API error: {e}")

    def create_landing_page(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        
        header_frame = tk.Frame(self.root, height=100, bg="#2980b9")
        header_frame.pack(fill=tk.X)
        
        canvas = tk.Canvas(header_frame, height=100, bg="#2980b9", highlightthickness=0)
        canvas.pack(fill=tk.X)
        
        for i in range(100):
            r = int(41 + (46-41) * i/100)
            g = int(128 + (204-128) * i/100)
            b = int(185 + (113-185) * i/100)
            color = f'#{r:02x}{g:02x}{b:02x}'
            canvas.create_line(0, i, 2000, i, fill=color)
        
        canvas.create_text(
            self.root.winfo_width()//2, 50, 
            text="Sign Language Interpreter", 
            font=("Arial", 24, "bold"), 
            fill="white"
        )
        
        canvas.create_text(
            self.root.winfo_width()//2, 80, 
            text="Bridging communication gaps through technology", 
            font=("Arial", 12), 
            fill="white"
        )
        
        content_frame = tk.Frame(self.root, bg="#f0f2f5", padx=20, pady=20)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)
        
        card1 = tk.Frame(content_frame, bg="white", padx=20, pady=20, relief=tk.RIDGE, bd=1)
        card1.grid(row=0, column=0, padx=10, pady=10, sticky=tk.NSEW)
        
        hand_icon = "👋"
        tk.Label(card1, text=hand_icon, font=("Arial", 48), bg="white", fg="#3498db").pack(pady=10)
        tk.Label(card1, text="Capture Sign", font=("Arial", 16, "bold"), bg="white").pack(pady=5)
        tk.Label(
            card1, 
            text="Create a dataset by capturing sign language gestures with your camera.", 
            font=("Arial", 10), 
            bg="white",
            wraplength=300
        ).pack(pady=10)
        
        get_started_btn = tk.Button(
            card1, 
            text="Get Started", 
            bg="#2c3e50", 
            fg="white",
            font=("Arial", 10, "bold"),
            width=20,
            height=2,
            command=self.start_capture_mode
        )
        get_started_btn.pack(pady=10)
        
        card2 = tk.Frame(content_frame, bg="white", padx=20, pady=20, relief=tk.RIDGE, bd=1)
        card2.grid(row=0, column=1, padx=10, pady=10, sticky=tk.NSEW)
        
        model_icon = "🧠"
        tk.Label(card2, text=model_icon, font=("Arial", 48), bg="white", fg="#27ae60").pack(pady=10)
        tk.Label(card2, text="Train Model", font=("Arial", 16, "bold"), bg="white").pack(pady=5)
        tk.Label(
            card2, 
            text="Train your custom machine learning model using captured gestures.", 
            font=("Arial", 10), 
            bg="white",
            wraplength=300
        ).pack(pady=10)
        
        train_btn = tk.Button(
            card2, 
            text="Train Now", 
            bg="#2c3e50", 
            fg="white",
            font=("Arial", 10, "bold"),
            width=20,
            height=2,
            command=self.start_train_mode
        )
        train_btn.pack(pady=10)
        
        card3 = tk.Frame(content_frame, bg="white", padx=20, pady=20, relief=tk.RIDGE, bd=1)
        card3.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky=tk.NSEW)
        
        camera_icon = "📹"
        tk.Label(card3, text=camera_icon, font=("Arial", 48), bg="white", fg="#e74c3c").pack(pady=10)
        tk.Label(card3, text="Real-time Detection", font=("Arial", 16, "bold"), bg="white").pack(pady=5)
        tk.Label(
            card3, 
            text="Detect and translate sign language in real-time using your webcam.", 
            font=("Arial", 10), 
            bg="white",
            wraplength=500
        ).pack(pady=10)
        
        start_detection_btn = tk.Button(
            card3, 
            text="Start Detection", 
            bg="#2c3e50", 
            fg="white",
            font=("Arial", 10, "bold"),
            width=20,
            height=2,
            command=self.start_detection_mode
        )
        start_detection_btn.pack(pady=10)
        
        footer_frame = tk.Frame(self.root, bg="#f0f2f5", pady=5)
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        footer_text = f"© 2025 AI Sign Language Interpreter. All rights reserved @Abiram Ramasamy."
        tk.Label(footer_frame, text=footer_text, fg="#7f8c8d", bg="#f0f2f5", font=("Arial", 8)).pack()
        
        if os.path.exists(self.model_path):
            try:
                self.model = keras.models.load_model(self.model_path)
                self.labels = sorted([d for d in os.listdir(self.data_dir) 
                                     if os.path.isdir(os.path.join(self.data_dir, d))])
                messagebox.showinfo("Model Loaded", "Existing model loaded successfully!")
            except Exception as e:
                print(f"Error loading model: {e}")
        
        self.create_chatbot_icon()

    def start_capture_mode(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        
        header_frame = tk.Frame(self.root, bg="#3498db", pady=10)
        header_frame.pack(fill=tk.X)
        
        back_btn = tk.Button(
            header_frame,
            text="← Back",
            bg="#2c3e50",
            fg="white",
            font=("Arial", 10),
            command=self.create_landing_page
        )
        back_btn.pack(side=tk.LEFT, padx=10)
        
        tk.Label(
            header_frame, 
            text="Capture Sign Language Gestures", 
            font=("Arial", 16, "bold"), 
            bg="#3498db", 
            fg="white"
        ).pack(pady=5)
        
        content_frame = tk.Frame(self.root, bg="#f0f2f5", padx=20, pady=20)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        control_frame = tk.Frame(content_frame, bg="white", padx=20, pady=20, relief=tk.RIDGE, bd=1)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        tk.Label(
            control_frame, 
            text="Add New Sign", 
            font=("Arial", 14, "bold"), 
            bg="white"
        ).pack(pady=10)
        
        tk.Label(control_frame, text="Sign Label:", bg="white").pack(anchor=tk.W, pady=(10, 0))
        self.label_entry = tk.Entry(control_frame, width=25)
        self.label_entry.pack(pady=(0, 10), fill=tk.X)
        
        tk.Label(control_frame, text="Number of Samples:", bg="white").pack(anchor=tk.W, pady=(10, 0))
        self.samples_entry = tk.Entry(control_frame, width=25)
        self.samples_entry.insert(0, "100")
        self.samples_entry.pack(pady=(0, 10), fill=tk.X)
        
        capture_btn = tk.Button(
            control_frame,
            text="Start Capturing",
            bg="#27ae60",
            fg="white",
            font=("Arial", 10, "bold"),
            command=self.start_capture
        )
        capture_btn.pack(pady=10, fill=tk.X)
        
        tk.Label(
            control_frame, 
            text="Existing Signs:", 
            font=("Arial", 12, "bold"), 
            bg="white"
        ).pack(anchor=tk.W, pady=(20, 5))
        
        list_frame = tk.Frame(control_frame, bg="white")
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.label_listbox = tk.Listbox(
            list_frame,
            width=25,
            height=10,
            yscrollcommand=scrollbar.set
        )
        self.label_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=self.label_listbox.yview)
        
        self.update_label_listbox()
        
        preview_frame = tk.Frame(content_frame, bg="white", padx=20, pady=20, relief=tk.RIDGE, bd=1)
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        tk.Label(
            preview_frame, 
            text="Camera Preview", 
            font=("Arial", 14, "bold"), 
            bg="white"
        ).pack(pady=10)
        
        self.preview_canvas = tk.Canvas(preview_frame, bg="black", width=640, height=480)
        self.preview_canvas.pack(pady=10)
        
        self.status_label = tk.Label(
            preview_frame, 
            text="Camera inactive", 
            bg="white", 
            fg="#7f8c8d",
            font=("Arial", 10)
        )
        self.status_label.pack(pady=5)
        
        self.create_chatbot_icon()

    def update_label_listbox(self):
        self.label_listbox.delete(0, tk.END)
        
        try:
            if os.path.exists(self.data_dir):
                labels = sorted([d for d in os.listdir(self.data_dir) 
                               if os.path.isdir(os.path.join(self.data_dir, d))])
                
                for label in labels:
                    img_dir = os.path.join(self.data_dir, label)
                    img_count = len([f for f in os.listdir(img_dir) 
                                   if f.endswith(('.png', '.jpg', '.jpeg'))])
                    self.label_listbox.insert(tk.END, f"{label} ({img_count} images)")
        except Exception as e:
            print(f"Error updating label list: {e}")

    def start_capture(self):
        label = self.label_entry.get().strip()
        if not label:
            messagebox.showerror("Error", "Please enter a label for the sign.")
            return
        
        try:
            num_samples = int(self.samples_entry.get())
            if num_samples <= 0:
                messagebox.showerror("Error", "Number of samples must be positive.")
                return
        except ValueError:
            messagebox.showerror("Error", "Invalid number of samples.")
            return
        
        label_dir = os.path.join(self.data_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            return
        
        self.detector = HandDetector(maxHands=1)
        
        self.status_label.config(text="Camera active. Preparing to capture...", fg="#e67e22")
        
        self.capture_thread = threading.Thread(
            target=self.capture_images,
            args=(label_dir, num_samples)
        )
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def capture_images(self, label_dir, num_samples):
        counter = 0
        offset = 20
        imgSize = residency

        self.root.after(0, lambda: self.status_label.config(
            text=f"Capturing images... (0/{num_samples})", 
            fg="#e74c3c"
        ))
        
        try:
            while counter < num_samples:
                success, img = self.cap.read()
                if not success:
                    continue
                
                hands, img = self.detector.findHands(img)
                
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']
                    
                    h_img, w_img, _ = img.shape
                    y1, y2 = max(0, y - offset), min(h_img, y + h + offset)
                    x1, x2 = max(0, x - offset), min(w_img, x + w + offset)
                    
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
                        
                        counter += 1
                        img_path = os.path.join(label_dir, f"{counter}.jpg")
                        cv2.imwrite(img_path, imgWhite)
                        
                        self.root.after(0, lambda count=counter: self.status_label.config(
                            text=f"Capturing images... ({count}/{num_samples})", 
                            fg="#e74c3c"
                        ))
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil)
                
                self.root.after(0, lambda img=img_tk: self.update_preview(img))
                
                time.sleep(0.1)
        
        except Exception as e:
            print(f"Error during capture: {e}")
            self.root.after(0, lambda: self.status_label.config(
                text=f"Error: {str(e)}", 
                fg="#e74c3c"
            ))
        
        finally:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
            self.root.after(0, lambda: self.status_label.config(
                text=f"Capture complete! {num_samples} images saved.", 
                fg="#27ae60"
            ))
            
            self.root.after(0, self.update_label_listbox)

    def update_preview(self, img):
        self.preview_canvas.img = img
        self.preview_canvas.create_image(0, 0, anchor=tk.NW, image=img)

    def start_train_mode(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        
        header_frame = tk.Frame(self.root, bg="#27ae60", pady=10)
        header_frame.pack(fill=tk.X)
        
        back_btn = tk.Button(
            header_frame,
            text="← Back",
            bg="#2c3e50",
            fg="white",
            font=("Arial", 10),
            command=self.create_landing_page
        )
        back_btn.pack(side=tk.LEFT, padx=10)
        
        tk.Label(
            header_frame, 
            text="Train Machine Learning Model", 
            font=("Arial", 16, "bold"), 
            bg="#27ae60", 
            fg="white"
        ).pack(pady=5)
        
        content_frame = tk.Frame(self.root, bg="#f0f2f5", padx=20, pady=20)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        train_frame = tk.Frame(content_frame, bg="white", padx=20, pady=20, relief=tk.RIDGE, bd=1)
        train_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(
            train_frame, 
            text="Training Parameters", 
            font=("Arial", 14, "bold"), 
            bg="white"
        ).pack(pady=10)
        
        param_frame = tk.Frame(train_frame, bg="white")
        param_frame.pack(pady=10, fill=tk.X)
        
        params = [
            ("Batch Size:", "64"),
            ("Epochs:", "10"),
            ("Test Split (%):", "20"),
            ("Learning Rate:", "0.001")
        ]
        
        self.param_entries = {}
        
        for i, (label_text, default_value) in enumerate(params):
            label = tk.Label(param_frame, text=label_text, bg="white")
            label.grid(row=i, column=0, sticky=tk.W, pady=5, padx=5)
            
            entry = tk.Entry(param_frame)
            entry.insert(0, default_value)
            entry.grid(row=i, column=1, sticky=tk.W+tk.E, pady=5, padx=5)
            
            self.param_entries[label_text] = entry
        
        stat_frame = tk.Frame(train_frame, bg="#f8f9fa", padx=10, pady=10, relief=tk.RIDGE, bd=1)
        stat_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(
            stat_frame, 
            text="Dataset Statistics", 
            font=("Arial", 12, "bold"), 
            bg="#f8f9fa"
        ).pack(anchor=tk.W)
        
        num_classes, total_images = self.get_dataset_stats()
        
        stats_text = f"""
        Number of Classes: {num_classes}
        Total Images: {total_images}
        """
        
        tk.Label(
            stat_frame, 
            text=stats_text, 
            bg="#f8f9fa", 
            justify=tk.LEFT
        ).pack(anchor=tk.W, pady=5)
        
        train_btn = tk.Button(
            train_frame,
            text="Start Training",
            bg="#27ae60",
            fg="white",
            font=("Arial", 12, "bold"),
            padx=20,
            pady=10,
            command=self.train_model
        )
        train_btn.pack(pady=20)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            train_frame, 
            orient=tk.HORIZONTAL, 
            length=400, 
            mode='determinate',
            variable=self.progress_var
        )
        self.progress_bar.pack(pady=10, fill=tk.X)
        
        self.train_status = tk.Label(
            train_frame, 
            text="Ready to train", 
            bg="white",
            font=("Arial", 10)
        )
        self.train_status.pack(pady=5)
        
        self.create_chatbot_icon()

    def get_dataset_stats(self):
        num_classes = 0
        total_images = 0
        
        try:
            if os.path.exists(self.data_dir):
                classes = [d for d in os.listdir(self.data_dir) 
                          if os.path.isdir(os.path.join(self.data_dir, d))]
                num_classes = len(classes)
                
                for cls in classes:
                    cls_path = os.path.join(self.data_dir, cls)
                    img_files = [f for f in os.listdir(cls_path) 
                               if f.endswith(('.png', '.jpg', '.jpeg'))]
                    total_images += len(img_files)
        except Exception as e:
            print(f"Error getting dataset stats: {e}")
        
        return num_classes, total_images

    def train_model(self):
        try:
            batch_size = int(self.param_entries["Batch Size:"].get())
            epochs = int(self.param_entries["Epochs:"].get())
            test_split = float(self.param_entries["Test Split (%):"].get()) / 100
            learning_rate = float(self.param_entries["Learning Rate:"].get())
            
            if batch_size <= 0 or epochs <= 0 or test_split <= 0 or test_split >= 1 or learning_rate <= 0:
                messagebox.showerror("Error", "Invalid parameters. All values must be positive and test split must be between 0 and 1.")
                return
            
        except ValueError:
            messagebox.showerror("Error", "Invalid parameters. Please enter numeric values.")
            return
        
        self.train_status.config(text="Loading dataset...", fg="#e67e22")
        self.progress_var.set(0)
        
        self.train_thread = threading.Thread(
            target=self.do_train_model,
            args=(batch_size, epochs, test_split, learning_rate)
        )
        self.train_thread.daemon = True
        self.train_thread.start()

    def do_train_model(self, batch_size, epochs, test_split, learning_rate):
        try:
            self.root.after(0, lambda: self.train_status.config(
                text="Loading and preprocessing images...", 
                fg="#e67e22"
            ))
            
            images = []
            labels = []
            label_map = {}
            label_counter = 0
            
            class_dirs = [d for d in os.listdir(self.data_dir) 
                         if os.path.isdir(os.path.join(self.data_dir, d))]
            
            for i, label in enumerate(sorted(class_dirs)):
                progress = (i / len(class_dirs)) * 20
                self.root.after(0, lambda p=progress: self.progress_var.set(p))
                
                label_path = os.path.join(self.data_dir, label)
                if label not in label_map:
                    label_map[label] = label_counter
                    label_counter += 1
                
                image_files = [f for f in os.listdir(label_path) 
                              if f.endswith(('.png', '.jpg', '.jpeg'))]
                
                for j, image_file in enumerate(image_files):
                    image_path = os.path.join(label_path, image_file)
                    img = cv2.imread(image_path)
                    if img is None:
                        continue
                    
                    img = cv2.resize(img, (224, 224))
                    images.append(img)
                    labels.append(label_map[label])
                    
                    if j % 10 == 0:
                        sub_progress = (i / len(class_dirs)) * 20 + (j / len(image_files)) * (20 / len(class_dirs))
                        self.root.after(0, lambda p=sub_progress: self.progress_var.set(p))
            
            X = np.array(images)
            y = np.array(labels)
            
            y = to_categorical(y)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_split, random_state=42
            )
            
            X_train, X_test = X_train / 255.0, X_test / 255.0
            
            self.root.after(0, lambda: self.train_status.config(
                text="Building model...", 
                fg="#e67e22"
            ))
            self.root.after(0, lambda: self.progress_var.set(25))
            
            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Conv2D(128, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(128, activation='relu'),
                Dense(len(label_map), activation='softmax')
            ])
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.root.after(0, lambda: self.train_status.config(
                text="Training model...", 
                fg="#e67e22"
            ))
            self.root.after(0, lambda: self.progress_var.set(30))
            
            class ProgressCallback(tf.keras.callbacks.Callback):
                def __init__(self, app, progress_start=30, progress_end=90):
                    self.app = app
                    self.progress_start = progress_start
                    self.progress_end = progress_end
                
                def on_epoch_begin(self, epoch, logs=None):
                    progress = self.progress_start + (epoch / epochs) * (self.progress_end - self.progress_start)
                    self.app.root.after(0, lambda p=progress: self.app.progress_var.set(p))
                    self.app.root.after(0, lambda e=epoch+1: self.app.train_status.config(
                        text=f"Training: Epoch {e}/{epochs}", 
                        fg="#e67e22"
                    ))
            
            progress_callback = ProgressCallback(self)
            model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_test, y_test),
                callbacks=[progress_callback]
            )
            
            self.root.after(0, lambda: self.train_status.config(
                text="Evaluating model...", 
                fg="#e67e22"
            ))
            self.root.after(0, lambda: self.progress_var.set(95))
            
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            
            model.save(self.model_path)
            
            inverse_label_map = {v: k for k, v in label_map.items()}
            
            self.labels = [inverse_label_map[i] for i in range(len(label_map))]
            self.model = model
            
            self.root.after(0, lambda: self.train_status.config(
                text=f"Training complete! Test accuracy: {test_acc:.2%}", 
                fg="#27ae60"
            ))
            self.root.after(0, lambda: self.progress_var.set(100))
            
            self.root.after(0, lambda: messagebox.showinfo(
                "Training Complete", 
                f"Model trained successfully!\n\nTest accuracy: {test_acc:.2%}\nTest loss: {test_loss:.4f}"
            ))
            
        except Exception as e:
            print(f"Error during training: {e}")
            self.root.after(0, lambda: self.train_status.config(
                text=f"Error: {str(e)}", 
                fg="#e74c3c"
            ))

    def start_detection_mode(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        
        header_frame = tk.Frame(self.root, bg="#e74c3c", pady=10)
        header_frame.pack(fill=tk.X)
        
        back_btn = tk.Button(
            header_frame,
            text="← Back",
            bg="#2c3e50",
            fg="white",
            font=("Arial", 10),
            command=self.create_landing_page
        )
        back_btn.pack(side=tk.LEFT, padx=10)
        
        tk.Label(
            header_frame, 
            text="Real-time Sign Language Detection", 
            font=("Arial", 16, "bold"), 
            bg="#e74c3c", 
            fg="white"
        ).pack(pady=5)
        
        content_frame = tk.Frame(self.root, bg="#f0f2f5")
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        video_frame = tk.Frame(content_frame, bg="white", padx=20, pady=20, relief=tk.RIDGE, bd=1)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(
            video_frame, 
            text="Camera Feed", 
            font=("Arial", 14, "bold"), 
            bg="white"
        ).pack(pady=5)
        
        self.video_canvas = tk.Canvas(video_frame, bg="black", width=640, height=480)
        self.video_canvas.pack(pady=10)
        
        self.detection_status = tk.Label(
            video_frame, 
            text="Detection not started", 
            bg="white", 
            fg="#7f8c8d",
            font=("Arial", 10)
        )
        self.detection_status.pack(pady=5)
        
        control_frame = tk.Frame(content_frame, bg="white", padx=20, pady=20, relief=tk.RIDGE, bd=1)
        control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10, expand=True)
        
        tk.Label(
            control_frame, 
            text="Detection Controls", 
            font=("Arial", 14, "bold"), 
            bg="white"
        ).pack(pady=5)
        
        tk.Label(
            control_frame, 
            text="Confidence Threshold:", 
            bg="white"
        ).pack(anchor=tk.W, pady=(10, 0))
        
        self.confidence_slider = tk.Scale(
            control_frame,
            from_=1,
            to=10,
            orient=tk.HORIZONTAL,
            length=200,
            bg="white"
        )
        self.confidence_slider.set(3)
        self.confidence_slider.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(
            control_frame, 
            text="Output Language:", 
            bg="white"
        ).pack(anchor=tk.W, pady=(10, 0))
        
        self.language_var = tk.StringVar()
        self.language_var.set("English")
        
        language_dropdown = ttk.Combobox(
            control_frame,
            textvariable=self.language_var,
            values=list(self.languages.keys()),
            state="readonly"
        )
        language_dropdown.pack(fill=tk.X, pady=(0, 10))
        
        self.detect_btn = tk.Button(
            control_frame,
            text="Start Detection",
            bg="#e74c3c",
            fg="white",
            font=("Arial", 10, "bold"),
            command=self.toggle_detection
        )
        self.detect_btn.pack(pady=10, fill=tk.X)
        
        clear_btn = tk.Button(
            control_frame,
            text="Clear Sentence",
            bg="#7f8c8d",
            fg="white",
            font=("Arial", 10),
            command=self.clear_sentence
        )
        clear_btn.pack(pady=5, fill=tk.X)
        
        speak_btn = tk.Button(
            control_frame,
            text="Speak Sentence",
            bg="#3498db",
            fg="white",
            font=("Arial", 10),
            command=self.speak_sentence
        )
        speak_btn.pack(pady=5, fill=tk.X)
        
        tk.Label(
            control_frame, 
            text="Detection Results", 
            font=("Arial", 14, "bold"), 
            bg="white"
        ).pack(pady=(20, 5))
        
        tk.Label(
            control_frame, 
            text="Current Word:", 
            bg="white",
            font=("Arial", 10, "bold")
        ).pack(anchor=tk.W, pady=(10, 0))
        
        self.current_word_label = tk.Label(
            control_frame, 
            text="None", 
            bg="#f8f9fa",
            font=("Arial", 16),
            width=20,
            height=2,
            relief=tk.RIDGE
        )
        self.current_word_label.pack(pady=(0, 10), fill=tk.X)
        
        tk.Label(
            control_frame, 
            text="Sentence:", 
            bg="white",
            font=("Arial", 10, "bold")
        ).pack(anchor=tk.W, pady=(10, 0))
        
        self.sentence_text = tk.Text(
            control_frame,
            wrap=tk.WORD,
            width=30,
            height=5,
            font=("Arial", 12)
        )
        self.sentence_text.pack(pady=(0, 10), fill=tk.BOTH, expand=True)
        
        if self.model is None and os.path.exists(self.model_path):
            try:
                self.model = keras.models.load_model(self.model_path)
                self.labels = sorted([d for d in os.listdir(self.data_dir) 
                                   if os.path.isdir(os.path.join(self.data_dir, d))])
                messagebox.showinfo("Model Loaded", "Existing model loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {e}")
        
        if self.model is None:
            messagebox.showwarning(
                "No Model", 
                "No trained model found. Please train a model first."
            )
        
        self.create_chatbot_icon()

    def toggle_detection(self):
        if self.is_detecting:
            self.is_detecting = False
            self.detect_btn.config(text="Start Detection", bg="#e74c3c")
            self.detection_status.config(text="Detection stopped", fg="#7f8c8d")
        else:
            if self.model is None:
                messagebox.showerror("Error", "No model loaded. Please train a model first.")
                return
            
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam.")
                return
            
            self.detector = HandDetector(maxHands=1)
            
            self.is_detecting = True
            self.detect_btn.config(text="Stop Detection", bg="#e74c3c")
            self.detection_status.config(text="Detection running...", fg="#e67e22")
            
            self.confidence_threshold = self.confidence_slider.get()
            
            self.detection_thread = threading.Thread(target=self.detect_signs)
            self.detection_thread.daemon = True
            self.detection_thread.start()

    def detect_signs(self):
        offset = 20
        imgSize = 224
        subtitle_text = ""
        subtitle_timer = 0
        subtitle_duration = 3
        predictions_window = []
        window_size = 5
        frame_skip = 2
        frame_count = 0

        try:
            while self.is_detecting:
                success, img = self.cap.read()
                if not success:
                    continue

                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                imgOutput = img.copy()
                hands, img = self.detector.findHands(img)

                if subtitle_text and time.time() < subtitle_timer:
                    subtitle_bg = imgOutput.copy()
                    cv2.rectangle(subtitle_bg, (0, imgOutput.shape[0]-60), 
                                (imgOutput.shape[1], imgOutput.shape[0]), (0, 0, 0), -1)
                    imgOutput = cv2.addWeighted(subtitle_bg, 0.7, imgOutput, 0.3, 0)
                    cv2.putText(imgOutput, subtitle_text, (20, imgOutput.shape[0]-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']
                    y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
                    x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)

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

                        imgWhite = imgWhite / 255.0
                        imgWhite = np.expand_dims(imgWhite, axis=0)
                        prediction = self.model.predict(imgWhite, verbose=0)
                        index = np.argmax(prediction)
                        confidence = prediction[0][index]

                        if len(self.labels) > index:
                            predicted_class = self.labels[index]
                            cv2.rectangle(imgOutput, (x-offset, y-offset), 
                                        (x+w+offset, y+h+offset), (255, 0, 255), 4)
                            cv2.putText(imgOutput, f"{predicted_class} ({confidence:.2f})",
                                    (x-offset, y-offset-10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9, (255, 0, 255), 2)

                            if confidence > 0.7:
                                predictions_window.append((predicted_class, confidence))
                                if len(predictions_window) > window_size:
                                    predictions_window.pop(0)
                                if len(predictions_window) == window_size:
                                    most_common = max(set([p[0] for p in predictions_window]), 
                                                    key=[p[0] for p in predictions_window].count)
                                    avg_confidence = sum(p[1] for p in predictions_window 
                                                    if p[0] == most_common) / window_size
                                    if avg_confidence > 0.8:
                                        selected_language = self.language_var.get()
                                        language_code = self.languages[selected_language]
                                        translated_word = most_common
                                        if selected_language != "English":
                                            try:
                                                translated = GoogleTranslator(source='en', target=language_code).translate(most_common)
                                                translated_word = translated or most_common
                                            except Exception:
                                                pass
                                        self.current_word = most_common
                                        self.root.after(0, lambda word=translated_word: self.update_current_word(word))
                                        subtitle_text = translated_word
                                        subtitle_timer = time.time() + subtitle_duration
                                        threading.Thread(target=self.do_speak, args=(translated_word,)).start()
                                        predictions_window.clear()

                img_rgb = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil)
                self.root.after(0, lambda img=img_tk: self.update_video(img))
                time.sleep(0.05)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Detection Error", str(e)))
        finally:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.root.after(0, lambda: self.detection_status.config(text="Detection stopped", fg="#7f8c8d"))

    def update_video(self, img):
        self.video_canvas.img = img
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=img)

    def update_current_word(self, word):
        self.current_word_label.config(text=word)
        
        if self.sentence:
            self.sentence += " " + word
        else:
            self.sentence = word
        
        self.sentence_text.delete(1.0, tk.END)
        self.sentence_text.insert(tk.END, self.sentence)

    def clear_sentence(self):
        self.sentence = ""
        self.current_word = ""
        
        self.sentence_text.delete(1.0, tk.END)
        self.current_word_label.config(text="None")

    def speak_sentence(self):
        sentence = self.sentence_text.get(1.0, tk.END).strip()
        
        if not sentence:
            messagebox.showinfo("Empty", "No text to speak.")
            return
        
        selected_language = self.language_var.get()
        language_code = self.languages[selected_language]
        
        if selected_language != "English":
            try:
                translated = GoogleTranslator(source='en', target=language_code).translate(sentence)
                if translated:
                    sentence = translated
            except Exception as e:
                print(f"Translation error: {e}")
        
        threading.Thread(target=self.do_speak, args=(sentence,)).start()

    def do_speak(self, text):
        try:
            selected_language = self.language_var.get()
            language_code = self.languages[selected_language]
            
            if language_code in self.voice_map:
                self.engine.setProperty('voice', self.voice_map[language_code])
            
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {e}")
            self.root.after(0, lambda: messagebox.showerror("TTS Error", str(e)))

if __name__ == "__main__":
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"GPU config error: {e}")
    
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()

import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import load_model

class ParkinsonsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Parkinson's Disease Detector")
        self.root.geometry("800x600")
        
        # Load model
        self.model_path = 'saved_models/cnn_model.h5'
        try:
            print(f"Loading model from {self.model_path}")
            self.model = load_model(self.model_path)
            print("Model loaded successfully")
            self.model_loaded = True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False
        
        # Setup UI
        self.setup_ui()
    
    def setup_ui(self):
        # Title
        title_label = tk.Label(self.root, text="Parkinson's Disease Detection", font=("Arial", 20))
        title_label.pack(pady=10)
        
        # Description
        descr_label = tk.Label(self.root, text="Upload a spiral or wave drawing to analyze")
        descr_label.pack(pady=5)
        
        # Warning if model not loaded
        if not self.model_loaded:
            warning_label = tk.Label(self.root, text="WARNING: Model not loaded!", fg="red", font=("Arial", 14))
            warning_label.pack(pady=5)
        
        # Frame for image display
        self.image_frame = tk.Frame(self.root, width=300, height=300, bg="lightgray")
        self.image_frame.pack(pady=10)
        
        # Default message in image frame
        self.default_msg = tk.Label(self.image_frame, text="Image will appear here", bg="lightgray")
        self.default_msg.place(relx=0.5, rely=0.5, anchor="center")
        
        # Button to upload image
        upload_btn = tk.Button(self.root, text="Upload Image", command=self.upload_image)
        upload_btn.pack(pady=10)
        
        # Results area
        result_label = tk.Label(self.root, text="Prediction Results:")
        result_label.pack(pady=5)
        
        self.result_var = tk.StringVar()
        self.result_var.set("No prediction yet")
        
        result_display = tk.Label(self.root, textvariable=self.result_var, font=("Arial", 16))
        result_display.pack(pady=5)
    
    def upload_image(self):
        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        
        if not file_path:
            return
        
        try:
            # Load and display image
            img = Image.open(file_path)
            img = img.resize((300, 300))
            img_tk = ImageTk.PhotoImage(img)
            
            # Remove default message
            self.default_msg.place_forget()
            
            # Create or update image label
            if hasattr(self, 'img_label'):
                self.img_label.config(image=img_tk)
                self.img_label.image = img_tk
            else:
                self.img_label = tk.Label(self.image_frame, image=img_tk)
                self.img_label.image = img_tk
                self.img_label.place(relx=0.5, rely=0.5, anchor="center")
            
            # Make prediction
            if self.model_loaded:
                result = self.predict_image(file_path)
                self.result_var.set(result)
            else:
                self.result_var.set("Error: Model not loaded")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error loading or processing image: {str(e)}")
            self.result_var.set(f"Error: {str(e)}")
    
    def predict_image(self, image_path):
        try:
            # Load and preprocess image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return "Error: Could not read image"
            
            # Resize
            img_resized = cv2.resize(img, (128, 128))
            
            # Normalize
            img_norm = img_resized.astype('float32') / 255.0
            
            # Reshape for model
            img_input = np.expand_dims(img_norm, axis=(0, -1))
            
            # Predict
            prediction = self.model.predict(img_input, verbose=0)
            pred_value = prediction[0][0]
            
            # Determine result
            if pred_value >= 0.5:
                result = "Parkinson's Disease"
                confidence = pred_value
            else:
                result = "Healthy"
                confidence = 1 - pred_value
            
            return f"{result} (Confidence: {confidence*100:.2f}%)"
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error during prediction: {str(e)}"

if __name__ == "__main__":
    root = tk.Tk()
    app = ParkinsonsApp(root)
    root.mainloop()

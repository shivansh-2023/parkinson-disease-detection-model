import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ParkinsonsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Parkinson's Disease Detector")
        self.root.geometry("1000x800")
        
        # Load model
        self.model_path = 'saved_models/cnn_model.h5'
        try:
            print(f"Loading model from {self.model_path}")
            self.model = load_model(self.model_path)
            self.model.summary()  # Display model architecture
            self.model_loaded = True
            
            # Testing model prediction on different values
            test_arr = np.array([[[0.1]]], dtype=np.float32)  # Black image
            white_arr = np.array([[[0.9]]], dtype=np.float32)  # White image
            
            # Test predictions on extreme values
            self.test_model_on_data(test_arr, "black image")
            self.test_model_on_data(white_arr, "white image")
            
            # Test with random noise
            random_arr = np.random.rand(1, 128, 128, 1).astype(np.float32)
            self.test_model_on_data(random_arr, "random noise")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False
        
        # Setup UI
        self.setup_ui()
    
    def test_model_on_data(self, data, description):
        """Test model prediction on sample data"""
        try:
            pred = self.model.predict(data, verbose=0)
            print(f"Test prediction on {description}: {pred[0][0]}")
            print(f"Class: {'Parkinson' if pred[0][0] >= 0.5 else 'Healthy'}")
        except Exception as e:
            print(f"Error testing on {description}: {e}")
    
    def setup_ui(self):
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.prediction_tab = ttk.Frame(notebook)
        self.debug_tab = ttk.Frame(notebook)
        
        notebook.add(self.prediction_tab, text="Prediction")
        notebook.add(self.debug_tab, text="Debug")
        
        # Set up prediction tab
        self.setup_prediction_tab()
        
        # Set up debug tab
        self.setup_debug_tab()
    
    def setup_prediction_tab(self):
        # Title
        title_label = tk.Label(self.prediction_tab, text="Parkinson's Disease Detection", font=("Arial", 20))
        title_label.pack(pady=10)
        
        # Description
        descr_label = tk.Label(self.prediction_tab, text="Upload a spiral or wave drawing to analyze")
        descr_label.pack(pady=5)
        
        # Warning if model not loaded
        if not self.model_loaded:
            warning_label = tk.Label(self.prediction_tab, text="WARNING: Model not loaded!", fg="red", font=("Arial", 14))
            warning_label.pack(pady=5)
        
        # Frame for image display
        self.image_frame = tk.Frame(self.prediction_tab, width=300, height=300, bg="lightgray")
        self.image_frame.pack(pady=10)
        
        # Default message in image frame
        self.default_msg = tk.Label(self.image_frame, text="Image will appear here", bg="lightgray")
        self.default_msg.place(relx=0.5, rely=0.5, anchor="center")
        
        # Frame for processed image
        self.processed_frame = tk.Frame(self.prediction_tab, width=300, height=300, bg="lightgray")
        self.processed_frame.pack(pady=10)
        
        # Default message in processed frame
        self.processed_msg = tk.Label(self.processed_frame, text="Processed image will appear here", bg="lightgray")
        self.processed_msg.place(relx=0.5, rely=0.5, anchor="center")
        
        # Button to upload image
        upload_btn = tk.Button(self.prediction_tab, text="Upload Image", command=self.upload_image)
        upload_btn.pack(pady=10)
        
        # Results area
        result_label = tk.Label(self.prediction_tab, text="Prediction Results:")
        result_label.pack(pady=5)
        
        self.result_var = tk.StringVar()
        self.result_var.set("No prediction yet")
        
        result_display = tk.Label(self.prediction_tab, textvariable=self.result_var, font=("Arial", 16))
        result_display.pack(pady=5)
    
    def setup_debug_tab(self):
        # Title
        title_label = tk.Label(self.debug_tab, text="Model Debugging", font=("Arial", 20))
        title_label.pack(pady=10)
        
        # Add controls for testing
        test_frame = tk.Frame(self.debug_tab)
        test_frame.pack(pady=10, fill=tk.X)
        
        tk.Label(test_frame, text="Generate test image:").pack(side=tk.LEFT, padx=5)
        
        # Add buttons for test scenarios
        tk.Button(test_frame, text="Test Empty Image", command=self.test_empty).pack(side=tk.LEFT, padx=5)
        tk.Button(test_frame, text="Test Spiral Pattern", command=self.test_spiral).pack(side=tk.LEFT, padx=5)
        tk.Button(test_frame, text="Test Random Noise", command=self.test_noise).pack(side=tk.LEFT, padx=5)
        
        # Frame for test plots
        self.plot_frame = tk.Frame(self.debug_tab)
        self.plot_frame.pack(pady=10, fill=tk.BOTH, expand=True)
    
    def test_empty(self):
        """Test with empty/black image"""
        empty_img = np.zeros((128, 128), dtype=np.float32)
        self.visualize_and_predict(empty_img, "Empty Image")
    
    def test_spiral(self):
        """Test with generated spiral pattern"""
        # Generate a synthetic spiral
        img = np.zeros((128, 128), dtype=np.float32)
        center = (64, 64)
        max_radius = 60
        
        for r in range(max_radius):
            angle = 0.5 * r
            x = int(center[0] + r * np.cos(angle))
            y = int(center[1] + r * np.sin(angle))
            if 0 <= x < 128 and 0 <= y < 128:
                cv2.circle(img, (x, y), 1, 1.0, -1)
        
        # Add some noise to make it more realistic
        noise = np.random.normal(0, 0.05, img.shape).astype(np.float32)
        noisy_spiral = np.clip(img + noise, 0, 1)
        
        self.visualize_and_predict(noisy_spiral, "Synthetic Spiral")
    
    def test_noise(self):
        """Test with random noise"""
        noise = np.random.rand(128, 128).astype(np.float32)
        self.visualize_and_predict(noise, "Random Noise")
    
    def visualize_and_predict(self, image, title):
        """Visualize test image and run prediction"""
        # Clear previous plot
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        # Create figure for visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(title)
        
        # Display original
        ax1.imshow(image, cmap='gray')
        ax1.set_title("Input Image")
        ax1.axis('off')
        
        # Prepare for model and display
        input_tensor = np.expand_dims(image, axis=(0, -1))
        
        # Get intermediate activations if model is available
        if self.model_loaded:
            # Make prediction
            prediction = self.model.predict(input_tensor, verbose=0)
            pred_value = prediction[0][0]
            
            # Determine result
            if pred_value >= 0.5:
                result = "Parkinson's Disease"
                confidence = pred_value
            else:
                result = "Healthy"
                confidence = 1 - pred_value
            
            result_text = f"{result} (Confidence: {confidence*100:.2f}%)"
            
            # Add prediction text to figure
            ax2.text(0.5, 0.5, result_text, 
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=ax2.transAxes,
                     fontsize=12)
            ax2.set_title("Prediction")
            ax2.axis('off')
        
        # Embed plot in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
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
            
            # Process and display the processed image
            processed = self.preprocess_image_for_display(file_path)
            
            # Remove processed message
            self.processed_msg.place_forget()
            
            # Create or update processed image label
            if hasattr(self, 'processed_label'):
                self.processed_label.config(image=processed)
                self.processed_label.image = processed
            else:
                self.processed_label = tk.Label(self.processed_frame, image=processed)
                self.processed_label.image = processed
                self.processed_label.place(relx=0.5, rely=0.5, anchor="center")
            
            # Make prediction
            if self.model_loaded:
                result = self.predict_image(file_path)
                self.result_var.set(result)
            else:
                self.result_var.set("Error: Model not loaded")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error loading or processing image: {str(e)}")
            self.result_var.set(f"Error: {str(e)}")
    
    def preprocess_image_for_display(self, image_path):
        """Preprocess image and return a version for display"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (300, 300))
        
        # Convert to PIL format for display
        img_pil = Image.fromarray(img_resized)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        return img_tk
    
    def predict_image(self, image_path):
        try:
            # Load and preprocess image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return "Error: Could not read image"
            
            # Save original shape for debugging
            orig_shape = img.shape
            print(f"Original image shape: {orig_shape}")
            
            # Resize
            img_resized = cv2.resize(img, (128, 128))
            print(f"Resized shape: {img_resized.shape}")
            
            # Print min/max values before normalization
            print(f"Before normalization - Min: {img_resized.min()}, Max: {img_resized.max()}")
            
            # Normalize
            img_norm = img_resized.astype('float32') / 255.0
            
            # Print min/max values after normalization
            print(f"After normalization - Min: {img_norm.min()}, Max: {img_norm.max()}")
            
            # Reshape for model
            img_input = np.expand_dims(img_norm, axis=(0, -1))
            print(f"Input tensor shape: {img_input.shape}")
            
            # Predict
            prediction = self.model.predict(img_input, verbose=0)
            pred_value = prediction[0][0]
            print(f"Raw prediction value: {pred_value}")
            
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

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf
from tensorflow.keras.models import load_model

class DiagnosticApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Parkinson's Disease Detector - Diagnostic Tool")
        self.root.geometry("1200x800")
        
        # Available model types
        self.model_types = {
            "Original CNN": "saved_models/cnn_model.h5",
            "Enhanced CNN": "saved_models/enhanced/enhanced_cnn_final.h5",
            "EfficientNet": "saved_models/enhanced/efficientnet_final.h5",
            "MobileNet": "saved_models/enhanced/mobilenet_final.h5",
            "Ensemble": "saved_models/enhanced/ensemble_final.h5"
        }
        
        # Initialize models dict
        self.models = {}
        
        # Load models
        self.load_models()
        
        # Setup UI
        self.setup_ui()
    
    def load_models(self):
        """Load available models"""
        for name, path in self.model_types.items():
            if os.path.exists(path):
                try:
                    print(f"Loading {name} model from {path}")
                    self.models[name] = load_model(path)
                    print(f"{name} model loaded successfully")
                except Exception as e:
                    print(f"Error loading {name} model: {e}")
        
        # Give feedback on loaded models
        if not self.models:
            print("No models could be loaded.")
        else:
            print(f"Successfully loaded {len(self.models)} models: {list(self.models.keys())}")
    
    def setup_ui(self):
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.comparison_tab = ttk.Frame(notebook)
        self.test_shapes_tab = ttk.Frame(notebook)
        
        notebook.add(self.comparison_tab, text="Model Comparison")
        notebook.add(self.test_shapes_tab, text="Test Patterns")
        
        # Set up comparison tab
        self.setup_comparison_tab()
        
        # Set up test shapes tab
        self.setup_test_shapes_tab()
    
    def setup_comparison_tab(self):
        # Top section for user controls
        control_frame = tk.Frame(self.comparison_tab)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Model selection
        model_label = tk.Label(control_frame, text="Select Models to Compare:")
        model_label.pack(side=tk.LEFT, padx=5)
        
        # Checkbuttons for models
        self.model_vars = {}
        models_frame = tk.Frame(control_frame)
        models_frame.pack(side=tk.LEFT, padx=5)
        
        for i, model_name in enumerate(self.model_types.keys()):
            var = tk.BooleanVar(value=model_name in self.models)
            self.model_vars[model_name] = var
            chk = tk.Checkbutton(models_frame, text=model_name, variable=var, 
                                 state=tk.NORMAL if model_name in self.models else tk.DISABLED)
            chk.grid(row=0, column=i, padx=5)
        
        # Image upload button
        upload_btn = tk.Button(control_frame, text="Upload Image", command=self.upload_image)
        upload_btn.pack(side=tk.RIGHT, padx=5)
        
        # Middle section for image display
        self.image_frame = tk.Frame(self.comparison_tab)
        self.image_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Default message in image frame
        self.image_label = tk.Label(self.image_frame, text="Upload an image to compare model predictions")
        self.image_label.pack(pady=20)
        
        # Bottom section for results
        self.results_frame = tk.Frame(self.comparison_tab)
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Placeholder for results
        self.results_text = tk.Text(self.results_frame, height=10, width=80)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Initially show models status
        self.update_results_text("Model Status:\n\n")
        for name in self.model_types.keys():
            status = "Loaded" if name in self.models else "Not available"
            self.update_results_text(f"{name}: {status}\n")
    
    def setup_test_shapes_tab(self):
        # Control panel
        control_frame = tk.Frame(self.test_shapes_tab)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Model selection
        model_label = tk.Label(control_frame, text="Select Model:")
        model_label.pack(side=tk.LEFT, padx=5)
        
        self.selected_model = tk.StringVar()
        model_options = list(self.models.keys())
        if model_options:
            self.selected_model.set(model_options[0])
        else:
            self.selected_model.set("No models available")
            model_options = ["No models available"]
        
        model_dropdown = ttk.Combobox(control_frame, textvariable=self.selected_model, 
                                      values=model_options, state="readonly")
        model_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Generate patterns buttons
        patterns_frame = tk.Frame(control_frame)
        patterns_frame.pack(side=tk.RIGHT, padx=5)
        
        tk.Button(patterns_frame, text="Test Empty", command=lambda: self.test_pattern("empty")).pack(side=tk.LEFT, padx=5)
        tk.Button(patterns_frame, text="Test Spiral", command=lambda: self.test_pattern("spiral")).pack(side=tk.LEFT, padx=5)
        tk.Button(patterns_frame, text="Test Wave", command=lambda: self.test_pattern("wave")).pack(side=tk.LEFT, padx=5)
        tk.Button(patterns_frame, text="Test Noise", command=lambda: self.test_pattern("noise")).pack(side=tk.LEFT, padx=5)
        tk.Button(patterns_frame, text="Test All", command=lambda: self.test_all_patterns()).pack(side=tk.LEFT, padx=5)
        
        # Results area - using matplotlib
        self.plots_frame = tk.Frame(self.test_shapes_tab)
        self.plots_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def update_results_text(self, text, clear=False):
        if clear:
            self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, text)
        self.results_text.see(tk.END)
    
    def upload_image(self):
        """Upload and process an image"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")])
        
        if not file_path:
            return
        
        try:
            # Load the image
            image = Image.open(file_path)
            
            # Display the image
            self.display_image(image)
            
            # Process and predict using the selected models
            self.process_and_predict(image)
        
        except Exception as e:
            messagebox.showerror("Error", f"Error processing image: {str(e)}")
    
    def display_image(self, image):
        """Display the image in the UI"""
        # Clear the image frame
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        
        # Resize image for display if needed
        display_img = image.copy()
        display_img.thumbnail((400, 400))
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(display_img)
        
        # Display image
        img_label = tk.Label(self.image_frame, image=photo)
        img_label.image = photo  # Keep a reference
        img_label.pack(pady=10)
    
    def preprocess_image(self, image):
        """Preprocess the image for model prediction"""
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            if img_array.shape[2] == 4:  # RGBA
                pil_img = Image.fromarray(img_array)
                pil_img = pil_img.convert('RGB')
                img_array = np.array(pil_img)
            
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Resize to model input size
        img_array = cv2.resize(img_array, (128, 128))
        
        # Normalize to [0, 1]
        img_array = img_array.astype('float32') / 255.0
        
        # Add batch and channel dimensions for model
        processed_img = np.expand_dims(img_array, axis=(0, -1))
        
        return processed_img
    
    def process_and_predict(self, image):
        """Process image and run predictions for selected models"""
        # Clear previous results
        self.update_results_text("Prediction Results:\n\n", clear=True)
        
        # Check which models are selected
        selected_models = [name for name, var in self.model_vars.items() 
                           if var.get() and name in self.models]
        
        if not selected_models:
            self.update_results_text("No models selected for comparison.")
            return
        
        # Preprocess the image
        processed_img = self.preprocess_image(image)
        
        # Make predictions with each model
        for model_name in selected_models:
            try:
                model = self.models[model_name]
                
                # Get model prediction
                prediction = model.predict(processed_img, verbose=0)
                
                # Process prediction
                pred_value = prediction[0][0]
                result_class = "Parkinson's Disease" if pred_value >= 0.5 else "Healthy"
                confidence = pred_value if pred_value >= 0.5 else 1 - pred_value
                
                # Update results text
                self.update_results_text(f"{model_name} Model:\n")
                self.update_results_text(f"  Prediction: {result_class}\n")
                self.update_results_text(f"  Confidence: {confidence*100:.2f}%\n")
                self.update_results_text(f"  Raw value: {pred_value}\n\n")
            
            except Exception as e:
                self.update_results_text(f"{model_name} Model: Error - {str(e)}\n\n")
    
    def test_pattern(self, pattern_type):
        """Test a specific pattern"""
        # Check if a model is selected
        model_name = self.selected_model.get()
        if model_name == "No models available" or model_name not in self.models:
            messagebox.showerror("Error", "No valid model selected")
            return
        
        # Generate the specified pattern
        if pattern_type == "empty":
            img = np.zeros((128, 128), dtype=np.float32)
            title = "Empty (Black) Image"
        
        elif pattern_type == "spiral":
            img = np.zeros((128, 128), dtype=np.float32)
            center = (64, 64)
            max_radius = 60
            
            for r in range(max_radius):
                angle = 0.5 * r
                x = int(center[0] + r * np.cos(angle))
                y = int(center[1] + r * np.sin(angle))
                if 0 <= x < 128 and 0 <= y < 128:
                    cv2.circle(img, (x, y), 1, 1.0, -1)
            
            title = "Spiral Pattern"
        
        elif pattern_type == "wave":
            img = np.zeros((128, 128), dtype=np.float32)
            
            for x in range(128):
                y = int(64 + 30 * np.sin(x * 0.05))
                if 0 <= y < 128:
                    cv2.circle(img, (x, y), 1, 1.0, -1)
            
            title = "Wave Pattern"
        
        elif pattern_type == "noise":
            img = np.random.rand(128, 128).astype(np.float32)
            title = "Random Noise"
        
        else:
            messagebox.showerror("Error", f"Unknown pattern type: {pattern_type}")
            return
        
        # Display the pattern and prediction
        self.visualize_pattern_prediction(img, title, model_name)
    
    def test_all_patterns(self):
        """Test all patterns with selected model"""
        # Clear previous plots
        for widget in self.plots_frame.winfo_children():
            widget.destroy()
        
        # Check if a model is selected
        model_name = self.selected_model.get()
        if model_name == "No models available" or model_name not in self.models:
            messagebox.showerror("Error", "No valid model selected")
            return
        
        # Create figure
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        fig.suptitle(f"Model Predictions: {model_name}", fontsize=16)
        
        # Generate patterns
        patterns = {
            "Empty": np.zeros((128, 128), dtype=np.float32),
            "Spiral": self.generate_spiral(),
            "Wave": self.generate_wave(),
            "Noise": np.random.rand(128, 128).astype(np.float32)
        }
        
        # Make predictions for each pattern
        for i, (name, img) in enumerate(patterns.items()):
            # Display pattern
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(name)
            axes[i].axis('off')
            
            # Get prediction
            input_tensor = np.expand_dims(img, axis=(0, -1))
            prediction = self.models[model_name].predict(input_tensor, verbose=0)[0][0]
            
            # Format prediction text
            result = "Parkinson" if prediction >= 0.5 else "Healthy"
            confidence = prediction if prediction >= 0.5 else 1 - prediction
            pred_text = f"{result}\n{confidence*100:.1f}%"
            
            # Add prediction as subtitle
            axes[i].set_xlabel(pred_text, fontsize=12)
        
        # Display in UI
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.plots_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def generate_spiral(self):
        """Generate a spiral pattern"""
        img = np.zeros((128, 128), dtype=np.float32)
        center = (64, 64)
        max_radius = 60
        
        for r in range(max_radius):
            angle = 0.5 * r
            x = int(center[0] + r * np.cos(angle))
            y = int(center[1] + r * np.sin(angle))
            if 0 <= x < 128 and 0 <= y < 128:
                cv2.circle(img, (x, y), 1, 1.0, -1)
        
        return img
    
    def generate_wave(self):
        """Generate a wave pattern"""
        img = np.zeros((128, 128), dtype=np.float32)
        
        for x in range(128):
            y = int(64 + 30 * np.sin(x * 0.05))
            if 0 <= y < 128:
                cv2.circle(img, (x, y), 1, 1.0, -1)
        
        return img
    
    def visualize_pattern_prediction(self, img, title, model_name):
        """Visualize a pattern and its prediction"""
        # Clear previous plots
        for widget in self.plots_frame.winfo_children():
            widget.destroy()
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f"{title} - {model_name} Model", fontsize=16)
        
        # Display pattern
        ax1.imshow(img, cmap='gray')
        ax1.set_title("Pattern")
        ax1.axis('off')
        
        # Get prediction
        input_tensor = np.expand_dims(img, axis=(0, -1))
        prediction = self.models[model_name].predict(input_tensor, verbose=0)[0][0]
        
        # Display prediction
        if prediction >= 0.5:
            result = "Parkinson's Disease"
            confidence = prediction
        else:
            result = "Healthy"
            confidence = 1 - prediction
        
        # Format prediction text
        result_text = f"Prediction: {result}\nConfidence: {confidence*100:.2f}%\nRaw Value: {prediction:.6f}"
        
        # Display prediction as text in second subplot
        ax2.text(0.5, 0.5, result_text, horizontalalignment='center',
                verticalalignment='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_title("Prediction Result")
        ax2.axis('off')
        
        # Display in UI
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.plots_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = DiagnosticApp(root)
    root.mainloop()

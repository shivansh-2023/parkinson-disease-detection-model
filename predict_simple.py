import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import gradio as gr
from PIL import Image
import traceback

class ParkinsonPredictor:
    def __init__(self, model_path='saved_models/cnn_model.h5', image_size=(128, 128)):
        self.image_size = image_size
        try:
            # Check if model exists
            if os.path.exists(model_path):
                print(f"Loading model from {model_path}")
                self.model = load_model(model_path)
                print("Model loaded successfully")
            else:
                print(f"Warning: Model not found at {model_path}")
                self.model = None
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            traceback.print_exc()
            self.model = None
    
    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        try:
            # Convert PIL image to numpy array
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image
            
            # Convert to grayscale if it's a color image
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Resize to target size
            resized = cv2.resize(gray, self.image_size)
            
            # Normalize pixel values to [0, 1]
            normalized = resized.astype('float32') / 255.0
            
            # Add batch and channel dimensions
            input_tensor = np.expand_dims(normalized, axis=(0, -1))  # (1, height, width, 1)
            
            return input_tensor
        
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            traceback.print_exc()
            return None
    
    def predict(self, image):
        """Make prediction on an image"""
        try:
            if self.model is None:
                return "Error: Model not loaded", 0.0
            
            # Preprocess the image
            input_tensor = self.preprocess_image(image)
            if input_tensor is None:
                return "Error: Failed to preprocess image", 0.0
            
            # Make prediction
            prediction = self.model.predict(input_tensor, verbose=0)
            prediction_value = prediction[0][0]
            
            # Convert to class and confidence
            if prediction_value >= 0.5:
                result = "Parkinson's Disease"
                confidence = float(prediction_value)
            else:
                result = "Healthy"
                confidence = float(1 - prediction_value)
            
            return result, confidence
        
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            traceback.print_exc()
            return f"Error: {str(e)}", 0.0

def create_interface():
    predictor = ParkinsonPredictor()
    
    def predict_image(image):
        if image is None:
            return "Please upload an image"
        
        try:
            result, confidence = predictor.predict(image)
            confidence_percentage = f"{confidence * 100:.2f}%"
            return f"{result} (Confidence: {confidence_percentage})"
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Create Gradio interface
    demo = gr.Interface(
        fn=predict_image,
        inputs=gr.Image(type="pil"),
        outputs=gr.Textbox(),
        title="Parkinson's Disease Detection",
        description="Upload a spiral or wave drawing to detect Parkinson's disease."
    )
    
    return demo

if __name__ == "__main__":
    try:
        print("Initializing Parkinson's Disease Detection interface...")
        demo = create_interface()
        print("Launching interface...")
        demo.launch(share=False)
        print("Interface launched successfully!")
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()

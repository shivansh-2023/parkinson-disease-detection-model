import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model
import cv2
import gradio as gr
from skimage.feature import hog
from PIL import Image
import traceback
from config import *

class ParkinsonsPredictor:
    """Class for making predictions on new Parkinson's disease data"""
    
    def __init__(self, model_dir='saved_models', image_size=(128, 128)):
        """
        Initialize the predictor
        
        Args:
            model_dir: Directory containing saved models
            image_size: Size to resize input images to
        """
        self.model_dir = model_dir
        self.image_size = image_size
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            # Load CNN model if it exists
            cnn_path = os.path.join(self.model_dir, 'cnn_model.h5')
            if os.path.exists(cnn_path):
                print(f"Loading CNN model from {cnn_path}")
                self.models['cnn'] = load_model(cnn_path)
                print("CNN model loaded successfully")
            
            # Load Random Forest model if it exists
            rf_path = os.path.join(self.model_dir, 'rf_model.pkl')
            if os.path.exists(rf_path):
                print(f"Loading Random Forest model from {rf_path}")
                self.models['rf'] = joblib.load(rf_path)
                print("Random Forest model loaded successfully")
            
            # Load SVM model if it exists
            svm_path = os.path.join(self.model_dir, 'svm_model.pkl')
            if os.path.exists(svm_path):
                print(f"Loading SVM model from {svm_path}")
                self.models['svm'] = joblib.load(svm_path)
                print("SVM model loaded successfully")
            
            # Load Gradient Boosting model if it exists
            gb_path = os.path.join(self.model_dir, 'gb_model.pkl')
            if os.path.exists(gb_path):
                print(f"Loading Gradient Boosting model from {gb_path}")
                self.models['gb'] = joblib.load(gb_path)
                print("Gradient Boosting model loaded successfully")
            
            # Load Hybrid model if it exists
            hybrid_path = os.path.join(self.model_dir, 'hybrid_model.h5')
            if os.path.exists(hybrid_path):
                print(f"Loading Hybrid model from {hybrid_path}")
                self.models['hybrid'] = load_model(hybrid_path)
                print("Hybrid model loaded successfully")
            
            if not self.models:
                print("No models found. Please train models first.")
        
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            traceback.print_exc()
    
    def preprocess_image(self, image_path, is_synthetic=False):
        """
        Preprocess an image for prediction
        
        Args:
            image_path: Path to the image or a PIL Image
            is_synthetic: Whether the image is synthetic (already grayscale)
            
        Returns:
            Preprocessed image array
        """
        try:
            if isinstance(image_path, str):
                # Load image from path
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError(f"Could not read image from {image_path}")
            elif isinstance(image_path, (Image.Image, np.ndarray)):
                # Convert PIL Image to OpenCV format or use numpy array directly
                if isinstance(image_path, Image.Image):
                    image_path = np.array(image_path)
                
                # Convert to grayscale if not already
                if len(image_path.shape) == 3 and image_path.shape[2] == 3:
                    image = cv2.cvtColor(image_path, cv2.COLOR_RGB2GRAY)
                else:
                    image = image_path
            else:
                raise ValueError(f"Unsupported image type: {type(image_path)}")
            
            # Resize the image
            image = cv2.resize(image, self.image_size)
            
            # Normalize to [0, 1]
            image = image.astype('float32') / 255.0
            
            # Return preprocessed image for different models
            return {
                'cnn': np.expand_dims(image, axis=(0, -1)),  # (1, height, width, 1)
                'rf': self.extract_hog_features(image),       # HOG features for traditional ML
                'svm': self.extract_hog_features(image),      # HOG features for traditional ML
                'gb': self.extract_hog_features(image),       # HOG features for traditional ML
                'hybrid': {
                    'image': np.expand_dims(image, axis=(0, -1)),  # Image input
                    'clinical': np.zeros((1, 5))  # Dummy clinical data, replace with actual data if available
                }
            }
        
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            traceback.print_exc()
            return None
    
    def extract_hog_features(self, image):
        """
        Extract HOG features from an image
        
        Args:
            image: Grayscale image array
            
        Returns:
            HOG features as a 1D array
        """
        # Extract HOG features
        features, _ = hog(
            image, 
            orientations=9, 
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), 
            visualize=True, 
            block_norm='L1',
            transform_sqrt=True
        )
        
        # Reshape to 2D array for sklearn models
        return features.reshape(1, -1)
    
    def predict(self, image_path, model_type='cnn', is_synthetic=False, clinical_data=None):
        """
        Make a prediction on a new image
        
        Args:
            image_path: Path to the image or a PIL Image
            model_type: Type of model to use for prediction
            is_synthetic: Whether the image is synthetic (already grayscale)
            clinical_data: Clinical data for hybrid model
            
        Returns:
            Prediction result and confidence score
        """
        try:
            if model_type not in self.models:
                available_models = list(self.models.keys())
                if not available_models:
                    return "Error", "No models loaded. Please train models first."
                
                # Use the first available model
                model_type = available_models[0]
                print(f"Model '{model_type}' not found. Using '{model_type}' instead.")
            
            # Preprocess the image
            preprocessed = self.preprocess_image(image_path, is_synthetic)
            if preprocessed is None:
                return "Error", "Failed to preprocess image."
            
            # Make prediction based on model type
            if model_type in ['cnn']:
                prediction = self.models[model_type].predict(preprocessed[model_type], verbose=0)
                prediction_value = prediction[0][0]
                
                # Convert to class and confidence
                predicted_class = 'Parkinson' if prediction_value >= 0.5 else 'Healthy'
                confidence = prediction_value if prediction_value >= 0.5 else 1 - prediction_value
                
                return predicted_class, float(confidence)
            
            elif model_type in ['rf', 'svm', 'gb']:
                # Get probability prediction for traditional ML models
                pred_proba = self.models[model_type].predict_proba(preprocessed[model_type])
                prediction_value = pred_proba[0][1]  # Probability of class 1 (Parkinson's)
                
                # Convert to class and confidence
                predicted_class = 'Parkinson' if prediction_value >= 0.5 else 'Healthy'
                confidence = prediction_value if prediction_value >= 0.5 else 1 - prediction_value
                
                return predicted_class, float(confidence)
            
            elif model_type == 'hybrid':
                # Update clinical data if provided
                if clinical_data is not None:
                    preprocessed['hybrid']['clinical'] = np.array(clinical_data).reshape(1, -1)
                
                # Make prediction with hybrid model
                prediction = self.models[model_type].predict([
                    preprocessed['hybrid']['image'],
                    preprocessed['hybrid']['clinical']
                ], verbose=0)
                
                prediction_value = prediction[0][0]
                
                # Convert to class and confidence
                predicted_class = 'Parkinson' if prediction_value >= 0.5 else 'Healthy'
                confidence = prediction_value if prediction_value >= 0.5 else 1 - prediction_value
                
                return predicted_class, float(confidence)
            
            else:
                return "Error", f"Unsupported model type: {model_type}"
        
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            traceback.print_exc()
            return "Error", str(e)

def create_gradio_interface():
    """Create a Gradio interface for the Parkinson's disease prediction"""
    # Initialize the predictor
    predictor = ParkinsonsPredictor()
    
    # Ensure we have at least one model loaded
    if not predictor.models:
        print("Warning: No models were loaded. Creating a simple interface with error message.")
        
        # Define a simple prediction function that returns an error
        def predict_error(_):
            return "Error: No trained models found. Please train models first."
        
        # Create a simple interface
        app = gr.Interface(
            fn=predict_error,
            inputs=gr.Image(type="pil", label="Drawing Image"),
            outputs=gr.Textbox(label="Result"),
            title="Parkinson's Disease Detection",
            description="No models available. Please train models first."
        )
        return app
    
    # Define the prediction function for Gradio
    def predict_parkinsons(image, model_type):
        if image is None:
            return "Please upload an image"
        
        try:
            result, confidence = predictor.predict(image, model_type=model_type)
            
            # Format the confidence as a percentage
            if isinstance(confidence, float):
                confidence_str = f"{confidence * 100:.2f}%"
            else:
                confidence_str = str(confidence)
            
            # Return formatted result
            if result == "Error":
                return f"Error: {confidence_str}"
            else:
                return f"Prediction: {result} (Confidence: {confidence_str})"
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            traceback.print_exc()
            return f"Error during prediction: {str(e)}"
    
    # Available models
    available_models = list(predictor.models.keys())
    default_model = available_models[0] if available_models else "None"
    
    # Create a simpler Gradio interface
    app = gr.Interface(
        fn=predict_parkinsons,
        inputs=[
            gr.Image(type="pil", label="Drawing Image"),
            gr.Dropdown(
                choices=available_models,
                value=default_model,
                label="Model Type"
            )
        ],
        outputs=gr.Textbox(label="Prediction Result"),
        title="Parkinson's Disease Detection",
        description="Upload a spiral or wave drawing to detect Parkinson's disease."
    )
    
    return app

if __name__ == "__main__":
    try:
        print("Creating Parkinson's Disease Detection interface...")
        app = create_gradio_interface()
        print("Launching interface...")
        app.launch(share=False)
        print("Interface launched successfully!")
    except Exception as e:
        print(f"Error launching application: {str(e)}")
        traceback.print_exc()

import os
import gradio as gr
import numpy as np
import cv2
from predict import ParkinsonsPredictor

class WebInterface:
    def __init__(self, model_dir='saved_models'):
        # Create models directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Check if models exist
        if not any(fname.endswith(('.h5', '.pkl')) for fname in os.listdir(model_dir)):
            raise FileNotFoundError(
                f"No trained models found in {model_dir}. "
                "Please train models first using 'python train.py'"
            )
        
        self.predictor = ParkinsonsPredictor(model_dir)

    def preprocess_image(self, image):
        """Preprocess the uploaded image"""
        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to model input size
        image = cv2.resize(image, (128, 128))
        
        # Normalize
        image = image.astype('float32') / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=-1)
        
        return image

    def predict_parkinsons(self, image):
        """Make prediction on the uploaded image"""
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image)
            
            # Make prediction
            prediction, confidence = self.predictor.predict(processed_image)
            
            # Format output
            result = "Parkinson's Disease" if prediction == 1 else "Healthy"
            confidence_percent = confidence * 100
            
            return f"Prediction: {result}\nConfidence: {confidence_percent:.2f}%"
        except Exception as e:
            return f"Error making prediction: {str(e)}"

    def launch(self, share=False):
        """Launch the web interface"""
        # Create interface
        interface = gr.Interface(
            fn=self.predict_parkinsons,
            inputs=gr.Image(label="Upload Spiral/Wave Drawing"),
            outputs=gr.Textbox(label="Prediction"),
            title="Parkinson's Disease Detection",
            description="Upload a spiral or wave drawing to detect Parkinson's disease",
            examples=[
                [os.path.join("data", "sample", "spiral", "healthy_1.jpg")],
                [os.path.join("data", "sample", "wave", "parkinsons_1.jpg")]
            ],
            allow_flagging="never"
        )
        
        # Launch interface
        interface.launch(share=share)

if __name__ == "__main__":
    try:
        web_interface = WebInterface()
        web_interface.launch(share=True)
    except FileNotFoundError as e:
        print(e)

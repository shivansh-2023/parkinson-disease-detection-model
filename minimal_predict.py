import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import gradio as gr

# Load the model
model_path = 'saved_models/cnn_model.h5'
if os.path.exists(model_path):
    print(f"Loading model from {model_path}")
    model = load_model(model_path)
    print("Model loaded successfully")
else:
    print(f"Model not found at {model_path}")
    model = None

# Define prediction function
def predict(image):
    if model is None:
        return "Error: Model not loaded"
    
    if image is None:
        return "Error: No image provided"
    
    try:
        # Convert to numpy array if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Convert to grayscale
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Resize
        resized = cv2.resize(gray, (128, 128))
        
        # Normalize
        normalized = resized.astype('float32') / 255.0
        
        # Prepare for model
        input_tensor = np.expand_dims(normalized, axis=(0, -1))
        
        # Predict
        prediction = model.predict(input_tensor, verbose=0)
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

# Create interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="Parkinson's Disease Detection",
    description="Upload a spiral or wave drawing to detect Parkinson's disease"
)

# Launch the interface
if __name__ == "__main__":
    print("Starting Gradio interface...")
    try:
        iface.launch(server_name="127.0.0.1", server_port=7861)
        print("Interface launched successfully!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error launching interface: {str(e)}")

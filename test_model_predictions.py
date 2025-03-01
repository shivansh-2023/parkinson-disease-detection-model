import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import load_model

def test_model(model_path):
    print(f"Testing model: {model_path}")
    
    try:
        # Load model
        model = load_model(model_path)
        print("Model loaded successfully")
        
        # Create test images
        black_img = np.zeros((1, 128, 128, 1), dtype=np.float32)
        white_img = np.ones((1, 128, 128, 1), dtype=np.float32)
        random_img = np.random.rand(1, 128, 128, 1).astype(np.float32)
        
        # Generate a spiral
        spiral_img = np.zeros((128, 128), dtype=np.float32)
        center = (64, 64)
        for r in range(60):
            angle = 0.5 * r
            x = int(center[0] + r * np.cos(angle))
            y = int(center[1] + r * np.sin(angle))
            if 0 <= x < 128 and 0 <= y < 128:
                spiral_img[y, x] = 1.0
        spiral_img = np.expand_dims(spiral_img, axis=(0, -1))
        
        # Get predictions
        black_pred = model.predict(black_img, verbose=0)[0][0]
        white_pred = model.predict(white_img, verbose=0)[0][0]
        random_pred = model.predict(random_img, verbose=0)[0][0]
        spiral_pred = model.predict(spiral_img, verbose=0)[0][0]
        
        print(f"Black image prediction: {black_pred} - {'Parkinson' if black_pred >= 0.5 else 'Healthy'}")
        print(f"White image prediction: {white_pred} - {'Parkinson' if white_pred >= 0.5 else 'Healthy'}")
        print(f"Random noise prediction: {random_pred} - {'Parkinson' if random_pred >= 0.5 else 'Healthy'}")
        print(f"Spiral image prediction: {spiral_pred} - {'Parkinson' if spiral_pred >= 0.5 else 'Healthy'}")
        
        # Check if all predictions are almost the same
        predictions = [black_pred, white_pred, random_pred, spiral_pred]
        if max(predictions) - min(predictions) < 0.1:
            print("\nISSUE DETECTED: All predictions are very similar, indicating the model is not distinguishing between inputs!")
            print("Possible reasons:")
            print("1. Model needs to be retrained with more diverse data")
            print("2. Model architecture may be too simple for the task")
            print("3. Issue with the data preprocessing")
            print("4. Model may have converged to always predict a single class")
        else:
            print("\nModel shows variability in predictions across different inputs.")
    
    except Exception as e:
        print(f"Error testing model: {e}")

if __name__ == "__main__":
    # Test the CNN model
    cnn_model_path = 'saved_models/cnn_model.h5'
    if os.path.exists(cnn_model_path):
        test_model(cnn_model_path)
    else:
        print(f"Model not found: {cnn_model_path}")
    
    # Check if enhanced models exist and test them
    enhanced_model_dir = 'saved_models/enhanced'
    if os.path.exists(enhanced_model_dir):
        for model_file in os.listdir(enhanced_model_dir):
            if model_file.endswith('.h5'):
                model_path = os.path.join(enhanced_model_dir, model_file)
                test_model(model_path)

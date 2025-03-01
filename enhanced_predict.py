import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import logging
import sys
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("parkinsons_prediction.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("EnhancedParkinsonsPredictor")

class EnhancedParkinsonsPredictor:
    """Enhanced predictor for Parkinson's disease from drawings"""
    
    def __init__(self, model_dir="saved_models/enhanced", image_size=(128, 128)):
        """
        Initialize the predictor
        
        Args:
            model_dir: Directory containing saved models
            image_size: Size to resize input images to
        """
        self.model_dir = model_dir
        self.image_size = image_size
        self.models = {}
        
        # Load available models
        self.load_models()
    
    def load_models(self):
        """Load all available models"""
        logger.info(f"Loading models from {self.model_dir}...")
        
        # Check for enhanced CNN model
        enhanced_cnn_path = os.path.join(self.model_dir, "enhanced_cnn_final.h5")
        if os.path.exists(enhanced_cnn_path):
            try:
                logger.info(f"Loading enhanced CNN model from {enhanced_cnn_path}")
                self.models["enhanced_cnn"] = load_model(enhanced_cnn_path)
                logger.info("Enhanced CNN model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading enhanced CNN model: {str(e)}")
        
        # Check for MobileNet model
        mobilenet_path = os.path.join(self.model_dir, "mobilenet_final.h5")
        if os.path.exists(mobilenet_path):
            try:
                logger.info(f"Loading MobileNet model from {mobilenet_path}")
                self.models["mobilenet"] = load_model(mobilenet_path)
                logger.info("MobileNet model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading MobileNet model: {str(e)}")
        
        # Check for EfficientNet model
        efficientnet_path = os.path.join(self.model_dir, "efficientnet_final.h5")
        if os.path.exists(efficientnet_path):
            try:
                logger.info(f"Loading EfficientNet model from {efficientnet_path}")
                self.models["efficientnet"] = load_model(efficientnet_path)
                logger.info("EfficientNet model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading EfficientNet model: {str(e)}")
        
        # Check for Ensemble model
        ensemble_path = os.path.join(self.model_dir, "ensemble_final.h5")
        if os.path.exists(ensemble_path):
            try:
                logger.info(f"Loading Ensemble model from {ensemble_path}")
                self.models["ensemble"] = load_model(ensemble_path)
                logger.info("Ensemble model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading Ensemble model: {str(e)}")
        
        # Also look for older model formats
        legacy_model_dir = "saved_models"
        if os.path.exists(legacy_model_dir):
            cnn_path = os.path.join(legacy_model_dir, "cnn_model.h5")
            if os.path.exists(cnn_path) and "cnn" not in self.models:
                try:
                    logger.info(f"Loading legacy CNN model from {cnn_path}")
                    self.models["cnn"] = load_model(cnn_path)
                    logger.info("Legacy CNN model loaded successfully")
                except Exception as e:
                    logger.error(f"Error loading legacy CNN model: {str(e)}")
        
        # Log loaded models
        if self.models:
            logger.info(f"Successfully loaded {len(self.models)} models: {list(self.models.keys())}")
        else:
            logger.warning("No models were loaded. Please train models first.")
    
    def preprocess_image(self, image):
        """
        Preprocess an image for prediction
        
        Args:
            image: Image to preprocess (PIL Image, numpy array, or path)
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Handle different input types
            if isinstance(image, str):
                # Load from path
                img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError(f"Could not read image from {image}")
                logger.info(f"Loaded image from path: {image}")
            
            elif isinstance(image, Image.Image):
                # Convert PIL image to numpy array
                img = np.array(image)
                logger.info("Converted PIL image to numpy array")
                
                # Convert to grayscale if needed
                if len(img.shape) == 3:
                    if img.shape[2] == 4:  # RGBA
                        # Convert RGBA to RGB first
                        pil_img = Image.fromarray(img)
                        pil_img = pil_img.convert('RGB')
                        img = np.array(pil_img)
                    
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    logger.info("Converted RGB image to grayscale")
            
            elif isinstance(image, np.ndarray):
                # Convert to grayscale if needed
                if len(image.shape) == 3:
                    if image.shape[2] == 4:  # RGBA
                        # Convert RGBA to RGB first
                        pil_img = Image.fromarray(image)
                        pil_img = pil_img.convert('RGB')
                        image = np.array(pil_img)
                    
                    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    logger.info("Converted RGB array to grayscale")
                else:
                    img = image
                    logger.info("Using provided numpy array (assumed grayscale)")
            
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            # Resize to target size
            img_resized = cv2.resize(img, self.image_size)
            logger.info(f"Resized image to {self.image_size}")
            
            # Ensure pixels are in [0, 255] range
            if img_resized.max() <= 1.0:
                img_resized = (img_resized * 255).astype(np.uint8)
            
            # Log stats before normalization
            logger.info(f"Image stats before normalization - Min: {img_resized.min()}, Max: {img_resized.max()}, Mean: {img_resized.mean()}")
            
            # Normalize to [0, 1]
            img_norm = img_resized.astype('float32') / 255.0
            
            # Create the formatted input for each model type
            input_tensors = {}
            
            # For CNN-based models, add channel dimension
            for model_type in ["enhanced_cnn", "cnn", "ensemble"]:
                if model_type in self.models:
                    input_tensors[model_type] = np.expand_dims(img_norm, axis=(0, -1))
            
            # For MobileNet and EfficientNet, convert to RGB and add batch dimension
            for model_type in ["mobilenet", "efficientnet"]:
                if model_type in self.models:
                    # Convert to RGB (3 channels)
                    img_rgb = np.stack([img_norm, img_norm, img_norm], axis=-1)
                    input_tensors[model_type] = np.expand_dims(img_rgb, axis=0)
            
            if not input_tensors:
                logger.warning("No models are loaded, but preprocessing completed")
            else:
                logger.info(f"Prepared input tensors for {len(input_tensors)} models")
            
            return input_tensors
        
        except Exception as e:
            logger.error(f"Error in preprocessing image: {str(e)}", exc_info=True)
            return None
    
    def predict(self, image, model_type=None):
        """
        Make a prediction on an image
        
        Args:
            image: Image to predict on
            model_type: Specific model to use, or None to use all available models
            
        Returns:
            Prediction results
        """
        try:
            # Check if any models are loaded
            if not self.models:
                return {"error": "No models are loaded. Please train models first."}
            
            # If model_type is specified, check if it's available
            if model_type and model_type not in self.models:
                available_models = list(self.models.keys())
                if not available_models:
                    return {"error": "No models loaded. Please train models first."}
                
                # Use the first available model
                model_type = available_models[0]
                logger.warning(f"Requested model '{model_type}' not found. Using '{model_type}' instead.")
            
            # Preprocess the image
            input_tensors = self.preprocess_image(image)
            if input_tensors is None:
                return {"error": "Failed to preprocess image."}
            
            # Initialize results dictionary
            results = {}
            
            # Make predictions with each model
            for model_name, model in self.models.items():
                # Skip if specific model requested and this isn't it
                if model_type and model_name != model_type:
                    continue
                
                # Check if we have input tensor for this model
                if model_name not in input_tensors:
                    logger.warning(f"No input tensor prepared for model {model_name}")
                    continue
                
                # Make prediction
                logger.info(f"Making prediction with {model_name} model")
                prediction = model.predict(input_tensors[model_name], verbose=0)
                
                # Log raw prediction
                logger.info(f"{model_name} raw prediction: {prediction[0][0]}")
                
                # Process prediction value
                pred_value = prediction[0][0]
                
                # Determine result class and confidence
                if pred_value >= 0.5:
                    result_class = "Parkinson's Disease"
                    confidence = float(pred_value)
                else:
                    result_class = "Healthy"
                    confidence = float(1 - pred_value)
                
                # Store result
                results[model_name] = {
                    "class": result_class,
                    "confidence": confidence,
                    "raw_value": float(pred_value)
                }
            
            return results
        
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def visualize_prediction(self, image, results):
        """
        Create visualization of the prediction
        
        Args:
            image: Original image
            results: Prediction results
            
        Returns:
            Matplotlib figure
        """
        try:
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Prepare image for display
            if isinstance(image, str):
                # Load from path
                display_img = cv2.imread(image)
                display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            elif isinstance(image, Image.Image):
                # Convert PIL image to numpy array
                display_img = np.array(image)
            else:
                # Use numpy array directly
                display_img = image
                if len(display_img.shape) == 2:  # Grayscale
                    display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)
                elif display_img.shape[2] == 4:  # RGBA
                    display_img = display_img[:, :, :3]  # Drop alpha channel
            
            # Display the image
            ax1.imshow(display_img)
            ax1.set_title("Input Image")
            ax1.axis('off')
            
            # Display results
            if "error" in results:
                ax2.text(0.5, 0.5, f"Error: {results['error']}", 
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform=ax2.transAxes,
                         color='red',
                         fontsize=12)
            else:
                # Format results text
                results_text = ""
                for model_name, result in results.items():
                    confidence_pct = result["confidence"] * 100
                    results_text += f"{model_name}: {result['class']} ({confidence_pct:.2f}%)\n"
                
                ax2.text(0.5, 0.5, results_text, 
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform=ax2.transAxes,
                         fontsize=12,
                         linespacing=1.5)
            
            ax2.set_title("Prediction Results")
            ax2.axis('off')
            
            plt.tight_layout()
            return fig
        
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}", exc_info=True)
            # Create error figure
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"Error creating visualization: {str(e)}", 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes,
                    color='red')
            ax.axis('off')
            return fig

def create_gradio_interface():
    """Create a Gradio interface for prediction"""
    # Initialize predictor
    predictor = EnhancedParkinsonsPredictor()
    
    # Get available models
    available_models = list(predictor.models.keys())
    if not available_models:
        available_models = ["No models available"]
    
    # Define prediction function
    def predict_image(image, model_type):
        if image is None:
            return "Please upload an image", None
        
        if model_type == "No models available":
            return "No models are available. Please train models first.", None
        
        # Make prediction
        results = predictor.predict(image, model_type if model_type != "All models" else None)
        
        # Create visualization
        fig = predictor.visualize_prediction(image, results)
        
        # Format text output
        if "error" in results:
            return f"Error: {results['error']}", fig
        else:
            output_text = ""
            for model_name, result in results.items():
                confidence_pct = result["confidence"] * 100
                output_text += f"Model: {model_name}\n"
                output_text += f"Prediction: {result['class']}\n"
                output_text += f"Confidence: {confidence_pct:.2f}%\n"
                output_text += "-" * 30 + "\n"
            
            return output_text, fig
    
    # Create Gradio interface
    with gr.Blocks(title="Enhanced Parkinson's Disease Detection") as app:
        gr.Markdown("# Enhanced Parkinson's Disease Detection")
        gr.Markdown("Upload a spiral or wave drawing to detect Parkinson's disease")
        
        with gr.Row():
            with gr.Column():
                # Input components
                input_image = gr.Image(type="pil", label="Upload Drawing")
                model_dropdown = gr.Dropdown(
                    choices=["All models"] + available_models if len(available_models) > 0 and available_models[0] != "No models available" else available_models,
                    value="All models" if len(available_models) > 0 and available_models[0] != "No models available" else available_models[0],
                    label="Select Model"
                )
                predict_button = gr.Button("Predict")
            
            with gr.Column():
                # Output components
                output_text = gr.Textbox(label="Prediction Results")
                output_plot = gr.Plot(label="Visualization")
        
        # Set up event handlers
        predict_button.click(
            predict_image,
            inputs=[input_image, model_dropdown],
            outputs=[output_text, output_plot]
        )
        
        # Instructions
        gr.Markdown("""
        ## Instructions
        1. Upload a spiral or wave drawing image (hand-drawn or digital)
        2. Select which model to use for prediction (or use all available models)
        3. Click the "Predict" button to analyze the image
        
        This enhanced model analyzes the patterns in your drawing to detect potential signs of Parkinson's disease.
        """)
    
    return app

if __name__ == "__main__":
    # Create Gradio interface
    interface = create_gradio_interface()
    
    # Launch interface
    interface.launch()

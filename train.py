import os
import argparse
import numpy as np
import tensorflow as tf
import joblib
import logging
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from data_loader import ParkinsonDataLoader
from model import ParkinsonsDetector
from config import *
import seaborn as sns
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("parkinsons_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ParkinsonsTraining")

def train_model(model_type="cnn", data_dir="data", epochs=50, batch_size=32, save_dir="saved_models"):
    """
    Train a model for Parkinson's detection
    
    Args:
        model_type: Type of model to train (cnn, rf, svm, gb, vgg16, resnet50, hybrid)
        data_dir: Directory containing the data
        epochs: Number of epochs for deep learning models
        batch_size: Batch size for deep learning models
        save_dir: Directory to save trained models
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize data loader
    data_loader = ParkinsonDataLoader()
    
    # Load appropriate data based on model type
    if model_type in ["cnn", "vgg16", "resnet50"]:
        # Load synthetic image dataset
        logger.info(f"Loading synthetic image dataset for {model_type} model...")
        dataset_path = os.path.join(data_dir, "synthetic_images")
        (train_images, train_labels, train_patterns), (test_images, test_labels, test_patterns) = \
            data_loader.load_synthetic_image_dataset(dataset_path)
        
        # Get corresponding test data for the specified pattern
        spiral_indices = np.where(test_patterns == "spiral")[0]
        test_spiral_images = test_images[spiral_indices]
        test_spiral_labels = test_labels[spiral_indices]
        
        # Initialize model
        detector = ParkinsonsDetector()
        
        # Build and train model
        if model_type == "cnn":
            model = detector.build_cnn_model()
        elif model_type == "vgg16":
            model = detector.build_transfer_model(base_model="vgg16")
        elif model_type == "resnet50":
            model = detector.build_transfer_model(base_model="resnet50")
            
        logger.info(f"Training {model_type} model...")
        callbacks = detector.get_callbacks()
        
        # Train the model
        history = detector.train_model(
            model_name=model_type,
            X_train=train_images, 
            y_train=train_labels,
            X_val=test_spiral_images,
            y_val=test_spiral_labels,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks
        )
        
        # Plot training history
        plot_history(history, model_type)
        
        # Evaluate the model
        logger.info(f"Evaluating {model_type} model...")
        eval_results = detector.evaluate_model(model_type, test_spiral_images, test_spiral_labels)
        
        # Save the model
        model_path = os.path.join(save_dir, f"{model_type}_model.h5")
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
    elif model_type in ["rf", "svm", "gb"]:
        # Load synthetic image dataset
        logger.info(f"Loading synthetic image dataset for {model_type} model...")
        dataset_path = os.path.join(data_dir, "synthetic_images")
        (train_images, train_labels, train_patterns), (test_images, test_labels, test_patterns) = \
            data_loader.load_synthetic_image_dataset(dataset_path)
        
        # Get corresponding test data for the specified pattern
        spiral_indices = np.where(test_patterns == "spiral")[0]
        test_spiral_images = test_images[spiral_indices]
        test_spiral_labels = test_labels[spiral_indices]
        
        # Extract HOG features
        logger.info("Extracting HOG features...")
        train_hog = data_loader.load_hog_features(train_images)
        test_hog = data_loader.load_hog_features(test_spiral_images)
        
        # Initialize model
        detector = ParkinsonsDetector()
        
        # Build and train traditional ML models
        logger.info(f"Training {model_type} model...")
        
        if model_type == "rf":
            model = detector.build_random_forest(train_hog, train_labels)
        elif model_type == "svm":
            model = detector.build_svm(train_hog, train_labels)
        elif model_type == "gb":
            model = detector.build_gradient_boosting(train_hog, train_labels)
            
        # Evaluate the model
        logger.info(f"Evaluating {model_type} model...")
        y_pred = model.predict(test_hog)
        
        # Print evaluation metrics
        accuracy = accuracy_score(test_spiral_labels, y_pred)
        logger.info(f"Accuracy: {accuracy:.4f}")
        
        # Print classification report
        logger.info("Classification Report:")
        logger.info(classification_report(test_spiral_labels, y_pred))
        
        # Plot confusion matrix
        plot_confusion_matrix(test_spiral_labels, y_pred, model_type)
        
        # Save the model
        model_path = os.path.join(save_dir, f"{model_type}_model.pkl")
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
    elif model_type == "hybrid":
        # Load synthetic image dataset
        logger.info("Loading synthetic image dataset for hybrid model...")
        dataset_path = os.path.join(data_dir, "synthetic_images")
        (train_images, train_labels, train_patterns), (test_images, test_labels, test_patterns) = \
            data_loader.load_synthetic_image_dataset(dataset_path)
        
        # Get corresponding test data for the specified pattern
        spiral_indices = np.where(test_patterns == "spiral")[0]
        test_spiral_images = test_images[spiral_indices]
        test_spiral_labels = test_labels[spiral_indices]
        
        # Load clinical data
        logger.info("Loading clinical data for hybrid model...")
        clinical_path = os.path.join(data_dir, "parkinson_clinical_data.csv")
        clinical_data = data_loader.load_clinical_data(clinical_path)
        
        # Create matched clinical features
        # Note: This is a simplified example - in reality you'd need to match patients properly
        clinical_features = CLINICAL_FEATURES
        
        # Generate synthetic clinical data matching the number of images
        train_clinical = np.random.rand(len(train_images), len(clinical_features))
        test_clinical = np.random.rand(len(test_spiral_images), len(clinical_features))
        
        # Initialize model
        detector = ParkinsonsDetector()
        
        # Build and train hybrid model
        logger.info("Training hybrid model...")
        model = detector.build_hybrid_model(
            image_shape=IMAGE_SHAPE,
            num_clinical_features=len(clinical_features)
        )
        
        callbacks = detector.get_callbacks()
        
        # Train the model
        history = detector.train_hybrid_model(
            X_train_img=train_images,
            X_train_clinical=train_clinical,
            y_train=train_labels,
            X_val_img=test_spiral_images,
            X_val_clinical=test_clinical,
            y_val=test_spiral_labels,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks
        )
        
        # Plot training history
        plot_history(history, model_type)
        
        # Evaluate the model
        logger.info("Evaluating hybrid model...")
        eval_results = detector.evaluate_hybrid_model(
            X_test_img=test_spiral_images,
            X_test_clinical=test_clinical,
            y_test=test_spiral_labels
        )
        
        # Save the model
        model_path = os.path.join(save_dir, f"{model_type}_model.h5")
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
    else:
        logger.error(f"Unsupported model type: {model_type}")
        return
    
    logger.info(f"Training completed for {model_type} model")
    
def plot_history(history, model_type):
    """Plot training history for deep learning models"""
    # Create figure directory if it doesn't exist
    fig_dir = "figures"
    os.makedirs(fig_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{model_type} Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_type} Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.tight_layout()
    plt.savefig(os.path.join("figures", f"{model_type}_history_{timestamp}.png"))
    plt.close()
    
def plot_confusion_matrix(y_true, y_pred, model_type):
    """Plot confusion matrix for model evaluation"""
    # Create figure directory if it doesn't exist
    fig_dir = "figures"
    os.makedirs(fig_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy', 'Parkinson'],
                yticklabels=['Healthy', 'Parkinson'])
    plt.title(f'{model_type} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.tight_layout()
    plt.savefig(os.path.join("figures", f"{model_type}_confusion_{timestamp}.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train models for Parkinson's detection")
    parser.add_argument("--model", type=str, default="cnn", 
                        choices=["cnn", "rf", "svm", "gb", "vgg16", "resnet50", "hybrid"],
                        help="Model type to train")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Directory containing the data")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                       help="Number of epochs for deep learning models")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                       help="Batch size for deep learning models")
    parser.add_argument("--save_dir", type=str, default=MODEL_SAVE_PATH,
                       help="Directory to save trained models")
    
    args = parser.parse_args()
    
    # Create figure directory
    os.makedirs("figures", exist_ok=True)
    
    # Train the specified model
    train_model(
        model_type=args.model,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )

if __name__ == "__main__":
    main()

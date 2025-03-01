import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import argparse
import logging
from datetime import datetime
import pandas as pd
import cv2
from tqdm import tqdm

from enhanced_data_generator import EnhancedDataGenerator
from enhanced_model import EnhancedParkinsonsDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_parkinsons_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EnhancedParkinsonsTraining")

class EnhancedTrainer:
    """Class to handle the training process for enhanced Parkinson's detection"""
    
    def __init__(self, data_dir="data/enhanced_synthetic_data", save_dir="saved_models/enhanced"):
        """
        Initialize the trainer
        
        Args:
            data_dir: Directory containing the data
            save_dir: Directory to save trained models
        """
        self.data_dir = data_dir
        self.save_dir = save_dir
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize the model
        self.detector = EnhancedParkinsonsDetector(model_dir=save_dir)
    
    def load_data(self):
        """
        Load data from the data directory
        
        Returns:
            Training and testing data
        """
        logger.info(f"Loading data from {self.data_dir}...")
        
        # Initialize data containers
        train_images = []
        train_labels = []
        test_images = []
        test_labels = []
        
        # Define class mapping
        class_mapping = {"healthy": 0, "parkinson": 1}
        
        # Load training data
        for condition in ["healthy", "parkinson"]:
            label = class_mapping[condition]
            
            # Load spiral images
            spiral_dir = os.path.join(self.data_dir, "train", condition, "spiral")
            for filename in tqdm(os.listdir(spiral_dir), desc=f"Loading {condition} spiral training images"):
                if filename.endswith(".png"):
                    img_path = os.path.join(spiral_dir, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = img.astype('float32') / 255.0  # Normalize
                        train_images.append(img)
                        train_labels.append(label)
            
            # Load wave images
            wave_dir = os.path.join(self.data_dir, "train", condition, "wave")
            for filename in tqdm(os.listdir(wave_dir), desc=f"Loading {condition} wave training images"):
                if filename.endswith(".png"):
                    img_path = os.path.join(wave_dir, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = img.astype('float32') / 255.0  # Normalize
                        train_images.append(img)
                        train_labels.append(label)
        
        # Load testing data
        for condition in ["healthy", "parkinson"]:
            label = class_mapping[condition]
            
            # Load spiral images
            spiral_dir = os.path.join(self.data_dir, "test", condition, "spiral")
            for filename in tqdm(os.listdir(spiral_dir), desc=f"Loading {condition} spiral test images"):
                if filename.endswith(".png"):
                    img_path = os.path.join(spiral_dir, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = img.astype('float32') / 255.0  # Normalize
                        test_images.append(img)
                        test_labels.append(label)
            
            # Load wave images
            wave_dir = os.path.join(self.data_dir, "test", condition, "wave")
            for filename in tqdm(os.listdir(wave_dir), desc=f"Loading {condition} wave test images"):
                if filename.endswith(".png"):
                    img_path = os.path.join(wave_dir, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = img.astype('float32') / 255.0  # Normalize
                        test_images.append(img)
                        test_labels.append(label)
        
        # Convert to numpy arrays
        train_images = np.array(train_images)
        train_labels = np.array(train_labels)
        test_images = np.array(test_images)
        test_labels = np.array(test_labels)
        
        # Reshape images to include channel dimension
        train_images = train_images.reshape(-1, 128, 128, 1)
        test_images = test_images.reshape(-1, 128, 128, 1)
        
        logger.info(f"Loaded {len(train_images)} training images and {len(test_images)} test images")
        
        return (train_images, train_labels), (test_images, test_labels)
    
    def train(self, model_type="enhanced_cnn", epochs=100, batch_size=32, augment=True):
        """
        Train the model
        
        Args:
            model_type: Type of model to train
            epochs: Number of training epochs
            batch_size: Batch size for training
            augment: Whether to use data augmentation
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for model type: {model_type}")
        
        # Load data
        (train_images, train_labels), (test_images, test_labels) = self.load_data()
        
        # Create data augmentation if requested
        if augment:
            logger.info("Setting up data augmentation...")
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            datagen.fit(train_images)
            
            # Train with data augmentation
            train_generator = datagen.flow(train_images, train_labels, batch_size=batch_size)
            history = self.detector.train_model(
                train_generator,
                None,  # Labels are already included in the generator
                test_images,
                test_labels,
                model_type=model_type,
                epochs=epochs,
                batch_size=batch_size
            )
        else:
            # Train without data augmentation
            history = self.detector.train_model(
                train_images,
                train_labels,
                test_images,
                test_labels,
                model_type=model_type,
                epochs=epochs,
                batch_size=batch_size
            )
        
        # Plot training history
        self.plot_training_history(history, model_type)
        
        # Evaluate model
        evaluation = self.detector.evaluate_model(test_images, test_labels, model_type)
        
        # Print evaluation metrics
        logger.info(f"Evaluation metrics: {evaluation['metrics']}")
        logger.info(f"Classification report:\n{evaluation['classification_report']}")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(evaluation['confusion_matrix'], model_type)
        
        return history
    
    def plot_training_history(self, history, model_type):
        """
        Plot training history
        
        Args:
            history: Training history
            model_type: Type of model trained
        """
        # Create figure
        plt.figure(figsize=(12, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title(f'{model_type} Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title(f'{model_type} Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"{model_type}_training_history.png"))
        plt.close()
    
    def plot_confusion_matrix(self, cm, model_type):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            model_type: Type of model evaluated
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Healthy', 'Parkinson'],
                    yticklabels=['Healthy', 'Parkinson'])
        plt.title(f'{model_type} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"{model_type}_confusion_matrix.png"))
        plt.close()
    
    def run_full_training(self):
        """
        Run full training process with all model types
        """
        # Generate data if needed
        if not os.path.exists(self.data_dir):
            logger.info("Data directory not found. Generating synthetic data...")
            data_generator = EnhancedDataGenerator(output_dir=self.data_dir)
            data_generator.generate_dataset()
        
        # Train all model types
        model_types = ["enhanced_cnn", "mobilenet", "efficientnet", "ensemble"]
        
        for model_type in model_types:
            logger.info(f"======= Training {model_type} model =======")
            self.train(model_type=model_type)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train enhanced Parkinson's disease detection models")
    parser.add_argument('--data_dir', type=str, default="data/enhanced_synthetic_data",
                        help='Directory containing the data')
    parser.add_argument('--save_dir', type=str, default="saved_models/enhanced",
                        help='Directory to save trained models')
    parser.add_argument('--model_type', type=str, default="enhanced_cnn", 
                        choices=["enhanced_cnn", "mobilenet", "efficientnet", "ensemble", "all"],
                        help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--no_augment', action='store_true',
                        help='Disable data augmentation')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Create trainer
    trainer = EnhancedTrainer(data_dir=args.data_dir, save_dir=args.save_dir)
    
    # Run training
    if args.model_type == "all":
        trainer.run_full_training()
    else:
        trainer.train(
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            augment=not args.no_augment
        )

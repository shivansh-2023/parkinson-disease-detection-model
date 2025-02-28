import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
import cv2
from tqdm import tqdm
import pandas as pd
import shutil
from config import *

def visualize_training_history(history, output_path=None, metrics=None):
    """
    Visualize the training history for a model.
    
    Args:
        history: Training history from model.fit()
        output_path: Path to save the plot
        metrics: List of metrics to visualize, defaults to ['loss', 'accuracy']
    """
    if metrics is None:
        metrics = ['loss', 'accuracy']
        
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))
    
    if len(metrics) == 1:
        axes = [axes]
        
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        if metric in history.history:
            ax.plot(history.history[metric], label=f'Training {metric}')
            
            val_metric = f'val_{metric}'
            if val_metric in history.history:
                ax.plot(history.history[val_metric], label=f'Validation {metric}')
                
            ax.set_title(f'{metric.capitalize()} over epochs')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {output_path}")
        
    plt.show()

def plot_roc_curve(y_true, y_pred_proba, title="ROC Curve", output_path=None):
    """
    Plot ROC curve for binary classification.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Probability estimates of the positive class
        title: Title for the plot
        output_path: Path to save the plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve plot saved to {output_path}")
        
    plt.show()
    
    return roc_auc

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", output_path=None):
    """
    Plot confusion matrix for binary classification.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        title: Title for the plot
        output_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.xticks([0.5, 1.5], ['Healthy', 'Parkinson'])
    plt.yticks([0.5, 1.5], ['Healthy', 'Parkinson'])
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {output_path}")
        
    plt.show()

def create_sample_data_structure(base_path="data", num_samples=20):
    """
    Create a sample data structure for testing the pipeline.
    
    Args:
        base_path: Base directory for data
        num_samples: Number of samples to create per class
    """
    # Create directories
    spiral_healthy_dir = os.path.join(base_path, "spiral", "healthy")
    spiral_parkinson_dir = os.path.join(base_path, "spiral", "parkinson")
    wave_healthy_dir = os.path.join(base_path, "spiral", "healthy")
    wave_parkinson_dir = os.path.join(base_path, "spiral", "parkinson")
    
    os.makedirs(spiral_healthy_dir, exist_ok=True)
    os.makedirs(spiral_parkinson_dir, exist_ok=True)
    os.makedirs(wave_healthy_dir, exist_ok=True)
    os.makedirs(wave_parkinson_dir, exist_ok=True)
    
    # Generate sample images
    for i in range(num_samples):
        # Create healthy spiral images
        healthy_spiral = generate_spiral_image(is_healthy=True)
        cv2.imwrite(os.path.join(spiral_healthy_dir, f"healthy_spiral_{i}.png"), healthy_spiral)
        
        # Create parkinson spiral images
        parkinson_spiral = generate_spiral_image(is_healthy=False)
        cv2.imwrite(os.path.join(spiral_parkinson_dir, f"parkinson_spiral_{i}.png"), parkinson_spiral)
        
        # Create healthy wave images
        healthy_wave = generate_wave_image(is_healthy=True)
        cv2.imwrite(os.path.join(wave_healthy_dir, f"healthy_wave_{i}.png"), healthy_wave)
        
        # Create parkinson wave images
        parkinson_wave = generate_wave_image(is_healthy=False)
        cv2.imwrite(os.path.join(wave_parkinson_dir, f"parkinson_wave_{i}.png"), parkinson_wave)
    
    # Generate sample clinical data
    clinical_data = generate_clinical_data(num_samples * 4)
    clinical_data.to_csv(os.path.join(base_path, "parkinson_clinical_data.csv"), index=False)
    
    print(f"Sample data structure created in {base_path}")

def generate_spiral_image(image_size=(128, 128), is_healthy=True):
    """
    Generate a synthetic spiral image for testing.
    
    Args:
        image_size: Size of the output image
        is_healthy: Whether to generate a healthy or Parkinson's spiral
    
    Returns:
        Grayscale image as a numpy array
    """
    # Create blank image
    img = np.zeros(image_size, dtype=np.uint8)
    
    # Center of spiral
    center_x, center_y = image_size[0] // 2, image_size[1] // 2
    
    # Parameters for spiral
    max_radius = min(image_size) // 2 - 10
    turns = 3
    points_per_turn = 100
    thickness = 2
    
    # Generate spiral points
    t = np.linspace(0, turns * 2 * np.pi, turns * points_per_turn)
    
    # Add tremor for Parkinson's
    tremor_amplitude = 0 if is_healthy else 5
    tremor_frequency = 20
    
    for i in range(len(t) - 1):
        # Calculate radius
        r1 = (t[i] / (turns * 2 * np.pi)) * max_radius
        r2 = (t[i + 1] / (turns * 2 * np.pi)) * max_radius
        
        # Add tremor for Parkinson's
        if not is_healthy:
            tremor1 = tremor_amplitude * np.sin(tremor_frequency * t[i])
            tremor2 = tremor_amplitude * np.sin(tremor_frequency * t[i + 1])
            r1 += tremor1
            r2 += tremor2
        
        # Calculate coordinates
        x1 = int(center_x + r1 * np.cos(t[i]))
        y1 = int(center_y + r1 * np.sin(t[i]))
        x2 = int(center_x + r2 * np.cos(t[i + 1]))
        y2 = int(center_y + r2 * np.sin(t[i + 1]))
        
        # Draw line segment
        cv2.line(img, (x1, y1), (x2, y2), 255, thickness)
    
    return img

def generate_wave_image(image_size=(128, 128), is_healthy=True):
    """
    Generate a synthetic wave image for testing.
    
    Args:
        image_size: Size of the output image
        is_healthy: Whether to generate a healthy or Parkinson's wave
    
    Returns:
        Grayscale image as a numpy array
    """
    # Create blank image
    img = np.zeros(image_size, dtype=np.uint8)
    
    # Parameters for wave
    amplitude = image_size[1] // 8
    frequency = 3
    thickness = 2
    
    # Add tremor for Parkinson's
    tremor_amplitude = 0 if is_healthy else 5
    tremor_frequency = 40
    
    # Generate wave points
    x_values = np.linspace(10, image_size[0] - 10, 200)
    
    for i in range(len(x_values) - 1):
        # Calculate y values (sine wave)
        t1 = x_values[i] / image_size[0] * (2 * np.pi * frequency)
        t2 = x_values[i + 1] / image_size[0] * (2 * np.pi * frequency)
        
        y1 = image_size[1] // 2 + amplitude * np.sin(t1)
        y2 = image_size[1] // 2 + amplitude * np.sin(t2)
        
        # Add tremor for Parkinson's
        if not is_healthy:
            tremor1 = tremor_amplitude * np.sin(tremor_frequency * t1)
            tremor2 = tremor_amplitude * np.sin(tremor_frequency * t2)
            y1 += tremor1
            y2 += tremor2
        
        # Draw line segment
        cv2.line(img, (int(x_values[i]), int(y1)), (int(x_values[i + 1]), int(y2)), 255, thickness)
    
    return img

def generate_clinical_data(num_samples):
    """
    Generate synthetic clinical data for testing.
    
    Args:
        num_samples: Number of samples to generate
    
    Returns:
        Pandas DataFrame with clinical data
    """
    np.random.seed(RANDOM_SEED)
    
    # Generate random data
    data = {
        'patient_id': [f"P{i:03d}" for i in range(num_samples)],
        'diagnosis': np.random.randint(0, 2, size=num_samples),  # 0: Healthy, 1: Parkinson's
        'age': np.random.randint(40, 85, size=num_samples),
        'tremor_frequency': np.zeros(num_samples),
        'drawing_speed': np.zeros(num_samples),
        'pressure_variation': np.zeros(num_samples),
        'spiral_tightness': np.zeros(num_samples)
    }
    
    # Generate more realistic clinical features based on diagnosis
    for i in range(num_samples):
        if data['diagnosis'][i] == 0:  # Healthy
            data['tremor_frequency'][i] = np.random.uniform(0.1, 2.0)
            data['drawing_speed'][i] = np.random.uniform(7.0, 10.0)
            data['pressure_variation'][i] = np.random.uniform(0.5, 2.0)
            data['spiral_tightness'][i] = np.random.uniform(0.8, 1.0)
        else:  # Parkinson's
            data['tremor_frequency'][i] = np.random.uniform(3.0, 8.0)
            data['drawing_speed'][i] = np.random.uniform(3.0, 6.0)
            data['pressure_variation'][i] = np.random.uniform(2.5, 5.0)
            data['spiral_tightness'][i] = np.random.uniform(0.3, 0.7)
    
    return pd.DataFrame(data)

def organize_dataset(source_dir, target_dir, pattern="*.png"):
    """
    Organize dataset files by copying them to a structured directory.
    
    Args:
        source_dir: Source directory containing dataset files
        target_dir: Target directory to create organized structure
        pattern: File pattern to match
    """
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist.")
        return
    
    # Create target directories
    spiral_healthy_dir = os.path.join(target_dir, "spiral", "healthy")
    spiral_parkinson_dir = os.path.join(target_dir, "spiral", "parkinson")
    wave_healthy_dir = os.path.join(target_dir, "wave", "healthy")
    wave_parkinson_dir = os.path.join(target_dir, "wave", "parkinson")
    
    os.makedirs(spiral_healthy_dir, exist_ok=True)
    os.makedirs(spiral_parkinson_dir, exist_ok=True)
    os.makedirs(wave_healthy_dir, exist_ok=True)
    os.makedirs(wave_parkinson_dir, exist_ok=True)
    
    # Find all files in source directory
    import glob
    files = glob.glob(os.path.join(source_dir, "**", pattern), recursive=True)
    
    num_copied = 0
    for file_path in tqdm(files, desc="Organizing files"):
        filename = os.path.basename(file_path)
        
        # Determine destination directory based on filename
        if "healthy" in filename.lower() and "spiral" in filename.lower():
            dest_dir = spiral_healthy_dir
        elif "parkinson" in filename.lower() and "spiral" in filename.lower():
            dest_dir = spiral_parkinson_dir
        elif "healthy" in filename.lower() and "wave" in filename.lower():
            dest_dir = wave_healthy_dir
        elif "parkinson" in filename.lower() and "wave" in filename.lower():
            dest_dir = wave_parkinson_dir
        else:
            # Skip files that don't match our criteria
            continue
        
        # Copy file to destination
        dest_path = os.path.join(dest_dir, filename)
        shutil.copy2(file_path, dest_path)
        num_copied += 1
    
    print(f"Organized {num_copied} files from {source_dir} to {target_dir}")

if __name__ == "__main__":
    # Test utility functions
    print("Creating sample data structure...")
    create_sample_data_structure()
    
    # Test visualization functions
    dummy_history = type('', (), {})()
    dummy_history.history = {
        'loss': [0.8, 0.6, 0.4, 0.3],
        'val_loss': [0.9, 0.7, 0.5, 0.4],
        'accuracy': [0.6, 0.7, 0.8, 0.85],
        'val_accuracy': [0.55, 0.65, 0.75, 0.8]
    }
    
    print("Testing training history visualization...")
    visualize_training_history(dummy_history)

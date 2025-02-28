import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from data_loader import ParkinsonDataLoader
from config import *

class ModelEvaluator:
    def __init__(self, model_dir="saved_models", data_loader=None):
        """Initialize the evaluator with model directory and optionally a data loader"""
        self.model_dir = model_dir
        self.models = {}
        self.results = {}
        
        # Create data loader if not provided
        if data_loader is None:
            self.data_loader = ParkinsonDataLoader()
        else:
            self.data_loader = data_loader
            
        # Load all available models
        self.load_available_models()
        
    def load_available_models(self):
        """Load all available trained models from the model directory"""
        print(f"Loading models from {self.model_dir}...")
        
        # Check for CNN model
        cnn_path = os.path.join(self.model_dir, "cnn_model.h5")
        if os.path.exists(cnn_path):
            print("Loading CNN model...")
            self.models['cnn'] = load_model(cnn_path)
        
        # Check for transfer learning models
        for base_model in ["vgg16", "resnet50"]:
            transfer_path = os.path.join(self.model_dir, f"transfer_{base_model}_model.h5")
            if os.path.exists(transfer_path):
                print(f"Loading Transfer Learning ({base_model}) model...")
                self.models[f'transfer_{base_model}'] = load_model(transfer_path)
        
        # Check for traditional ML models
        for model_name in ["rf", "svm", "gb"]:
            model_path = os.path.join(self.model_dir, f"{model_name}_model.pkl")
            if os.path.exists(model_path):
                print(f"Loading {model_name.upper()} model...")
                self.models[model_name] = joblib.load(model_path)
                
        if not self.models:
            print("No models found in the specified directory.")
        else:
            print(f"Loaded {len(self.models)} models.")
            
    def load_test_data(self, spiral_dataset_path=SPIRAL_DATASET_PATH, 
                      wave_dataset_path=WAVE_DATASET_PATH,
                      clinical_data_path=CLINICAL_DATA_PATH):
        """Load and prepare test data for evaluation"""
        print("Loading test data...")
        
        # Load image datasets (spiral and wave)
        X_spiral, y_spiral = self.data_loader.load_image_dataset(
            spiral_dataset_path, image_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
        
        X_wave, y_wave = self.data_loader.load_image_dataset(
            wave_dataset_path, image_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
        
        # Combine datasets
        X_images = np.concatenate([X_spiral, X_wave], axis=0)
        y_images = np.concatenate([y_spiral, y_wave], axis=0)
        
        # Convert to RGB for transfer learning
        X_images_rgb = np.repeat(X_images, 3, axis=-1)
        
        # Extract HOG features
        X_hog = self.data_loader.extract_hog_features(X_images, params=HOG_PARAMS)
        
        # Load clinical data if available
        if os.path.exists(clinical_data_path):
            clinical_data = self.data_loader.load_clinical_data(clinical_data_path)
            X_clinical = clinical_data[0]
            y_clinical = clinical_data[1]
            
            # Ensure we have corresponding clinical data for each image
            # This is a simplification - in practice, you'd need proper matching between images and clinical data
            if len(X_clinical) < len(X_images):
                print("Warning: Clinical data size is less than image data size. Using only matching samples.")
                X_images = X_images[:len(X_clinical)]
                X_images_rgb = X_images_rgb[:len(X_clinical)]
                X_hog = X_hog[:len(X_clinical)]
                y_images = y_images[:len(X_clinical)]
        else:
            X_clinical = None
            y_clinical = None
            
        # Get train/test split indices
        # In a real scenario, you would use a proper test set that the model hasn't seen during training
        # Here we use data_loader's functionality to get the same split used in training
        _, _, test_indices = self.data_loader.get_train_val_test_indices(
            len(X_images), val_size=0.1, test_size=0.2)
        
        # Extract test data
        X_test_images = X_images[test_indices]
        X_test_images_rgb = X_images_rgb[test_indices]
        X_test_hog = X_hog[test_indices]
        y_test = y_images[test_indices]
        
        if X_clinical is not None:
            X_test_clinical = X_clinical[test_indices]
        else:
            X_test_clinical = None
            
        print(f"Test data loaded: {len(X_test_images)} samples, {np.sum(y_test)} positive, {len(y_test) - np.sum(y_test)} negative")
        
        return {
            'X_test_images': X_test_images,
            'X_test_images_rgb': X_test_images_rgb,
            'X_test_hog': X_test_hog,
            'X_test_clinical': X_test_clinical,
            'y_test': y_test
        }
        
    def predict_with_model(self, model_name, test_data):
        """Make predictions using a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
            
        model = self.models[model_name]
        
        if model_name.startswith('cnn'):
            # CNN model expects grayscale images with a channel dimension
            y_pred_proba = model.predict(test_data['X_test_images'])
        elif model_name.startswith('transfer'):
            # Transfer learning models expect RGB images
            y_pred_proba = model.predict(test_data['X_test_images_rgb'])
        else:
            # Traditional ML models use HOG features
            y_pred_proba = model.predict_proba(test_data['X_test_hog'])[:, 1]
            
        # Convert probabilities to binary predictions
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        if len(y_pred.shape) > 1:
            y_pred = y_pred.flatten()
            y_pred_proba = y_pred_proba.flatten()
            
        return y_pred, y_pred_proba
        
    def evaluate_model(self, model_name, test_data):
        """Evaluate a specific model on test data"""
        print(f"\nEvaluating {model_name} model...")
        
        # Make predictions
        y_pred, y_pred_proba = self.predict_with_model(model_name, test_data)
        y_test = test_data['y_test']
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        specificity = np.sum((y_test == 0) & (y_pred == 0)) / np.sum(y_test == 0)
        f1 = f1_score(y_test, y_pred)
        
        # Calculate ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'auc': auc,
            'roc_curve': {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds
            },
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Print results
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall (Sensitivity): {recall:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print("Confusion Matrix:")
        print(cm)
        
        return self.results[model_name]
        
    def evaluate_ensemble(self, test_data):
        """Evaluate an ensemble of all loaded models"""
        print("\nEvaluating ensemble model...")
        
        if not self.models:
            raise ValueError("No models loaded for ensemble evaluation.")
            
        # Get predictions from all models
        all_preds_proba = []
        for model_name in self.models.keys():
            _, y_pred_proba = self.predict_with_model(model_name, test_data)
            all_preds_proba.append(y_pred_proba)
            
        # Average predictions
        ensemble_pred_proba = np.mean(all_preds_proba, axis=0)
        ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        y_test = test_data['y_test']
        accuracy = accuracy_score(y_test, ensemble_pred)
        precision = precision_score(y_test, ensemble_pred)
        recall = recall_score(y_test, ensemble_pred)
        specificity = np.sum((y_test == 0) & (ensemble_pred == 0)) / np.sum(y_test == 0)
        f1 = f1_score(y_test, ensemble_pred)
        
        # Calculate ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_test, ensemble_pred_proba)
        auc = roc_auc_score(y_test, ensemble_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, ensemble_pred)
        
        # Store results
        self.results['ensemble'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'auc': auc,
            'roc_curve': {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds
            },
            'confusion_matrix': cm,
            'y_pred': ensemble_pred,
            'y_pred_proba': ensemble_pred_proba
        }
        
        # Print results
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall (Sensitivity): {recall:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print("Confusion Matrix:")
        print(cm)
        
        return self.results['ensemble']
        
    def evaluate_all(self):
        """Evaluate all models and an ensemble"""
        # Load test data
        test_data = self.load_test_data()
        
        # Evaluate each model
        for model_name in self.models.keys():
            self.evaluate_model(model_name, test_data)
            
        # Evaluate ensemble
        self.evaluate_ensemble(test_data)
        
        return self.results
        
    def plot_roc_curves(self, save_path=None):
        """Plot ROC curves for all evaluated models"""
        if not self.results:
            raise ValueError("No evaluation results found. Run evaluate_all() first.")
            
        plt.figure(figsize=(10, 8))
        
        for model_name, result in self.results.items():
            if 'roc_curve' in result:
                fpr = result['roc_curve']['fpr']
                tpr = result['roc_curve']['tpr']
                auc = result['auc']
                
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
                
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Parkinson\'s Disease Detection Models')
        plt.legend(loc='lower right')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to {save_path}")
            
        plt.show()
        
    def plot_confusion_matrices(self, save_path=None):
        """Plot confusion matrices for all evaluated models"""
        if not self.results:
            raise ValueError("No evaluation results found. Run evaluate_all() first.")
            
        num_models = len(self.results)
        cols = min(3, num_models)
        rows = (num_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
            
        for i, (model_name, result) in enumerate(self.results.items()):
            if 'confusion_matrix' in result:
                cm = result['confusion_matrix']
                
                row = i // cols
                col = i % cols
                
                if rows > 1 and cols > 1:
                    ax = axes[row, col]
                else:
                    ax = axes[i]
                    
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('True Label')
                ax.set_title(f'{model_name} Confusion Matrix')
                ax.set_xticklabels(['Healthy', 'Parkinson'])
                ax.set_yticklabels(['Healthy', 'Parkinson'])
                
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrices saved to {save_path}")
            
        plt.show()
        
    def save_results_to_csv(self, output_path="evaluation_results.csv"):
        """Save evaluation metrics to a CSV file"""
        if not self.results:
            raise ValueError("No evaluation results found. Run evaluate_all() first.")
            
        # Extract metrics for each model
        metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'auc']
        results_df = pd.DataFrame(columns=['model'] + metrics)
        
        for model_name, result in self.results.items():
            row = {'model': model_name}
            for metric in metrics:
                if metric in result:
                    row[metric] = result[metric]
            
            results_df = results_df.append(row, ignore_index=True)
            
        # Save to CSV
        results_df.to_csv(output_path, index=False)
        print(f"Evaluation results saved to {output_path}")
        
        return results_df

def main(args=None):
    """Main function to be called from other modules or CLI"""
    if args is None:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Evaluate Parkinson's Disease Detection Models")
        parser.add_argument("--model_dir", type=str, default=MODEL_SAVE_PATH,
                            help="Directory containing trained models")
        parser.add_argument("--data_path", type=str, default=DATA_PATH,
                            help="Path to data directory")
        parser.add_argument("--plot", action="store_true",
                            help="Generate and display plots")
        parser.add_argument("--save_results", action="store_true",
                            help="Save evaluation results to CSV")
        parser.add_argument("--output_dir", type=str, default="evaluation_results",
                            help="Directory to save evaluation results and plots")
        
        args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize data loader and model
    data_loader = ParkinsonDataLoader()
    model_evaluator = ModelEvaluator(model_dir=args.model_dir, data_loader=data_loader)
    
    # Load image datasets
    X_spiral, y_spiral = data_loader.load_image_dataset(
        os.path.join(args.data_path, "spiral"), 
        image_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
    
    X_wave, y_wave = data_loader.load_image_dataset(
        os.path.join(args.data_path, "wave"), 
        image_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
    
    # Combine datasets
    X_images = np.concatenate([X_spiral, X_wave], axis=0)
    y = np.concatenate([y_spiral, y_wave], axis=0)
    
    # Get train/val/test split
    _, _, test_indices = data_loader.get_train_val_test_indices(
        len(X_images), val_size=0.1, test_size=0.2)
    
    X_test = X_images[test_indices]
    y_test = y[test_indices]
    
    print(f"Test data: {X_test.shape[0]} samples")
    
    # Prepare RGB images for transfer learning models
    X_test_rgb = np.repeat(X_test, 3, axis=-1)
    
    # Extract HOG features for traditional ML models
    X_test_hog = data_loader.extract_hog_features(X_test)
    
    # Find all model files in the model directory
    model_files = []
    for root, dirs, files in os.walk(args.model_dir):
        for file in files:
            if file.endswith(".h5") or file.endswith(".pkl"):
                model_files.append(os.path.join(root, file))
    
    if not model_files:
        print(f"No model files found in {args.model_dir}")
        return
    
    # Evaluate each model
    all_results = []
    
    for model_file in model_files:
        model_name = os.path.basename(model_file).replace("_model.h5", "").replace("_model.pkl", "")
        print(f"\nEvaluating {model_name} model...")
        
        # Load model
        if model_file.endswith(".h5"):
            tf_model = tf.keras.models.load_model(model_file)
            if "transfer" in model_name:
                test_data = X_test_rgb
            else:
                test_data = X_test
            metrics = model_evaluator.evaluate_model(model_name, tf_model, test_data, y_test, 
                                                   save_results=args.save_results, 
                                                   output_dir=args.output_dir)
        else:  # .pkl file
            ml_model = joblib.load(model_file)
            metrics = model_evaluator.evaluate_traditional_ml_model(model_name, ml_model, X_test_hog, y_test,
                                                                  save_results=args.save_results,
                                                                  output_dir=args.output_dir)
        
        # Add model name and metrics to results
        result = {"Model": model_name}
        result.update(metrics)
        all_results.append(result)
        
        # Plot confusion matrix and ROC curve
        if args.plot:
            if model_file.endswith(".h5"):
                y_pred = tf_model.predict(test_data)
                if y_pred.shape[1] > 1:  # One-hot encoded predictions
                    y_pred = np.argmax(y_pred, axis=1)
                else:
                    y_pred = (y_pred > 0.5).astype(int).flatten()
            else:
                if hasattr(ml_model, "predict_proba"):
                    y_pred_proba = ml_model.predict_proba(X_test_hog)[:, 1]
                    y_pred = (y_pred_proba > 0.5).astype(int)
                else:
                    y_pred = ml_model.predict(X_test_hog)
            
            # Plot confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                      xticklabels=['Healthy', 'Parkinsons'], 
                      yticklabels=['Healthy', 'Parkinsons'])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f"{model_name}_confusion_matrix.png"))
            plt.close()
            
            # Plot ROC curve if probability predictions are available
            if hasattr(ml_model, "predict_proba") or model_file.endswith(".h5"):
                if model_file.endswith(".h5"):
                    y_pred_proba = tf_model.predict(test_data)
                    if y_pred_proba.shape[1] > 1:  # One-hot encoded predictions
                        y_pred_proba = y_pred_proba[:, 1]
                    else:
                        y_pred_proba = y_pred_proba.flatten()
                else:
                    y_pred_proba = ml_model.predict_proba(X_test_hog)[:, 1]
                
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - {model_name}')
                plt.legend(loc="lower right")
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, f"{model_name}_roc_curve.png"))
                plt.close()
    
    # Save all results to CSV
    if args.save_results and all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(args.output_dir, "model_comparison.csv"), index=False)
        print(f"\nResults saved to {os.path.join(args.output_dir, 'model_comparison.csv')}")
        
        # Create a comparison bar chart
        plt.figure(figsize=(12, 6))
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        for i, metric in enumerate(metrics_to_plot):
            if metric in results_df.columns:
                plt.subplot(2, 2, i+1)
                sns.barplot(x='Model', y=metric, data=results_df)
                plt.title(f'Model Comparison - {metric.capitalize()}')
                plt.xticks(rotation=45)
                plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "model_comparison.png"))
        plt.close()
    
    # Print summary of results
    print("\n=== Evaluation Summary ===")
    for result in all_results:
        print(f"{result['Model']} - Accuracy: {result['accuracy']:.4f}, F1 Score: {result['f1_score']:.4f}")
    
    if args.save_results:
        print(f"\nDetailed results saved to {args.output_dir}")
    
    return all_results

if __name__ == "__main__":
    main()

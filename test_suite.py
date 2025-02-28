import os
import unittest
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model import ParkinsonsDetector
from data_loader import ParkinsonDataLoader
from config import *

class TestParkinsonsDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data and model instances once for all tests"""
        # Create data directories if they don't exist
        os.makedirs(SPIRAL_DATASET_PATH, exist_ok=True)
        os.makedirs(WAVE_DATASET_PATH, exist_ok=True)
        os.makedirs(os.path.join(SPIRAL_DATASET_PATH, 'healthy'), exist_ok=True)
        os.makedirs(os.path.join(SPIRAL_DATASET_PATH, 'parkinson'), exist_ok=True)
        os.makedirs(os.path.join(WAVE_DATASET_PATH, 'healthy'), exist_ok=True)
        os.makedirs(os.path.join(WAVE_DATASET_PATH, 'parkinson'), exist_ok=True)
        
        # Initialize data loader
        cls.data_loader = ParkinsonDataLoader()
        
        # Generate mock data for testing
        cls.mock_data = cls.generate_mock_data()
        
        # Initialize model
        cls.model = ParkinsonsDetector(image_shape=IMAGE_SHAPE)
        
    @staticmethod
    def generate_mock_data(num_samples=100):
        """Generate mock data for testing"""
        # Generate mock images (random noise)
        X_images = np.random.rand(num_samples, IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2])
        
        # Generate random labels (0 or 1)
        y = np.random.randint(0, 2, size=num_samples)
        
        # Generate mock clinical data
        X_clinical = np.random.rand(num_samples, len(CLINICAL_FEATURES))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_images, y, test_size=0.2, random_state=RANDOM_SEED)
        X_clinical_train, X_clinical_test, _, _ = train_test_split(
            X_clinical, y, test_size=0.2, random_state=RANDOM_SEED)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_clinical_train': X_clinical_train,
            'X_clinical_test': X_clinical_test
        }
        
    def test_data_loader_shape(self):
        """Test the shape of data from the data loader"""
        # Extract HOG features
        X_hog = self.data_loader.extract_hog_features(self.mock_data['X_train'], params=HOG_PARAMS)
        
        # Check shapes
        self.assertEqual(self.mock_data['X_train'].shape[0], X_hog.shape[0])
        self.assertGreater(X_hog.shape[1], 0)  # HOG features should have at least 1 feature
        
    def test_data_augmentation(self):
        """Test data augmentation"""
        # Get a small subset of data
        X_subset = self.mock_data['X_train'][:5]
        y_subset = self.mock_data['y_train'][:5]
        
        # Apply augmentation
        X_aug, y_aug = self.data_loader.augment_data(X_subset, y_subset, augment_factor=2)
        
        # Check shapes
        self.assertEqual(X_aug.shape[0], X_subset.shape[0] * 2)
        self.assertEqual(y_aug.shape[0], y_subset.shape[0] * 2)
        
    def test_cnn_model_build(self):
        """Test building CNN model"""
        # Build CNN model
        cnn_model = self.model.build_cnn_model()
        
        # Check model type
        self.assertIsInstance(cnn_model, tf.keras.Model)
        
        # Check input shape
        self.assertEqual(cnn_model.input_shape[1:], IMAGE_SHAPE)
        
        # Check output shape
        self.assertEqual(cnn_model.output_shape[1:], (1,))
        
    def test_transfer_learning_model_build(self):
        """Test building transfer learning model"""
        # Build transfer learning model
        transfer_model = self.model.build_transfer_learning_model('vgg16')
        
        # Check model type
        self.assertIsInstance(transfer_model, tf.keras.Model)
        
        # Check input shape (transfer learning models expect RGB input)
        self.assertEqual(transfer_model.input_shape[1:3], (IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
        self.assertEqual(transfer_model.input_shape[3], 3)  # RGB channels
        
        # Check output shape
        self.assertEqual(transfer_model.output_shape[1:], (1,))
        
    def test_traditional_ml_models(self):
        """Test traditional ML models"""
        # Extract HOG features
        X_hog_train = self.data_loader.extract_hog_features(
            self.mock_data['X_train'], params=HOG_PARAMS)
        X_hog_test = self.data_loader.extract_hog_features(
            self.mock_data['X_test'], params=HOG_PARAMS)
        
        for model_type in ['rf', 'svm', 'gb']:
            # Train model
            ml_model = self.model.train_traditional_ml_model(
                X_hog_train, self.mock_data['y_train'], model_type=model_type)
            
            # Make predictions
            y_pred = ml_model.predict(X_hog_test)
            
            # Check prediction shape
            self.assertEqual(len(y_pred), len(self.mock_data['y_test']))
            
            # Check prediction values
            self.assertTrue(np.all((y_pred == 0) | (y_pred == 1)))
            
    def test_hybrid_model(self):
        """Test hybrid model combining CNN features with clinical data"""
        # Build hybrid model
        hybrid_model = self.model.build_hybrid_model(
            clinical_features_dim=len(CLINICAL_FEATURES))
        
        # Check model type
        self.assertIsInstance(hybrid_model, tf.keras.Model)
        
        # Check that model has multiple inputs
        self.assertEqual(len(hybrid_model.inputs), 2)
        
        # Check input shapes
        self.assertEqual(hybrid_model.inputs[0].shape[1:], IMAGE_SHAPE)
        self.assertEqual(hybrid_model.inputs[1].shape[1:], (len(CLINICAL_FEATURES),))
        
        # Check output shape
        self.assertEqual(hybrid_model.outputs[0].shape[1:], (1,))
        
    def test_model_evaluation(self):
        """Test model evaluation"""
        # Build a simple CNN model
        cnn_model = self.model.build_cnn_model()
        
        # Compile model
        cnn_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=CNN_PARAMS['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model for 1 epoch (just for testing)
        cnn_model.fit(
            self.mock_data['X_train'], self.mock_data['y_train'],
            epochs=1, batch_size=BATCH_SIZE, verbose=0
        )
        
        # Evaluate model
        eval_metrics = self.model.evaluate_model(
            'cnn', cnn_model, self.mock_data['X_test'], self.mock_data['y_test'])
        
        # Check that evaluation metrics exist
        self.assertIn('accuracy', eval_metrics)
        self.assertIn('precision', eval_metrics)
        self.assertIn('recall', eval_metrics)
        self.assertIn('f1_score', eval_metrics)
        self.assertIn('auc', eval_metrics)
        
        # Check that metrics are within valid ranges
        for metric_name, metric_value in eval_metrics.items():
            if metric_name != 'confusion_matrix':
                self.assertGreaterEqual(metric_value, 0.0)
                self.assertLessEqual(metric_value, 1.0)

if __name__ == '__main__':
    unittest.main()

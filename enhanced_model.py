import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Dense, 
    BatchNormalization, Activation, Flatten, Concatenate, SpatialDropout2D,
    Add, DepthwiseConv2D, Reshape, GaussianNoise
)
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2, EfficientNetB0
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

class EnhancedParkinsonsDetector:
    """Enhanced model for Parkinson's disease detection from spiral/wave drawings"""
    
    def __init__(self, image_shape=(128, 128, 1), num_classes=2, model_dir="saved_models"):
        """
        Initialize the model
        
        Args:
            image_shape: Input image shape (height, width, channels)
            num_classes: Number of output classes
            model_dir: Directory to save trained models
        """
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.model_dir = model_dir
        self.models = {}
    
    def build_enhanced_cnn(self):
        """
        Build an enhanced CNN model with residual connections and regularization
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = Input(shape=self.image_shape)
        
        # Add noise for robustness during training
        x = GaussianNoise(0.1)(inputs)
        
        # First block
        x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SpatialDropout2D(0.1)(x)
        x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2))(x)
        
        # Second block with residual connection
        shortcut = Conv2D(64, (1, 1), strides=(2, 2), padding='same')(x)
        
        x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SpatialDropout2D(0.1)(x)
        x = Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        
        # Add residual connection
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        
        # Third block
        x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SpatialDropout2D(0.2)(x)
        x = MaxPooling2D((2, 2))(x)
        
        # Fourth block
        x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SpatialDropout2D(0.2)(x)
        x = MaxPooling2D((2, 2))(x)
        
        # Global pooling and dense layers
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        x = Dense(512, kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.4)(x)
        
        # Output layer
        outputs = Dense(1, activation='sigmoid')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name="enhanced_cnn")
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def build_mobilenet_model(self):
        """
        Build a MobileNetV2-based model (lightweight but powerful)
        
        Returns:
            Compiled Keras model
        """
        # Create base model from MobileNetV2
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(self.image_shape[0], self.image_shape[1], 3)
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Create new model on top
        inputs = Input(shape=self.image_shape)
        
        # Convert grayscale to RGB (if needed)
        if self.image_shape[2] == 1:
            x = Conv2D(3, (1, 1), padding='same')(inputs)
        else:
            x = inputs
        
        # Pass through the base model
        x = base_model(x)
        
        # Add custom layers
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
        x = Dropout(0.4)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        # Assemble the model
        model = Model(inputs=inputs, outputs=outputs, name="mobilenet_parkinsons")
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def build_efficient_net_model(self):
        """
        Build an EfficientNet-based model
        
        Returns:
            Compiled Keras model
        """
        # Create base model from EfficientNetB0
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(self.image_shape[0], self.image_shape[1], 3)
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Create new model on top
        inputs = Input(shape=self.image_shape)
        
        # Convert grayscale to RGB (if needed)
        if self.image_shape[2] == 1:
            x = Conv2D(3, (1, 1), padding='same')(inputs)
        else:
            x = inputs
        
        # Pass through the base model
        x = base_model(x)
        
        # Add custom layers
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
        x = Dropout(0.4)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        # Assemble the model
        model = Model(inputs=inputs, outputs=outputs, name="efficientnet_parkinsons")
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def build_ensemble_model(self):
        """
        Build a model that combines multiple CNNs in an ensemble
        
        Returns:
            Compiled Keras model
        """
        # Create input
        inputs = Input(shape=self.image_shape)
        
        # Branch 1: Simple CNN path
        branch1 = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
        branch1 = MaxPooling2D((2, 2))(branch1)
        branch1 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch1)
        branch1 = MaxPooling2D((2, 2))(branch1)
        branch1 = Conv2D(128, (3, 3), padding='same', activation='relu')(branch1)
        branch1 = GlobalAveragePooling2D()(branch1)
        branch1 = Dense(128, activation='relu')(branch1)
        
        # Branch 2: Deep CNN path
        branch2 = Conv2D(32, (7, 7), padding='same', activation='relu')(inputs)
        branch2 = MaxPooling2D((2, 2))(branch2)
        branch2 = Conv2D(64, (5, 5), padding='same', activation='relu')(branch2)
        branch2 = MaxPooling2D((2, 2))(branch2)
        branch2 = Conv2D(128, (3, 3), padding='same', activation='relu')(branch2)
        branch2 = GlobalAveragePooling2D()(branch2)
        branch2 = Dense(128, activation='relu')(branch2)
        
        # Branch 3: Residual path
        branch3 = Conv2D(32, (3, 3), padding='same')(inputs)
        branch3 = BatchNormalization()(branch3)
        branch3 = Activation('relu')(branch3)
        
        shortcut = Conv2D(64, (1, 1), strides=(2, 2), padding='same')(branch3)
        
        branch3 = Conv2D(64, (3, 3), padding='same')(branch3)
        branch3 = BatchNormalization()(branch3)
        branch3 = Activation('relu')(branch3)
        branch3 = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(branch3)
        branch3 = BatchNormalization()(branch3)
        branch3 = Add()([branch3, shortcut])
        branch3 = Activation('relu')(branch3)
        branch3 = GlobalAveragePooling2D()(branch3)
        branch3 = Dense(128, activation='relu')(branch3)
        
        # Combine branches
        combined = Concatenate()([branch1, branch2, branch3])
        
        # Final dense layers
        x = Dense(256, activation='relu')(combined)
        x = Dropout(0.4)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs, name="ensemble_cnn")
        
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def train_model(self, train_data, train_labels, test_data=None, test_labels=None, 
                   model_type="enhanced_cnn", epochs=100, batch_size=32):
        """
        Train the specified model
        
        Args:
            train_data: Training data or data generator
            train_labels: Training labels (None if using a generator)
            test_data: Test data (optional)
            test_labels: Test labels (optional)
            model_type: Type of model to train
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        # Build the appropriate model
        if model_type == "enhanced_cnn":
            model = self.build_enhanced_cnn()
        elif model_type == "mobilenet":
            model = self.build_mobilenet_model()
        elif model_type == "efficientnet":
            model = self.build_efficient_net_model()
        elif model_type == "ensemble":
            model = self.build_ensemble_model()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Prepare callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=f"{self.model_dir}/{model_type}_best.h5",
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Check if we're using a generator or normal data
        is_generator = hasattr(train_data, 'next') or hasattr(train_data, '__next__') or hasattr(train_data, 'flow')
        
        # Train the model
        if test_data is not None and test_labels is not None:
            if is_generator:
                # Using a generator
                history = model.fit(
                    train_data,
                    validation_data=(test_data, test_labels),
                    epochs=epochs,
                    steps_per_epoch=len(train_data) if hasattr(train_data, '__len__') else None,
                    callbacks=callbacks
                )
            else:
                # Using regular numpy arrays
                history = model.fit(
                    train_data, train_labels,
                    validation_data=(test_data, test_labels),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks
                )
        else:
            # Use validation split if no test data
            if is_generator:
                history = model.fit(
                    train_data,
                    epochs=epochs,
                    steps_per_epoch=len(train_data) if hasattr(train_data, '__len__') else None,
                    validation_split=0.2,
                    callbacks=callbacks
                )
            else:
                history = model.fit(
                    train_data, train_labels,
                    validation_split=0.2,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks
                )
        
        # Save the trained model
        model.save(f"{self.model_dir}/{model_type}_final.h5")
        
        # Store the model in the instance
        self.models[model_type] = model
        
        return history
    
    def evaluate_model(self, test_data, test_labels, model_type="enhanced_cnn"):
        """
        Evaluate the specified model
        
        Args:
            test_data: Test data
            test_labels: Test labels
            model_type: Type of model to evaluate
            
        Returns:
            Evaluation metrics
        """
        # Make sure the model exists
        if model_type not in self.models:
            raise ValueError(f"Model '{model_type}' not found. Please train it first.")
        
        # Get the model
        model = self.models[model_type]
        
        # Evaluate the model
        metrics = model.evaluate(test_data, test_labels)
        
        # Generate detailed metrics
        y_pred_prob = model.predict(test_data)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Calculate classification report and confusion matrix
        report = classification_report(test_labels, y_pred)
        cm = confusion_matrix(test_labels, y_pred)
        
        # Return all metrics
        return {
            'metrics': dict(zip(model.metrics_names, metrics)),
            'classification_report': report,
            'confusion_matrix': cm
        }

if __name__ == "__main__":
    # Example usage
    detector = EnhancedParkinsonsDetector()
    print("Enhanced Parkinson's Detector model created. Use train_model() to train the model.")

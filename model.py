import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import VGG16, ResNet50
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from tensorflow.keras import regularizers
from skimage import feature
import numpy as np

class ParkinsonsDetector:
    def __init__(self, image_shape=(128, 128, 1)):
        self.image_shape = image_shape
        self.models = {}
        
    def extract_hog_features(self, image):
        """Extract Histogram of Oriented Gradients features"""
        features = feature.hog(image, orientations=9,
                              pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                              transform_sqrt=True, block_norm="L1")
        return features
    
    def build_cnn_model(self):
        """Build a CNN model for spiral/wave image classification"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=self.image_shape, padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.3),
            
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.4),
            
            Flatten(),
            Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc')]
        )
        
        self.models['cnn'] = model
        return model
    
    def build_transfer_model(self, base_model='vgg16'):
        """Build a transfer learning model with VGG16 or ResNet50"""
        if base_model == 'vgg16':
            base = VGG16(weights='imagenet', include_top=False, 
                        input_shape=(self.image_shape[0], self.image_shape[1], 3))
        elif base_model == 'resnet50':
            base = ResNet50(weights='imagenet', include_top=False, 
                           input_shape=(self.image_shape[0], self.image_shape[1], 3))
        else:
            raise ValueError("base_model must be 'vgg16' or 'resnet50'")
            
        # Freeze base model layers
        for layer in base.layers:
            layer.trainable = False
            
        x = Flatten()(base.output)
        x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        output = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=base.input, outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc')]
        )
        
        self.models['transfer'] = model
        return model
    
    def build_ensemble_model(self):
        """Build Random Forest and SVM models for HOG features"""
        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=200, 
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Support Vector Machine
        svm = SVC(
            C=10.0,
            kernel='rbf',
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=42
        )
        
        # Gradient Boosting
        gb = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.models['rf'] = rf
        self.models['svm'] = svm
        self.models['gb'] = gb
        
        return {'rf': rf, 'svm': svm, 'gb': gb}
    
    def build_hybrid_model(self):
        """Build a hybrid model combining CNN features with traditional ML models"""
        img_input = Input(shape=self.image_shape)
        
        # CNN Feature Extractor
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        
        x = Flatten()(x)
        cnn_features = Dense(128, activation='relu')(x)
        
        # Clinical Features Input
        clinical_input = Input(shape=(5,))  # Assuming 5 clinical features
        
        # Combine Features
        combined = concatenate([cnn_features, clinical_input])
        
        x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        output = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=[img_input, clinical_input], outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                   tf.keras.metrics.Precision(name='precision'),
                   tf.keras.metrics.Recall(name='recall'),
                   tf.keras.metrics.AUC(name='auc')]
        )
        
        self.models['hybrid'] = model
        return model
    
    def get_callbacks(self):
        """Get standard callbacks for model training"""
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        return [early_stopping, reduce_lr]
    
    def train_model(self, model_name, X_train, y_train, X_val=None, y_val=None, 
                   batch_size=32, epochs=50, callbacks=None):
        """Train a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Build it first.")
            
        model = self.models[model_name]
        
        if model_name in ['rf', 'svm', 'gb']:
            # Traditional ML models
            model.fit(X_train, y_train)
            return None
        else:
            # Deep learning models
            validation_data = None
            if X_val is not None and y_val is not None:
                if model_name == 'hybrid':
                    validation_data = ([X_val[0], X_val[1]], y_val)
                else:
                    validation_data = (X_val, y_val)
                    
            if callbacks is None:
                callbacks = self.get_callbacks()
                
            if model_name == 'hybrid':
                history = model.fit(
                    [X_train[0], X_train[1]], y_train,
                    validation_data=validation_data,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=callbacks,
                    class_weight={0: 1, 1: 2.5}  # Addressing class imbalance
                )
            else:
                history = model.fit(
                    X_train, y_train,
                    validation_data=validation_data,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=callbacks,
                    class_weight={0: 1, 1: 2.5}  # Addressing class imbalance
                )
                
            return history
    
    def evaluate_model(self, model_name, X_test, y_test):
        """Evaluate model performance"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")
            
        model = self.models[model_name]
        
        if model_name in ['rf', 'svm', 'gb']:
            # Traditional ML models
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
            # Calculate sensitivity and specificity from confusion matrix
            tn, fp, fn, tp = conf_matrix.ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            
            return {
                'accuracy': accuracy,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'confusion_matrix': conf_matrix,
                'classification_report': report
            }
        else:
            # Deep learning models
            if model_name == 'hybrid':
                loss, accuracy, precision, recall, auc = model.evaluate(
                    [X_test[0], X_test[1]], y_test, verbose=0
                )
                y_pred = model.predict([X_test[0], X_test[1]]) > 0.5
            else:
                loss, accuracy, precision, recall, auc = model.evaluate(
                    X_test, y_test, verbose=0
                )
                y_pred = model.predict(X_test) > 0.5
                
            conf_matrix = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
            # Calculate sensitivity and specificity from confusion matrix
            tn, fp, fn, tp = conf_matrix.ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            
            return {
                'loss': loss,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'auc': auc,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'confusion_matrix': conf_matrix,
                'classification_report': report
            }

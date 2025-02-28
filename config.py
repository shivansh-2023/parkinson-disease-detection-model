# Path configuration
DATA_PATH = 'data/'
SPIRAL_DATASET_PATH = 'data/spiral/'
WAVE_DATASET_PATH = 'data/wave/'
CLINICAL_DATA_PATH = 'data/parkinson_clinical_data.csv'
MODEL_SAVE_PATH = 'saved_models/'

# Model parameters
IMAGE_SHAPE = (128, 128, 1)
RGB_IMAGE_SHAPE = (128, 128, 3)  # For transfer learning models

# Clinical features (column names in clinical data)
CLINICAL_FEATURES = [
    'age', 
    'tremor_frequency', 
    'drawing_speed', 
    'pressure_variation', 
    'spiral_tightness'
]

# Training parameters
EPOCHS = 50
BATCH_SIZE = 32
CLASS_WEIGHTS = {0: 1, 1: 2.5}  # For handling class imbalance
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42  # For reproducibility

# Hyperparameters for models
CNN_PARAMS = {
    'learning_rate': 0.0001,
    'dropout_rate': 0.5,
    'l2_regularization': 0.01
}

RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'class_weight': 'balanced'
}

SVM_PARAMS = {
    'C': 10.0,
    'kernel': 'rbf',
    'gamma': 'scale',
    'probability': True,
    'class_weight': 'balanced'
}

GB_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 5,
    'min_samples_split': 5,
    'min_samples_leaf': 2
}

# HOG feature extraction parameters
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'transform_sqrt': True,
    'block_norm': 'L1'
}

# Transfer learning parameters
TRANSFER_LEARNING_PARAMS = {
    'base_model': 'vgg16',  # 'vgg16' or 'resnet50'
    'learning_rate': 0.0001,
    'dropout_rate': 0.5,
    'l2_regularization': 0.01,
    'trainable_layers': 0  # Number of top layers to make trainable
}

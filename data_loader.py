import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from skimage import feature
from glob import glob
from tqdm import tqdm

class ParkinsonDataLoader:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.data = None
        self.image_data = None
        self.clinical_data = None
        
    def load_image_dataset(self, dataset_path, image_size=(128, 128)):
        """
        Load spiral/wave images from directory structure
        Expected structure:
        dataset_path/
            spiral/
                training/
                    healthy/
                    parkinson/
                testing/
                    healthy/
                    parkinson/
            wave/
                training/
                    healthy/
                    parkinson/
                testing/
                    healthy/
                    parkinson/
        """
        print("[INFO] Loading image dataset...")
        
        patterns = ['spiral', 'wave']
        splits = ['training', 'testing']
        classes = ['healthy', 'parkinson']
        
        # Initialize data dictionaries
        train_data = {'images': [], 'labels': [], 'pattern': []}
        test_data = {'images': [], 'labels': [], 'pattern': []}
        
        for pattern in patterns:
            print(f"[INFO] Processing {pattern} images...")
            
            for split in splits:
                data_dict = train_data if split == 'training' else test_data
                
                for cls_idx, cls in enumerate(classes):
                    path = os.path.join(dataset_path, pattern, split, cls)
                    if not os.path.exists(path):
                        print(f"[WARNING] Path does not exist: {path}")
                        continue
                        
                    image_paths = glob(os.path.join(path, "*.png"))
                    image_paths.extend(glob(os.path.join(path, "*.jpg")))
                    
                    print(f"[INFO] Loading {len(image_paths)} {cls} images from {path}")
                    
                    for img_path in tqdm(image_paths):
                        # Load image
                        image = cv2.imread(img_path)
                        if image is None:
                            print(f"[WARNING] Could not load image: {img_path}")
                            continue
                            
                        # Convert to grayscale
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        
                        # Resize
                        image = cv2.resize(image, image_size)
                        
                        # Threshold to make drawing appear as white on black
                        image = cv2.threshold(image, 0, 255,
                                             cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                                             
                        # Normalize pixel values to range [0, 1]
                        image = image.astype("float32") / 255.0
                        
                        # Reshape for CNN (add channel dimension)
                        image = np.expand_dims(image, axis=-1)
                        
                        # Add to dataset
                        data_dict['images'].append(image)
                        data_dict['labels'].append(cls_idx)  # 0 for healthy, 1 for parkinson
                        data_dict['pattern'].append(pattern)
                    
        # Convert to numpy arrays
        train_images = np.array(train_data['images'])
        train_labels = np.array(train_data['labels'])
        train_patterns = np.array(train_data['pattern'])
        
        test_images = np.array(test_data['images'])
        test_labels = np.array(test_data['labels'])
        test_patterns = np.array(test_data['pattern'])
        
        print(f"[INFO] Dataset loaded: {len(train_images)} training, {len(test_images)} testing images")
        
        return (train_images, train_labels, train_patterns), (test_images, test_labels, test_patterns)
    
    def load_hog_features(self, images):
        """Extract HOG features from images"""
        print("[INFO] Extracting HOG features...")
        features = []
        
        for image in tqdm(images):
            # Convert to single channel if needed
            if image.ndim > 2 and image.shape[2] == 1:
                img = image[:, :, 0]
            else:
                img = image
                
            # Extract HOG features
            hog_features = feature.hog(img, orientations=9,
                                      pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                      transform_sqrt=True, block_norm="L1")
            features.append(hog_features)
            
        return np.array(features)
    
    def load_clinical_data(self, file_path):
        """Load clinical data from CSV file"""
        print(f"[INFO] Loading clinical data from {file_path}")
        self.clinical_data = pd.read_csv(file_path)
        self.clinical_data = self.clinical_data.dropna()
        
        return self.clinical_data
    
    def clean_data(self):
        """Clean and preprocess clinical data"""
        if self.clinical_data is None:
            raise ValueError("Clinical data not loaded. Call load_clinical_data first.")
            
        print("[INFO] Cleaning clinical data...")
        
        # Apply data cleaning rules based on clinical knowledge
        self.clinical_data = self.clinical_data[(self.clinical_data['tremor_frequency'] < 15) & 
                                              (self.clinical_data['pressure_variation'] > 0.1)]
        
        return self.clinical_data
    
    def preprocess_clinical(self, test_size=0.2, random_state=42):
        """Preprocess clinical data for model training"""
        if self.clinical_data is None:
            raise ValueError("Clinical data not loaded. Call load_clinical_data first.")
            
        print("[INFO] Preprocessing clinical data...")
        
        # Split features and target
        X = self.clinical_data.drop(['patient_id', 'diagnosis'], axis=1)
        y = self.clinical_data['diagnosis'].map({'healthy': 0, 'parkinson': 1})
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        
        # Apply SMOTE to handle class imbalance
        smote = SMOTE(random_state=random_state)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
        
        return (X_train_bal, X_test, y_train_bal, y_test)
    
    def prepare_hybrid_data(self, image_data, clinical_data, clinical_feature_columns):
        """Prepare data for hybrid model (CNN + clinical features)"""
        # Extract selected clinical features
        X_clinical = clinical_data[clinical_feature_columns].values
        
        # Match clinical data with image data if needed
        # This assumes image_data and clinical_data have the same ordering
        
        # Return format suitable for hybrid model:
        # ([image_input, clinical_input], labels)
        return ([image_data, X_clinical], clinical_data['diagnosis'].values)
    
    def split_by_pattern(self, images, labels, patterns, pattern_name):
        """Split dataset by pattern type (spiral or wave)"""
        mask = patterns == pattern_name
        return images[mask], labels[mask]
    
    def load_synthetic_image_dataset(self, dataset_path, image_size=(128, 128)):
        """
        Load synthetic spiral and wave images from the specified directory structure
        
        Args:
            dataset_path: Path to the dataset directory
            image_size: Size to resize images to
            
        Returns:
            tuple: ((train_images, train_labels, train_patterns), (test_images, test_labels, test_patterns))
        """
        # Check if the dataset path exists
        if not os.path.exists(dataset_path):
            print(f"Dataset path {dataset_path} does not exist, using generated data folder")
            dataset_path = "data"  # Use default data folder
        
        # Define patterns
        patterns = ["spiral", "wave"]
        
        # Initialize lists for data
        train_images = []
        train_labels = []
        train_patterns = []
        test_images = []
        test_labels = []
        test_patterns = []
        
        # Load images for each pattern
        for pattern in patterns:
            pattern_path = os.path.join(dataset_path, pattern)
            
            if not os.path.exists(pattern_path):
                print(f"Pattern path {pattern_path} does not exist, skipping...")
                continue
            
            # Load training data
            train_path = os.path.join(pattern_path, "training")
            if os.path.exists(train_path):
                # Load healthy samples
                healthy_path = os.path.join(train_path, "healthy")
                if os.path.exists(healthy_path):
                    healthy_files = os.listdir(healthy_path)
                    for file in healthy_files:
                        if file.endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(healthy_path, file)
                            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            
                            if image is not None:
                                image = cv2.resize(image, image_size)
                                image = image.astype('float32') / 255.0
                                image = np.expand_dims(image, axis=-1)
                                
                                train_images.append(image)
                                train_labels.append(0)  # 0 for healthy
                                train_patterns.append(pattern)
                
                # Load Parkinson's samples
                parkinson_path = os.path.join(train_path, "parkinson")
                if os.path.exists(parkinson_path):
                    parkinson_files = os.listdir(parkinson_path)
                    for file in parkinson_files:
                        if file.endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(parkinson_path, file)
                            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            
                            if image is not None:
                                image = cv2.resize(image, image_size)
                                image = image.astype('float32') / 255.0
                                image = np.expand_dims(image, axis=-1)
                                
                                train_images.append(image)
                                train_labels.append(1)  # 1 for Parkinson's
                                train_patterns.append(pattern)
            
            # Load testing data
            test_path = os.path.join(pattern_path, "testing")
            if os.path.exists(test_path):
                # Load healthy samples
                healthy_path = os.path.join(test_path, "healthy")
                if os.path.exists(healthy_path):
                    healthy_files = os.listdir(healthy_path)
                    for file in healthy_files:
                        if file.endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(healthy_path, file)
                            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            
                            if image is not None:
                                image = cv2.resize(image, image_size)
                                image = image.astype('float32') / 255.0
                                image = np.expand_dims(image, axis=-1)
                                
                                test_images.append(image)
                                test_labels.append(0)  # 0 for healthy
                                test_patterns.append(pattern)
                
                # Load Parkinson's samples
                parkinson_path = os.path.join(test_path, "parkinson")
                if os.path.exists(parkinson_path):
                    parkinson_files = os.listdir(parkinson_path)
                    for file in parkinson_files:
                        if file.endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(parkinson_path, file)
                            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            
                            if image is not None:
                                image = cv2.resize(image, image_size)
                                image = image.astype('float32') / 255.0
                                image = np.expand_dims(image, axis=-1)
                                
                                test_images.append(image)
                                test_labels.append(1)  # 1 for Parkinson's
                                test_patterns.append(pattern)
        
        # Convert lists to numpy arrays
        train_images = np.array(train_images) if train_images else np.empty((0, *image_size, 1))
        train_labels = np.array(train_labels) if train_labels else np.empty((0,))
        train_patterns = np.array(train_patterns) if train_patterns else np.empty((0,))
        test_images = np.array(test_images) if test_images else np.empty((0, *image_size, 1))
        test_labels = np.array(test_labels) if test_labels else np.empty((0,))
        test_patterns = np.array(test_patterns) if test_patterns else np.empty((0,))
        
        # Print dataset information
        print(f"Loaded {len(train_images)} training images and {len(test_images)} testing images")
        print(f"Training classes: {np.unique(train_labels, return_counts=True)}")
        print(f"Testing classes: {np.unique(test_labels, return_counts=True)}")
        
        return (train_images, train_labels, train_patterns), (test_images, test_labels, test_patterns)

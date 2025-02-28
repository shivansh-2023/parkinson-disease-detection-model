import os
import argparse
import shutil
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
from utils import create_sample_data_structure, organize_dataset
from config import *

def prepare_sample_data(data_dir="data", num_samples=20, overwrite=False):
    """
    Prepare sample data for testing the pipeline.
    
    Args:
        data_dir: Base directory for data
        num_samples: Number of samples to create per class
        overwrite: Whether to overwrite existing data
    """
    if overwrite:
        print(f"Overwriting existing data in {data_dir}...")
        # Create the directories
        for dir_path in [
            os.path.join(data_dir, "spiral", "healthy"),
            os.path.join(data_dir, "spiral", "parkinson"),
            os.path.join(data_dir, "wave", "healthy"),
            os.path.join(data_dir, "wave", "parkinson")
        ]:
            os.makedirs(dir_path, exist_ok=True)
            # Remove any existing files
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
    
    # Create sample data
    create_sample_data_structure(base_path=data_dir, num_samples=num_samples)
    
    print(f"Sample data created in {data_dir}")

def process_raw_data(raw_dir="data/raw", processed_dir="data", pattern="*.png"):
    """
    Process raw data by organizing it into the proper directory structure.
    
    Args:
        raw_dir: Directory containing raw data files
        processed_dir: Directory to store processed data
        pattern: File pattern to match
    """
    if not os.path.exists(raw_dir):
        print(f"Raw data directory {raw_dir} does not exist.")
        return
    
    # Organize dataset
    organize_dataset(raw_dir, processed_dir, pattern=pattern)
    
    print(f"Raw data from {raw_dir} organized into {processed_dir}")

def preprocess_images(data_dir="data", output_dir=None, target_size=(128, 128)):
    """
    Preprocess images by resizing, converting to grayscale, and thresholding.
    
    Args:
        data_dir: Directory containing the organized dataset
        output_dir: Directory to save preprocessed images (if None, overwrite original)
        target_size: Target size for resizing images
    """
    if output_dir is None:
        output_dir = data_dir
    
    # Directory paths
    dirs = [
        (os.path.join(data_dir, "spiral", "healthy"), os.path.join(output_dir, "spiral", "healthy")),
        (os.path.join(data_dir, "spiral", "parkinson"), os.path.join(output_dir, "spiral", "parkinson")),
        (os.path.join(data_dir, "wave", "healthy"), os.path.join(output_dir, "wave", "healthy")),
        (os.path.join(data_dir, "wave", "parkinson"), os.path.join(output_dir, "wave", "parkinson"))
    ]
    
    # Create output directories
    for _, out_dir in dirs:
        os.makedirs(out_dir, exist_ok=True)
    
    # Process images
    total_processed = 0
    for src_dir, dest_dir in dirs:
        if not os.path.exists(src_dir):
            print(f"Source directory {src_dir} does not exist. Skipping.")
            continue
        
        files = [f for f in os.listdir(src_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        for file in tqdm(files, desc=f"Processing {os.path.basename(src_dir)}"):
            src_path = os.path.join(src_dir, file)
            dest_path = os.path.join(dest_dir, file)
            
            # Read image
            img = cv2.imread(src_path)
            if img is None:
                print(f"Could not read image {src_path}. Skipping.")
                continue
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Resize
            resized = cv2.resize(gray, target_size)
            
            # Threshold to make drawing appear as white on black
            thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            
            # Save preprocessed image
            cv2.imwrite(dest_path, thresh)
            total_processed += 1
    
    print(f"Preprocessed {total_processed} images")

def create_clinical_data_file(data_dir="data", num_samples=None, overwrite=False):
    """
    Create a synthetic clinical data file for testing.
    
    Args:
        data_dir: Directory to save the clinical data file
        num_samples: Number of samples to generate (if None, count images)
        overwrite: Whether to overwrite existing file
    """
    clinical_data_path = os.path.join(data_dir, "parkinson_clinical_data.csv")
    
    if os.path.exists(clinical_data_path) and not overwrite:
        print(f"Clinical data file already exists at {clinical_data_path}. Use --overwrite to replace.")
        return
    
    # Count images if num_samples not specified
    if num_samples is None:
        num_samples = 0
        for category in ["healthy", "parkinson"]:
            for test_type in ["spiral", "wave"]:
                dir_path = os.path.join(data_dir, test_type, category)
                if os.path.exists(dir_path):
                    num_samples += len([f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if num_samples == 0:
        print("No images found and no sample count specified.")
        return
    
    # Generate random data
    np.random.seed(RANDOM_SEED)
    
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
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(clinical_data_path, index=False)
    
    print(f"Clinical data file created with {num_samples} samples at {clinical_data_path}")

def main():
    parser = argparse.ArgumentParser(description="Prepare data for Parkinson's disease detection")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Base directory for data")
    parser.add_argument("--create_sample", action="store_true",
                        help="Create sample data for testing")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of samples to create per class")
    parser.add_argument("--process_raw", action="store_true",
                        help="Process raw data into the proper directory structure")
    parser.add_argument("--raw_dir", type=str, default="data/raw",
                        help="Directory containing raw data files")
    parser.add_argument("--preprocess", action="store_true",
                        help="Preprocess images (resize, grayscale, threshold)")
    parser.add_argument("--create_clinical", action="store_true",
                        help="Create synthetic clinical data file")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing data")
    
    args = parser.parse_args()
    
    # Create base directories if they don't exist
    os.makedirs(args.data_dir, exist_ok=True)
    if args.raw_dir:
        os.makedirs(args.raw_dir, exist_ok=True)
    
    # Execute requested operations
    if args.create_sample:
        prepare_sample_data(data_dir=args.data_dir, num_samples=args.num_samples, overwrite=args.overwrite)
    
    if args.process_raw:
        process_raw_data(raw_dir=args.raw_dir, processed_dir=args.data_dir)
    
    if args.preprocess:
        preprocess_images(data_dir=args.data_dir)
    
    if args.create_clinical:
        create_clinical_data_file(data_dir=args.data_dir, num_samples=args.num_samples if args.create_sample else None, overwrite=args.overwrite)
        
    # If no options were specified, show help
    if not (args.create_sample or args.process_raw or args.preprocess or args.create_clinical):
        parser.print_help()

if __name__ == "__main__":
    main()

import os
import numpy as np
import cv2
from tqdm import tqdm
import random
import math

def generate_synthetic_spiral(size=(128, 128), turns=3, noise_level=0.1, is_parkinsons=False):
    """Generate a synthetic spiral image"""
    img = np.ones(size, dtype=np.uint8) * 255
    
    # Create spiral coordinates
    theta = np.linspace(0, 2 * np.pi * turns, 200)
    
    # Spiral equation: r = a * theta
    spiral_constant = 0.1
    radius = spiral_constant * theta
    
    # Convert to cartesian coordinates
    center_x, center_y = size[0] // 2, size[1] // 2
    x = center_x + radius * np.cos(theta) * (size[0] / 2 - 10) / (spiral_constant * 2 * np.pi * turns)
    y = center_y + radius * np.sin(theta) * (size[1] / 2 - 10) / (spiral_constant * 2 * np.pi * turns)
    
    # Add Parkinson's-like tremor
    if is_parkinsons:
        # Add more intense tremor
        tremor_frequency = random.uniform(4.0, 8.0)  # Hz
        tremor_amplitude = random.uniform(2.0, 5.0)  # pixels
        
        # Apply tremor to the spiral
        tremor_x = tremor_amplitude * np.sin(tremor_frequency * theta)
        tremor_y = tremor_amplitude * np.cos(tremor_frequency * theta)
        
        x += tremor_x
        y += tremor_y
        
        # Add some random breaks in the line (micrographia)
        breaks = np.random.choice([0, 1], size=len(x)-1, p=[0.9, 0.1])
    else:
        # Add slight natural tremor
        tremor_amplitude = random.uniform(0.5, 1.5)  # pixels
        tremor_x = tremor_amplitude * np.sin(theta)
        tremor_y = tremor_amplitude * np.cos(theta)
        
        x += tremor_x
        y += tremor_y
        
        # Fewer breaks for healthy individuals
        breaks = np.random.choice([0, 1], size=len(x)-1, p=[0.98, 0.02])
    
    # Add general noise
    x += np.random.normal(0, size[0] * noise_level, x.shape)
    y += np.random.normal(0, size[1] * noise_level, y.shape)
    
    # Clip to image boundaries
    x = np.clip(x, 0, size[0]-1)
    y = np.clip(y, 0, size[1]-1)
    
    # Draw the spiral
    points = np.column_stack((x, y)).astype(np.int32)
    
    for i in range(len(points)-1):
        if breaks[i] == 0:  # Only draw if not a break
            cv2.line(img, tuple(points[i]), tuple(points[i+1]), 0, thickness=2)
    
    return img

def generate_synthetic_wave(size=(128, 128), frequency=2, noise_level=0.1, is_parkinsons=False):
    """Generate a synthetic wave image"""
    img = np.ones(size, dtype=np.uint8) * 255
    
    # Create wave coordinates
    x = np.linspace(10, size[0]-10, 200)
    
    # Basic wave equation: y = amplitude * sin(frequency * x)
    amplitude = size[1] / 4
    y_baseline = size[1] / 2
    y = y_baseline + amplitude * np.sin(frequency * np.pi * x / size[0])
    
    # Add Parkinson's-like tremor
    if is_parkinsons:
        # Add more intense tremor
        tremor_frequency = random.uniform(4.0, 8.0)  # Hz
        tremor_amplitude = random.uniform(2.0, 5.0)  # pixels
        
        # Apply tremor
        fine_tremor = tremor_amplitude * np.sin(tremor_frequency * x / size[0] * 20)
        y += fine_tremor
        
        # Add variable pressure (thicker/thinner lines)
        thickness = np.random.choice([1, 2, 3], size=len(x)-1)
        
        # Add some random breaks in the line
        breaks = np.random.choice([0, 1], size=len(x)-1, p=[0.9, 0.1])
    else:
        # Add slight natural tremor
        tremor_amplitude = random.uniform(0.5, 1.0)  # pixels
        fine_tremor = tremor_amplitude * np.sin(np.pi * x / size[0] * 10)
        y += fine_tremor
        
        # More consistent thickness for healthy individuals
        thickness = np.random.choice([2, 3], size=len(x)-1, p=[0.7, 0.3])
        
        # Fewer breaks for healthy individuals
        breaks = np.random.choice([0, 1], size=len(x)-1, p=[0.98, 0.02])
    
    # Add general noise
    y += np.random.normal(0, size[1] * noise_level, y.shape)
    
    # Clip to image boundaries
    y = np.clip(y, 0, size[1]-1)
    
    # Draw the wave
    points = np.column_stack((x, y)).astype(np.int32)
    
    for i in range(len(points)-1):
        if breaks[i] == 0:  # Only draw if not a break
            cv2.line(img, tuple(points[i]), tuple(points[i+1]), 0, thickness=thickness[i])
    
    return img

def generate_dataset(base_path="data", num_samples=100):
    """Generate a synthetic dataset of spiral and wave images"""
    patterns = ["spiral", "wave"]
    splits = ["training", "testing"]
    classes = ["healthy", "parkinson"]
    
    # Determine samples per split
    train_samples = int(num_samples * 0.8)
    test_samples = num_samples - train_samples
    
    # Create directories if they don't exist
    for pattern in patterns:
        for split in splits:
            for cls in classes:
                os.makedirs(os.path.join(base_path, pattern, split, cls), exist_ok=True)
    
    print(f"Generating {num_samples} synthetic images per class...")
    
    # Generate spirals
    for is_parkinsons, cls in enumerate(classes):
        print(f"Generating {cls} spiral images...")
        
        # Training
        for i in tqdm(range(train_samples)):
            # Spirals
            spiral_img = generate_synthetic_spiral(
                is_parkinsons=bool(is_parkinsons), 
                noise_level=0.05 + (0.1 if is_parkinsons else 0)
            )
            cv2.imwrite(
                os.path.join(base_path, "spiral", "training", cls, f"{cls}_{i:03d}.jpg"),
                spiral_img
            )
            
            # Waves
            wave_img = generate_synthetic_wave(
                is_parkinsons=bool(is_parkinsons),
                noise_level=0.05 + (0.1 if is_parkinsons else 0)
            )
            cv2.imwrite(
                os.path.join(base_path, "wave", "training", cls, f"{cls}_{i:03d}.jpg"),
                wave_img
            )
        
        # Testing
        for i in tqdm(range(test_samples)):
            # Spirals with different parameters
            spiral_img = generate_synthetic_spiral(
                is_parkinsons=bool(is_parkinsons),
                turns=random.uniform(2.5, 3.5),
                noise_level=0.05 + (0.1 if is_parkinsons else 0)
            )
            cv2.imwrite(
                os.path.join(base_path, "spiral", "testing", cls, f"{cls}_{i:03d}.jpg"),
                spiral_img
            )
            
            # Waves with different parameters
            wave_img = generate_synthetic_wave(
                is_parkinsons=bool(is_parkinsons),
                frequency=random.uniform(1.8, 2.5),
                noise_level=0.05 + (0.1 if is_parkinsons else 0)
            )
            cv2.imwrite(
                os.path.join(base_path, "wave", "testing", cls, f"{cls}_{i:03d}.jpg"),
                wave_img
            )
    
    print(f"Dataset generation complete. Created {num_samples} images per class.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic Parkinson's dataset")
    parser.add_argument("--output_dir", type=str, default="data", help="Base output directory")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples per class")
    
    args = parser.parse_args()
    
    generate_dataset(base_path=args.output_dir, num_samples=args.samples)

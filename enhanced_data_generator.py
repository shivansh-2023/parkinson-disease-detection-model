import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

class EnhancedDataGenerator:
    """Generate enhanced synthetic data for Parkinson's disease detection"""
    
    def __init__(self, output_dir="data/enhanced_synthetic_data", image_size=(128, 128)):
        """
        Initialize the generator
        
        Args:
            output_dir: Directory to save generated images
            image_size: Size of the generated images
        """
        self.output_dir = output_dir
        self.image_size = image_size
        
        # Create output directories
        os.makedirs(os.path.join(output_dir, "train", "healthy", "spiral"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "train", "healthy", "wave"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "train", "parkinson", "spiral"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "train", "parkinson", "wave"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "test", "healthy", "spiral"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "test", "healthy", "wave"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "test", "parkinson", "spiral"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "test", "parkinson", "wave"), exist_ok=True)
    
    def generate_dataset(self, num_train=500, num_test=100):
        """
        Generate a complete dataset of synthetic images
        
        Args:
            num_train: Number of training images per class
            num_test: Number of test images per class
        """
        print("Generating enhanced synthetic dataset...")
        
        # Generate training data
        print("Generating training data...")
        self._generate_class_data("train", "healthy", "spiral", num_train, tremor_level=(0.0, 0.1))
        self._generate_class_data("train", "healthy", "wave", num_train, tremor_level=(0.0, 0.1))
        self._generate_class_data("train", "parkinson", "spiral", num_train, tremor_level=(0.3, 0.8))
        self._generate_class_data("train", "parkinson", "wave", num_train, tremor_level=(0.3, 0.8))
        
        # Generate test data
        print("Generating test data...")
        self._generate_class_data("test", "healthy", "spiral", num_test, tremor_level=(0.0, 0.1))
        self._generate_class_data("test", "healthy", "wave", num_test, tremor_level=(0.0, 0.1))
        self._generate_class_data("test", "parkinson", "spiral", num_test, tremor_level=(0.3, 0.8))
        self._generate_class_data("test", "parkinson", "wave", num_test, tremor_level=(0.3, 0.8))
        
        print("Dataset generation complete!")
    
    def _generate_class_data(self, split, condition, pattern_type, count, tremor_level):
        """
        Generate images for a specific class
        
        Args:
            split: 'train' or 'test'
            condition: 'healthy' or 'parkinson'
            pattern_type: 'spiral' or 'wave'
            count: Number of images to generate
            tremor_level: Range of tremor levels as (min, max)
        """
        output_dir = os.path.join(self.output_dir, split, condition, pattern_type)
        
        for i in tqdm(range(count), desc=f"{split}-{condition}-{pattern_type}"):
            if pattern_type == "spiral":
                image = self._generate_spiral(
                    tremor=np.random.uniform(tremor_level[0], tremor_level[1]),
                    completeness=np.random.uniform(0.85, 1.0) if condition == "healthy" else np.random.uniform(0.7, 0.95),
                    irregularity=np.random.uniform(0.0, 0.1) if condition == "healthy" else np.random.uniform(0.1, 0.3),
                    thickness=np.random.randint(1, 3) if condition == "healthy" else np.random.randint(1, 4)
                )
            else:  # wave
                image = self._generate_wave(
                    tremor=np.random.uniform(tremor_level[0], tremor_level[1]),
                    amplitude=np.random.uniform(0.9, 1.1) if condition == "healthy" else np.random.uniform(0.7, 1.3),
                    frequency=np.random.uniform(0.95, 1.05) if condition == "healthy" else np.random.uniform(0.8, 1.2),
                    thickness=np.random.randint(1, 3) if condition == "healthy" else np.random.randint(1, 4)
                )
            
            # Save the image
            output_path = os.path.join(output_dir, f"{i+1:04d}.png")
            cv2.imwrite(output_path, image)
    
    def _generate_spiral(self, tremor=0.0, completeness=1.0, irregularity=0.0, thickness=1):
        """
        Generate a spiral pattern
        
        Args:
            tremor: Amount of tremor/noise (0.0 to 1.0)
            completeness: How complete the spiral is (0.0 to 1.0)
            irregularity: Irregularity in the spiral spacing (0.0 to 1.0)
            thickness: Line thickness
            
        Returns:
            Spiral image as numpy array
        """
        image = np.zeros(self.image_size, dtype=np.uint8)
        
        center_x, center_y = self.image_size[0] // 2, self.image_size[1] // 2
        max_radius = min(center_x, center_y) * 0.9
        
        # Calculate number of turns based on completeness
        turns = 3 * completeness
        
        # Generate points along spiral
        points = []
        angle_step = 0.1
        for angle in np.arange(0, turns * 2 * np.pi, angle_step):
            # Calculate radius with some irregularity
            radius = angle / (2 * np.pi) * max_radius / turns
            radius *= (1 + irregularity * np.sin(angle * 5))
            
            # Add tremor/noise
            tremor_offset_x = np.random.normal(0, tremor * 5) if tremor > 0 else 0
            tremor_offset_y = np.random.normal(0, tremor * 5) if tremor > 0 else 0
            
            # Calculate point coordinates
            x = int(center_x + radius * np.cos(angle) + tremor_offset_x)
            y = int(center_y + radius * np.sin(angle) + tremor_offset_y)
            
            # Ensure point is within image bounds
            if 0 <= x < self.image_size[0] and 0 <= y < self.image_size[1]:
                points.append((x, y))
        
        # Draw spiral
        for i in range(1, len(points)):
            cv2.line(image, points[i-1], points[i], 255, thickness)
        
        return image
    
    def _generate_wave(self, tremor=0.0, amplitude=1.0, frequency=1.0, thickness=1):
        """
        Generate a wave pattern
        
        Args:
            tremor: Amount of tremor/noise (0.0 to 1.0)
            amplitude: Amplitude multiplier (0.5 to 1.5)
            frequency: Frequency multiplier (0.5 to 1.5)
            thickness: Line thickness
            
        Returns:
            Wave image as numpy array
        """
        image = np.zeros(self.image_size, dtype=np.uint8)
        
        width, height = self.image_size
        
        # Base amplitude and frequency
        base_amplitude = height // 6
        base_frequency = 3.0  # Number of complete waves
        
        # Apply multipliers
        amplitude = base_amplitude * amplitude
        frequency = base_frequency * frequency
        
        # Generate points along wave
        points = []
        for x in range(0, width):
            # Calculate sine wave position
            progress = x / width  # 0.0 to 1.0
            wave_y = height // 2 + amplitude * np.sin(progress * frequency * 2 * np.pi)
            
            # Add tremor/noise
            tremor_offset = np.random.normal(0, tremor * 10) if tremor > 0 else 0
            
            y = int(wave_y + tremor_offset)
            
            # Ensure point is within image bounds
            if 0 <= y < height:
                points.append((x, y))
        
        # Draw wave
        for i in range(1, len(points)):
            cv2.line(image, points[i-1], points[i], 255, thickness)
        
        return image
    
    def visualize_samples(self, num_samples=4):
        """
        Visualize sample generated images
        
        Args:
            num_samples: Number of samples to visualize per class
        """
        plt.figure(figsize=(12, 10))
        
        # Generate samples for visualization without saving
        for i, condition in enumerate(['healthy', 'parkinson']):
            for j, pattern in enumerate(['spiral', 'wave']):
                for k in range(num_samples):
                    plt.subplot(4, num_samples, i*2*num_samples + j*num_samples + k + 1)
                    
                    if pattern == 'spiral':
                        tremor = 0.05 if condition == 'healthy' else 0.4
                        completeness = 0.95 if condition == 'healthy' else 0.8
                        irregularity = 0.05 if condition == 'healthy' else 0.2
                        
                        image = self._generate_spiral(
                            tremor=tremor, 
                            completeness=completeness,
                            irregularity=irregularity
                        )
                    else:  # wave
                        tremor = 0.05 if condition == 'healthy' else 0.4
                        amplitude = 1.0 if condition == 'healthy' else 1.2
                        frequency = 1.0 if condition == 'healthy' else 0.9
                        
                        image = self._generate_wave(
                            tremor=tremor,
                            amplitude=amplitude,
                            frequency=frequency
                        )
                    
                    plt.imshow(image, cmap='gray')
                    plt.title(f"{condition} {pattern}")
                    plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "samples.png"))
        plt.close()

if __name__ == "__main__":
    # Create generator
    generator = EnhancedDataGenerator()
    
    # Visualize sample images
    generator.visualize_samples()
    
    # Generate complete dataset
    generator.generate_dataset(num_train=500, num_test=100)

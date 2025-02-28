import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import cv2

def excel_to_image(excel_file, output_directory, sheet_name=0, x_col='X', y_col='Y', 
                  label_col='Class', id_col='ID', image_size=(128, 128)):
    """
    Convert drawing coordinates from Excel file to images
    
    Parameters:
    -----------
    excel_file : str
        Path to the Excel file containing drawing coordinates
    output_directory : str
        Directory to save the generated images
    sheet_name : str or int, default=0
        Name or index of the sheet in Excel file
    x_col : str, default='X'
        Column name for X coordinates
    y_col : str, default='Y'
        Column name for Y coordinates
    label_col : str, default='Class'
        Column name for label (0 for healthy, 1 for Parkinson's)
    id_col : str, default='ID'
        Column name for patient ID
    image_size : tuple, default=(128, 128)
        Size of the output images
    """
    # Load Excel data
    print(f"Loading data from {excel_file}...")
    if excel_file.endswith('.csv'):
        df = pd.read_csv(excel_file)
    else:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
    
    # Create output directories
    healthy_dir = os.path.join(output_directory, 'healthy')
    parkinson_dir = os.path.join(output_directory, 'parkinson')
    
    os.makedirs(healthy_dir, exist_ok=True)
    os.makedirs(parkinson_dir, exist_ok=True)
    
    # Get unique patients
    unique_ids = df[id_col].unique()
    
    print(f"Converting data for {len(unique_ids)} patients...")
    for patient_id in tqdm(unique_ids):
        patient_data = df[df[id_col] == patient_id]
        
        # Get label (0 for healthy, 1 for Parkinson's)
        try:
            label = patient_data[label_col].iloc[0]
            is_parkinsons = bool(label)
        except:
            print(f"Warning: Could not determine label for patient {patient_id}, skipping...")
            continue
        
        # Create blank image
        img = np.ones(image_size, dtype=np.uint8) * 255
        
        # Get X and Y coordinates
        try:
            x_coords = patient_data[x_col].values
            y_coords = patient_data[y_col].values
            
            # Normalize coordinates to image size
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            if x_range == 0 or y_range == 0:
                print(f"Warning: Invalid coordinate range for patient {patient_id}, skipping...")
                continue
            
            # Scale coordinates to image size with padding
            padding = 10
            scaled_x = ((x_coords - x_min) / x_range) * (image_size[0] - 2*padding) + padding
            scaled_y = ((y_coords - y_min) / y_range) * (image_size[1] - 2*padding) + padding
            
            # Draw lines
            points = np.column_stack((scaled_x, scaled_y)).astype(np.int32)
            for i in range(len(points) - 1):
                cv2.line(img, tuple(points[i]), tuple(points[i+1]), 0, 2)
        
        except Exception as e:
            print(f"Error processing patient {patient_id}: {e}")
            continue
        
        # Save image
        output_dir = parkinson_dir if is_parkinsons else healthy_dir
        file_name = f"{patient_id}.jpg"
        output_path = os.path.join(output_dir, file_name)
        
        cv2.imwrite(output_path, img)
    
    print(f"Generated images saved to {output_directory}")
    print(f"Total healthy images: {len(os.listdir(healthy_dir))}")
    print(f"Total Parkinson's images: {len(os.listdir(parkinson_dir))}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Excel data to images")
    parser.add_argument("--excel_file", type=str, required=True, help="Path to Excel file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--sheet_name", type=str, default=0, help="Sheet name in Excel file")
    parser.add_argument("--x_col", type=str, default="X", help="Column name for X coordinates")
    parser.add_argument("--y_col", type=str, default="Y", help="Column name for Y coordinates")
    parser.add_argument("--label_col", type=str, default="Class", help="Column name for label")
    parser.add_argument("--id_col", type=str, default="ID", help="Column name for patient ID")
    
    args = parser.parse_args()
    
    excel_to_image(
        excel_file=args.excel_file,
        output_directory=args.output_dir,
        sheet_name=args.sheet_name,
        x_col=args.x_col,
        y_col=args.y_col,
        label_col=args.label_col,
        id_col=args.id_col
    )
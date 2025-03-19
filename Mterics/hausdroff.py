

"""
Created on Thu Mar 20 03:25:05 2025

@author: Istiak Ahmed
"""

# ====== result optimized hausdroff (Normalized) =======================================

import os
import numpy as np
from PIL import Image
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm
import matplotlib.pyplot as plt

# Class RGB values for visualization
class_rgb_values = {
    0: [0, 0, 0],        # Background
    1: [153, 178, 199],  # Calcification
    2: [170, 110, 240],  # Axilla Findings
    3: [216, 155, 8],    # Tissue
    4: [58, 237, 28]     # Mass
}

# Mapping of pixel values to class labels
pixel_to_class = {
    0: 0,    # Background
    118: 1,  # Calcification
    154: 2,  # Axilla Findings
    156: 3,  # Tissue
    181: 4   # Mass
}

def plot_mask(mask, title, class_rgb_values):
    """
    Plot a mask using the class RGB values for visualization.
    """
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in class_rgb_values.items():
        rgb_mask[mask == class_id] = color
    plt.imshow(rgb_mask)
    plt.title(title)
    plt.show()

def calculate_hausdorff_distance_by_class(gt_folder, pred_folder, num_classes=5):
    """
    Calculate the Hausdorff distance for each class between ground truth and predicted masks.
    
    Args:
        gt_folder (str): Path to the folder containing ground truth masks (.png).
        pred_folder (str): Path to the folder containing predicted masks (.png).
        num_classes (int): Number of classes (including background).
    
    Returns:
        dict: A dictionary with mean Hausdorff distances for each class.
    """
    # List all files in the folders
    gt_files = sorted(os.listdir(gt_folder))
    pred_files = sorted(os.listdir(pred_folder))
    
    # Initialize results
    hausdorff_distances = {c: [] for c in range(num_classes)}
    
    # Use tqdm to create a progress bar for the loop over files
    for gt_file, pred_file in tqdm(zip(gt_files, pred_files), total=len(gt_files), desc="Processing Images"):
        # Load masks as grayscale images
        gt_mask = np.array(Image.open(os.path.join(gt_folder, gt_file)).convert("L"))
        pred_mask = np.array(Image.open(os.path.join(pred_folder, pred_file)).convert("L"))
        
        # Ensure masks have the same shape
        assert gt_mask.shape == pred_mask.shape, "Ground truth and predicted masks must have the same shape."
        
        # Convert pixel values to class labels
        gt_mask_mapped = np.zeros_like(gt_mask)
        pred_mask_mapped = np.zeros_like(pred_mask)
        for pixel_value, class_label in pixel_to_class.items():
            gt_mask_mapped[gt_mask == pixel_value] = class_label
            pred_mask_mapped[pred_mask == pixel_value] = class_label
        
        # Debug: Print unique values in masks
        print(f"\nFile: {gt_file}")
        print(f"Ground Truth Mask Mapped Unique Values: {np.unique(gt_mask_mapped)}")
        print(f"Predicted Mask Mapped Unique Values: {np.unique(pred_mask_mapped)}")
        
        # Debug: Visualize masks
        plot_mask(gt_mask_mapped, f"Ground Truth Mask: {gt_file}", class_rgb_values)
        plot_mask(pred_mask_mapped, f"Predicted Mask: {pred_file}", class_rgb_values)
        
        # Compute Hausdorff distance for each class
        for c in range(num_classes):
            # Create binary masks for class c
            gt_binary = (gt_mask_mapped == c).astype(np.bool_)
            pred_binary = (pred_mask_mapped == c).astype(np.bool_)
            
            # Debug: Print number of pixels for each class
            print(f"Class {c}: GT Pixels = {np.sum(gt_binary)}, Pred Pixels = {np.sum(pred_binary)}")
            
            # Get coordinates of non-zero pixels
            gt_coords = np.argwhere(gt_binary)
            pred_coords = np.argwhere(pred_binary)
            
            # Handle cases where one of the masks is empty
            if len(gt_coords) == 0 or len(pred_coords) == 0:
                print(f"Class {c}: One of the masks is empty. Skipping calculation.")
                hausdorff_distances[c].append(np.nan)  # Use NaN for missing classes
                continue
            
            # Compute directed Hausdorff distances
            d1 = directed_hausdorff(gt_coords, pred_coords)[0]
            d2 = directed_hausdorff(pred_coords, gt_coords)[0]
            
            # Symmetric Hausdorff distance
            hausdorff_dist = max(d1, d2)
            
            # Normalize Hausdorff distance by image diagonal
            image_diagonal = np.sqrt(gt_mask.shape[0]**2 + gt_mask.shape[1]**2)
            hausdorff_dist_normalized = hausdorff_dist / image_diagonal
            
            hausdorff_distances[c].append(hausdorff_dist_normalized)
            print(f"Class {c}: Hausdorff Distance = {hausdorff_dist_normalized:.4f}")
            
            # Debug: Visualize Class 4 masks if present
            if c == 4 and np.sum(gt_binary) > 0 and np.sum(pred_binary) > 0:
                gt_mask_class4 = gt_binary.astype(np.uint8) * 255
                pred_mask_class4 = pred_binary.astype(np.uint8) * 255
                
                plt.imshow(gt_mask_class4, cmap='gray')
                plt.title(f"Ground Truth Mask - Class 4: {gt_file}")
                plt.show()
                
                plt.imshow(pred_mask_class4, cmap='gray')
                plt.title(f"Predicted Mask - Class 4: {pred_file}")
                plt.show()
    
    # Aggregate results (mean Hausdorff distance per class)
    mean_hausdorff_distances = {}
    for c in range(num_classes):
        valid_distances = [d for d in hausdorff_distances[c] if not np.isnan(d)]
        if len(valid_distances) > 0:
            mean_hausdorff_distances[c] = np.mean(valid_distances)
        else:
            mean_hausdorff_distances[c] = np.nan  # No valid distances for this class
    
    # Debug: Print number of valid distances per class
    print("\nNumber of Valid Distances per Class:")
    for c in range(num_classes):
        valid_distances = [d for d in hausdorff_distances[c] if not np.isnan(d)]
        print(f"Class {c}: {len(valid_distances)} valid distances")
    
    return mean_hausdorff_distances

# Example usage
if __name__ == "__main__":
    gt_folder = "F:/Istiak/Dataset/BSHL/2024/Breast Mammography/Breast Cancer/multimodal_merged_set_1_2_3_BIRADS_rev/aug/cropped/256/random_state_0/test/masks"
    pred_folder = "F:/Istiak/Dataset/BSHL/2024/Breast Mammography/Breast Cancer/multimodal_merged_set_1_2_3_BIRADS_rev/aug/cropped/256/random_state_0/test/predicted_masks"
    
    # Calculate Hausdorff distances
    results = calculate_hausdorff_distance_by_class(gt_folder, pred_folder, num_classes=5)
    
    print("\nMean Hausdorff Distances by Class:")
    for c, dist in results.items():
        print(f"Class {c}: {dist:.4f}")

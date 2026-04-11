"""
================================================================================
BREAST MAMMOGRAM MULTI-MODAL ANALYSIS SYSTEM WITH RULE-BASED REFINEMENT
================================================================================

This script analyzes mammogram images and provides:
1. Segmentation visualization with bounding boxes (Original, Mask, Overlay)
2. Rule-based clinical predictions where segmentation is the primary detector
3. Clinical model only provides detailed characterization when segmentation detects findings
4. MAX value consolidation across images for final patient-level prediction

RULES:
- Mass detected in segmentation → mass=1, mass-related features = most probable non-zero categories
- No mass in segmentation → ALL mass-related features = 0
- Calcification detected → calcification=1, calcification_type = most probable non-zero
- No calcification → calcification=0, calcification_type=0
- Axilla detected → axilla_findings=1
- No axilla → axilla_findings=0
================================================================================
"""

import os
import cv2
import numpy as np
import joblib
import tensorflow as tf
from keras.models import load_model
from keras.utils import normalize
from tensorflow.keras.saving import register_keras_serializable
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# ================================================================================
# CONFIGURATION
# ================================================================================

# Image sizes for different models
SIZE_X_SEGMENTATION, SIZE_Y_SEGMENTATION = 256, 256
SIZE_X_CLINICAL, SIZE_Y_CLINICAL = 512, 512

# Model paths - UPDATE THESE PATHS
SEGMENTATION_MODEL_PATH = "I:/Breast_Paper/papers/Methods/Models/256/best_model1_t1_120_att_Epochs (1).keras"

CLINICAL_MODEL_PATHS = {
    'feature_extractor': 'I:/Breast_Paper/papers/Methods/Models/Inference/models/512/v2/densenet121_feature_extractor_512x512_v2.keras',
    'mlp_model': 'I:/Breast_Paper/papers/Methods/Models/Inference/models/512/v2/image_to_clinical_model_512x512_v2.keras',
    'scaler': 'I:/Breast_Paper/papers/Methods/Models/Inference/models/512/v2/image_feature_scaler_512x512_v2.pkl',
    'label_encoders': 'I:/Breast_Paper/papers/Methods/Models/Inference/models/512/v2/label_encoders.pkl'
}

# Patient folder path - UPDATE THIS
PATIENT_FOLDER_PATH = r"I:/Breast_Paper/report_only/models/P1"

# Class RGB values for segmentation visualization
class_rgb_values = {
    0: [0, 0, 0],        # Background
    1: [153, 178, 199],  # Calcification
    2: [170, 110, 240],  # Axilla_Findings
    3: [216, 155, 8],    # Tissue
    4: [58, 237, 28]     # Mass
}

class_names = {
    0: "Background",
    1: "Calcification",
    2: "Axilla_Findings",
    3: "Tissue",
    4: "Mass"
}

# Mass-related features that should be set to non-zero when mass is detected
MASS_FEATURES = ['mass_definition', 'mass_density', 'mass_shape', 'mass_calcification']

# Define the desired feature order for display
FEATURE_ORDER = [
    'mass', 
    'mass_definition', 
    'mass_density', 
    'mass_shape', 
    'mass_calcification',
    'axilla_findings',
    'calcification', 
    'calcification_type', 
    'acr_breast_density', 
    'BIRADS_CAT'
]

# Minimum area threshold for detection (in pixels)
MIN_AREA_THRESHOLD = 50

# ================================================================================
# CUSTOM LOSS FUNCTIONS
# ================================================================================

@register_keras_serializable()
def focal_loss(y_true, y_pred, gamma=2.0):
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    cross_entropy = -y_true * tf.math.log(y_pred)
    loss = tf.pow(1 - y_pred, gamma) * cross_entropy
    return tf.reduce_mean(loss, axis=-1)

@register_keras_serializable()
def soft_dice_loss(y_true, y_pred, smooth=1):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    sum_true = tf.reduce_sum(y_true, axis=(1, 2, 3))
    sum_pred = tf.reduce_sum(y_pred, axis=(1, 2, 3))
    dice_coefficient = (2. * intersection + smooth) / (sum_true + sum_pred + smooth)
    dice_loss = 1 - dice_coefficient
    return tf.reduce_mean(dice_loss)

@register_keras_serializable()
def combined_loss(y_true, y_pred, gamma=2.0, alpha=0.5):
    focal = focal_loss(y_true, y_pred, gamma)
    dice = soft_dice_loss(y_true, y_pred)
    return alpha * focal + (1 - alpha) * dice

@register_keras_serializable()
class CustomMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self, num_classes, name='mean_iou', dtype='float32', ignore_class=None,
                 sparse_y_true=True, sparse_y_pred=True, axis=-1):
        super(CustomMeanIoU, self).__init__(num_classes=num_classes, name=name, dtype=dtype, ignore_class=ignore_class)
        self.sparse_y_true = sparse_y_true
        self.sparse_y_pred = sparse_y_pred
        self.axis = axis

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.math.argmax(y_true, axis=-1)
        y_pred = tf.math.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

    def get_config(self):
        config = super().get_config()
        config.update({'sparse_y_true': self.sparse_y_true, 'sparse_y_pred': self.sparse_y_pred, 'axis': self.axis})
        return config

# ================================================================================
# MAIN ANALYZER CLASS
# ================================================================================

class MammogramAnalyzer:
    def __init__(self, segmentation_model_path, clinical_model_paths):
        print("Loading models...")
        
        # Load segmentation model
        custom_objects = {
            'combined_loss': combined_loss,
            'focal_loss': focal_loss,
            'soft_dice_loss': soft_dice_loss,
            'soft_dice_coefficient': lambda y_true, y_pred: 1 - soft_dice_loss(y_true, y_pred),
            'CustomMeanIoU': CustomMeanIoU
        }
        self.segmentation_model = load_model(segmentation_model_path, custom_objects=custom_objects, compile=False)
        print(f"✅ Segmentation model loaded (input size: {SIZE_X_SEGMENTATION}x{SIZE_Y_SEGMENTATION})")
        
        # Load clinical prediction models
        self.feature_extractor = tf.keras.models.load_model(clinical_model_paths['feature_extractor'])
        self.mlp_model = tf.keras.models.load_model(clinical_model_paths['mlp_model'])
        self.scaler = joblib.load(clinical_model_paths['scaler'])
        self.label_encoders = joblib.load(clinical_model_paths['label_encoders'])
        print(f"✅ Clinical models loaded (input size: {SIZE_X_CLINICAL}x{SIZE_Y_CLINICAL})\n")
    
    def preprocess_image_for_segmentation(self, image_path):
        """Preprocess image for segmentation model"""
        img = cv2.imread(image_path, 1)
        if img is None:
            raise ValueError(f"Could not load: {image_path}")
        img = cv2.resize(img, (SIZE_X_SEGMENTATION, SIZE_Y_SEGMENTATION))
        img = img.astype(np.float32)
        img = normalize(img, axis=1)
        return np.expand_dims(img, axis=0)
    
    def preprocess_image_for_clinical(self, image_path):
        """Preprocess image for clinical prediction model"""
        img = cv2.imread(image_path, 1)
        if img is None:
            raise ValueError(f"Could not load: {image_path}")
        img = cv2.resize(img, (SIZE_X_CLINICAL, SIZE_Y_CLINICAL))
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return np.expand_dims(img, axis=0)
    
    def get_segmentation_mask(self, image_path):
        """Get segmentation mask from segmentation model"""
        img_processed = self.preprocess_image_for_segmentation(image_path)
        pred = self.segmentation_model.predict(img_processed, verbose=0)
        pred_mask = np.argmax(pred[0], axis=-1)
        return pred_mask
    
    def get_detection_boxes(self, pred_mask):
        """Extract bounding boxes for Mass, Calcification, and Axilla Findings"""
        boxes = {}
        
        for class_id, class_name in class_names.items():
            if class_name in ['Mass', 'Calcification', 'Axilla_Findings']:
                binary_mask = (pred_mask == class_id).astype(np.uint8)
                if np.sum(binary_mask) > 0:
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    boxes[class_name] = []
                    for contour in contours:
                        if len(contour) > 0:
                            x, y, w, h = cv2.boundingRect(contour)
                            if w * h > MIN_AREA_THRESHOLD:
                                boxes[class_name].append((x, y, w, h))
        return boxes
    
    def check_segmentation_findings(self, pred_mask):
        """Check which findings are present in segmentation - SAME LOGIC as get_detection_boxes"""
        findings = {
            'Mass': False,
            'Calcification': False,
            'Axilla_Findings': False
        }
        
        for class_id, class_name in class_names.items():
            if class_name in findings:
                binary_mask = (pred_mask == class_id).astype(np.uint8)
                if np.sum(binary_mask) > 0:
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        if len(contour) > 0:
                            x, y, w, h = cv2.boundingRect(contour)
                            if w * h > MIN_AREA_THRESHOLD:
                                findings[class_name] = True
                                break  # Found at least one valid region
        
        return findings
    
    def visualize_segmentation(self, image_path, pred_mask):
        """Create and display original, mask, and overlay with bounding boxes"""
        # Load and resize original to match segmentation size for visualization
        original = cv2.imread(image_path, 1)
        original = cv2.resize(original, (SIZE_X_SEGMENTATION, SIZE_Y_SEGMENTATION))
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Create colored mask
        colored_mask = np.zeros((SIZE_X_SEGMENTATION, SIZE_Y_SEGMENTATION, 3), dtype=np.uint8)
        for class_id, rgb in class_rgb_values.items():
            colored_mask[pred_mask == class_id] = rgb
        
        # Create overlay
        overlay = cv2.addWeighted(original_rgb, 0.6, colored_mask, 0.4, 0)
        
        # Get bounding boxes
        boxes = self.get_detection_boxes(pred_mask)
        
        # Draw bounding boxes on overlay and original
        for class_name, box_list in boxes.items():
            color = class_rgb_values[list(class_names.keys())[list(class_names.values()).index(class_name)]]
            for (x, y, w, h) in box_list:
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
                cv2.putText(overlay, class_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.rectangle(original_rgb, (x, y), (x + w, y + h), color, 2)
                cv2.putText(original_rgb, class_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original_rgb)
        axes[0].set_title('Original with Bounding Boxes')
        axes[0].axis('off')
        
        axes[1].imshow(colored_mask)
        axes[1].set_title('Segmentation Mask')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay with Bounding Boxes')
        axes[2].axis('off')
        
        plt.suptitle(f'{Path(image_path).name}', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        # Print detection summary
        detected = []
        for class_name in ['Mass', 'Calcification', 'Axilla_Findings']:
            if class_name in boxes and boxes[class_name]:
                detected.append(f"{class_name} ({len(boxes[class_name])} region(s))")
        
        if detected:
            print(f"✓ Detected: {', '.join(detected)}")
        else:
            print("✗ No findings detected")
        
        return boxes
    
    def get_clinical_predictions_with_probs(self, image_path):
        """Get clinical predictions with full probability distributions"""
        img = self.preprocess_image_for_clinical(image_path)
        features = self.feature_extractor.predict(img, verbose=0)
        features_scaled = self.scaler.transform(features)
        predictions = self.mlp_model.predict(features_scaled, verbose=0)
        
        results = {}
        for col, pred in predictions.items():
            pred_probs = pred[0]
            
            # Store full probability distribution
            if len(pred_probs) == 1:
                # Binary classification
                probs_array = np.array([1 - pred_probs[0], pred_probs[0]])
                classes = self.label_encoders[col].classes_
                results[col] = {
                    'probabilities': dict(zip(classes, probs_array))
                }
            else:
                # Multi-class classification
                classes = self.label_encoders[col].classes_
                results[col] = {
                    'probabilities': dict(zip(classes, pred_probs))
                }
        
        return results
    
    def apply_rule_based_refinement(self, segmentation_findings, clinical_probs):
        """
        Apply rule-based refinement:
        - If segmentation detects Mass → mass=1, mass-related features = most probable non-zero
        - If no Mass → all mass-related features = 0
        - If segmentation detects Calcification → calcification=1, calcification_type = most probable non-zero
        - If no Calcification → calcification=0, calcification_type=0
        - If segmentation detects Axilla → axilla_findings=1
        - If no Axilla → axilla_findings=0
        """
        refined = {}
        
        # Handle Mass
        if segmentation_findings['Mass']:
            refined['mass'] = 1
            
            # For each mass-related feature, find the most probable non-zero category
            for feature in MASS_FEATURES:
                probs = clinical_probs.get(feature, {}).get('probabilities', {})
                if probs:
                    # Filter out zero category and get the one with highest probability
                    non_zero_probs = {k: v for k, v in probs.items() if int(k) != 0}
                    if non_zero_probs:
                        most_probable = max(non_zero_probs, key=non_zero_probs.get)
                        refined[feature] = int(most_probable)
                    else:
                        refined[feature] = 0
                else:
                    refined[feature] = 0
        else:
            refined['mass'] = 0
            for feature in MASS_FEATURES:
                refined[feature] = 0
        
        # Handle Calcification
        if segmentation_findings['Calcification']:
            refined['calcification'] = 1
            
            # For calcification_type, find the most probable non-zero category
            probs = clinical_probs.get('calcification_type', {}).get('probabilities', {})
            if probs:
                non_zero_probs = {k: v for k, v in probs.items() if int(k) != 0}
                if non_zero_probs:
                    most_probable = max(non_zero_probs, key=non_zero_probs.get)
                    refined['calcification_type'] = int(most_probable)
                else:
                    refined['calcification_type'] = 0
            else:
                refined['calcification_type'] = 0
        else:
            refined['calcification'] = 0
            refined['calcification_type'] = 0
        
        # Handle Axilla Findings
        if segmentation_findings['Axilla_Findings']:
            refined['axilla_findings'] = 1
        else:
            refined['axilla_findings'] = 0
        
        # Keep other features unchanged (acr_breast_density, BIRADS_CAT, etc.)
        other_features = ['acr_breast_density', 'BIRADS_CAT']
        for feature in other_features:
            if feature in clinical_probs:
                probs = clinical_probs[feature]['probabilities']
                if probs:
                    most_probable = max(probs, key=probs.get)
                    refined[feature] = int(most_probable)
                else:
                    refined[feature] = 0
            else:
                refined[feature] = 0
        
        return refined
    
    def predict_clinical_with_rules(self, image_path):
        """Get clinical predictions with rule-based refinement"""
        # Get segmentation mask and findings
        seg_mask = self.get_segmentation_mask(image_path)
        segmentation_findings = self.check_segmentation_findings(seg_mask)
        
        # Get clinical probabilities
        clinical_probs = self.get_clinical_predictions_with_probs(image_path)
        
        # Apply rule-based refinement
        refined_predictions = self.apply_rule_based_refinement(segmentation_findings, clinical_probs)
        
        return refined_predictions, segmentation_findings, seg_mask

# ================================================================================
# PATIENT ANALYSIS
# ================================================================================

def process_patient_folder(patient_folder_path, analyzer):
    patient_folder = Path(patient_folder_path)
    patient_id = patient_folder.name
    
    # Get unique image files (no duplicates)
    image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(patient_folder.glob(f"*{ext}"))
        image_files.extend(patient_folder.glob(f"*{ext.upper()}"))
    
    # Remove duplicates by using set
    image_files = list(set(image_files))
    
    if not image_files:
        print(f"No images found in {patient_folder}")
        return None
    
    print(f"\n{'='*80}")
    print(f"PATIENT: {patient_id}")
    print(f"{'='*80}\n")
    
    # Store per-image predictions
    all_predictions = []
    
    # Process each image
    for img_path in sorted(image_files):
        print(f"\n{'-'*80}")
        print(f"Image: {img_path.name}")
        print(f"{'-'*80}")
        
        # Get refined clinical predictions and segmentation
        clinical_results, segmentation_findings, seg_mask = analyzer.predict_clinical_with_rules(str(img_path))
        
        # Visualize segmentation
        analyzer.visualize_segmentation(str(img_path), seg_mask)
        
        # Print segmentation findings
        print(f"\nSegmentation Findings:")
        for finding, detected in segmentation_findings.items():
            print(f"  {finding}: {'✓' if detected else '✗'}")
        
        # Print refined clinical predictions in desired order
        print(f"\nRefined Clinical Predictions:")
        for feature in FEATURE_ORDER:
            if feature in clinical_results:
                print(f"  {feature}: {clinical_results[feature]}")
        
        # Store for consolidation (using original order for DataFrame)
        prediction_row = {'image_id': img_path.stem}
        for feature in FEATURE_ORDER:
            prediction_row[feature] = clinical_results.get(feature, 0)
        all_predictions.append(prediction_row)
    
    # Create DataFrame with proper column order
    df = pd.DataFrame(all_predictions)
    # Ensure columns are in the desired order
    df = df[['image_id'] + FEATURE_ORDER]
    
    # Print per-image predictions table
    print(f"\n{'='*80}")
    print("PER-IMAGE CLINICAL PREDICTIONS (After Rule-Based Refinement)")
    print(f"{'='*80}\n")
    print(df.to_string(index=False))
    
    # Calculate final predictions (MAX value across all images)
    final_predictions = {}
    for feature in FEATURE_ORDER:
        final_predictions[feature] = df[feature].max()
    
    # Print final prediction
    print(f"\n{'='*80}")
    print("FINAL PATIENT-LEVEL PREDICTION (MAX Value Across All Images)")
    print(f"{'='*80}\n")
    
    final_df = pd.DataFrame([final_predictions])
    print(final_df.to_string(index=False))
    
    # Compact format in desired order
    compact_values = [str(final_predictions[f]) for f in FEATURE_ORDER]
    print(f"\nCompact Format: [{', '.join(compact_values)}]")
    
    print(f"\n{'='*80}")
    print(f"Analysis complete for patient: {patient_id}")
    print(f"{'='*80}")

# ================================================================================
# MAIN
# ================================================================================

def main():
    print("="*80)
    print("BREAST MAMMOGRAM ANALYSIS SYSTEM WITH RULE-BASED REFINEMENT")
    print("="*80)
    
    if not os.path.exists(PATIENT_FOLDER_PATH):
        print(f"\nERROR: Patient folder not found: {PATIENT_FOLDER_PATH}")
        return
    
    analyzer = MammogramAnalyzer(SEGMENTATION_MODEL_PATH, CLINICAL_MODEL_PATHS)
    process_patient_folder(PATIENT_FOLDER_PATH, analyzer)

if __name__ == "__main__":
    main()

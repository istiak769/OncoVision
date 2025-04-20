# -*- coding: utf-8 -*-
"""
Created on Fri Mar 07 11:57:40 2025

@author: Istiak Ahmed
"""


#----------------- Model Prediction (Original, Ground Truth Label & Predicted Label) ---------------------------------# 

import os
import cv2
import numpy as np
from keras.models import load_model
from keras.utils import normalize
import tensorflow as tf
from keras.saving import register_keras_serializable
from PIL import Image
import matplotlib.pyplot as plt
import random

# Define custom loss and metric functions (from your original code)
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

# Load the model once
model_path = "H:/Research/Breast Cancer/multimodal_merged_set_1_2_3_BIRADS_rev/aug/cropped/512/multimodal_merged_aug_set_1_2_3_resized_512_updated_124epochs.keras"
custom_objects = {
    'combined_loss': combined_loss,
    'focal_loss': focal_loss,
    'soft_dice_loss': soft_dice_loss,
    'soft_dice_coefficient': lambda y_true, y_pred: 1 - soft_dice_loss(y_true, y_pred),
    'CustomMeanIoU': CustomMeanIoU
}
model = load_model(model_path, custom_objects=custom_objects, compile=True)

# Class RGB values
class_rgb_values = {
    0: [0, 0, 0],        # Background
    1: [153, 178, 199],  # Calcification
    2: [170, 110, 240],  # Axilla_Findings
    3: [216, 155, 8],    # Tissue
    4: [58, 237, 28]     # Mass
}

def segment_image(image, image_size=(512, 512)):
    # Convert PIL image to numpy array
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
    if img is None:
        raise ValueError("Could not load image")
   
    original_shape = img.shape[:2]
    img_resized = cv2.resize(img, image_size)
    img_normalized = normalize(np.array([img_resized], dtype=np.float32), axis=1)
    prediction = model.predict(img_normalized)
    predicted_mask = np.argmax(prediction, axis=-1)[0]
    predicted_mask_resized = cv2.resize(predicted_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create RGB mask
    rgb_mask = np.zeros((*predicted_mask_resized.shape, 3), dtype=np.uint8)
    for class_idx, rgb in class_rgb_values.items():
        rgb_mask[predicted_mask_resized == class_idx] = rgb

    # Convert back to PIL for visualization
    rgb_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_mask)

def save_plot_in_vector_format(fig, save_path, format='eps'):
    """
    Save the plot in a vector format (e.g., EPS or SVG).
    
    Args:
        fig: The Matplotlib figure object to save.
        save_path: Path (including filename) where the plot will be saved.
        format: Format to save the plot ('eps' or 'svg'). Default is 'eps'.
    """
    if format not in ['eps', 'svg']:
        raise ValueError("Unsupported format. Use 'eps' or 'svg'.")
    
    # Save the figure in the specified vector format
    fig.savefig(save_path, format=format, bbox_inches='tight', dpi=600)
    print(f"Plot saved in {format.upper()} format at: {save_path}")


def visualize_random_image_and_mask(images_folder, masks_folder, image_size=(512, 512), save_path=None):
    """
    Visualize a random image, true mask, and predicted mask.
    Optionally save the plot in a vector format.
    
    Args:
        images_folder: Path to the folder containing input images.
        masks_folder: Path to the folder containing true masks.
        image_size: Size to which images are resized (default: (256, 256)).
        save_path: Path to save the plot in vector format (optional).
    """
    # Get list of images and masks
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not image_files:
        print("No images found in the images folder.")
        return

    # Select a random image
    random_image_file = random.choice(image_files)
    print(f"Selected random image: {random_image_file}")

    # Load the image and corresponding mask
    image_path = os.path.join(images_folder, random_image_file)
    mask_path = os.path.join(masks_folder, random_image_file)  # Assuming mask has the same filename

    image = Image.open(image_path)
    true_mask = Image.open(mask_path)

    # Predict the segmentation mask
    predicted_mask = segment_image(image, image_size)

    # Create the figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    # Original Image
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=7)
    axes[0].axis('off')

    # True Mask
    axes[1].imshow(true_mask, cmap='cividis')
    axes[1].set_title("True Mask", fontsize=7)
    axes[1].axis('off')

    # Predicted Mask
    axes[2].imshow(predicted_mask, cmap='cividis')
    axes[2].set_title("Predicted Mask", fontsize=7)
    axes[2].axis('off')

    plt.tight_layout()

    # Save the plot in vector format if save_path is provided
    if save_path:
        save_plot_in_vector_format(fig, save_path, format='eps')  # Change format to 'svg' if needed

    plt.show()


# Example usage
images_folder = "H:/Research/Breast Cancer/multimodal_merged_set_1_2_3_BIRADS_rev/aug/cropped/512/images"  # Replace with your images folder path
masks_folder = "H:/Research/Breast Cancer/multimodal_merged_set_1_2_3_BIRADS_rev/aug/cropped/512/masks"  # Replace with your masks folder path

# Define the path to save the plot in vector format
save_path = "H:/Research/Breast Cancer/multimodal_merged_set_1_2_3_BIRADS_rev/aug/cropped/512/Visualization/random_image_and_mask_visualization.eps"  # Change to 'random_image_and_mask_visualization.svg' for SVG format

# Visualize and save the plot
visualize_random_image_and_mask(images_folder, masks_folder, save_path=save_path)







#----------------- Error Map Analysis (Original, Ground Truth Label, Predicted Label & Error Map Overlay) ---------------------------------# 

import os
import cv2
import numpy as np
from keras.models import load_model
from keras.utils import normalize
import tensorflow as tf
from keras.saving import register_keras_serializable
from PIL import Image
import matplotlib.pyplot as plt
import random

# [Previous custom loss and metric functions remain the same]
# ... (keeping your existing custom functions unchanged)

# Load the model
model_path = "H:/Research/Breast Cancer/multimodal_merged_set_1_2_3_BIRADS_rev/aug/cropped/512/multimodal_merged_aug_set_1_2_3_resized_512_updated_124epochs.keras"
custom_objects = {
    'combined_loss': combined_loss,
    'focal_loss': focal_loss,
    'soft_dice_loss': soft_dice_loss,
    'soft_dice_coefficient': lambda y_true, y_pred: 1 - soft_dice_loss(y_true, y_pred),
    'CustomMeanIoU': CustomMeanIoU
}
model = load_model(model_path, custom_objects=custom_objects, compile=True)

# Desired RGB colors for display
class_rgb_values_rgb = {
    0: [0, 0, 0],        # Background
    1: [153, 178, 199],  # Calcification
    2: [170, 110, 240],  # Axilla_Findings
    3: [216, 155, 8],    # Tissue
    4: [58, 237, 28]     # Mass
}

# BGR values for true mask conversion (reversed from RGB, matching your true mask)
class_rgb_values_bgr = {k: [rgb[2], rgb[1], rgb[0]] for k, rgb in class_rgb_values_rgb.items()}

def segment_image(image, image_size=(512, 512)):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if img is None:
        raise ValueError("Could not load image")
   
    original_shape = img.shape[:2]
    img_resized = cv2.resize(img, image_size)
    img_normalized = normalize(np.array([img_resized], dtype=np.float32), axis=1)
    prediction = model.predict(img_normalized)
    predicted_mask = np.argmax(prediction, axis=-1)[0]
    predicted_mask_resized = cv2.resize(predicted_mask, (original_shape[1], original_shape[0]), 
                                      interpolation=cv2.INTER_NEAREST)
    
    # Use RGB values directly for display
    rgb_mask = np.zeros((*predicted_mask_resized.shape, 3), dtype=np.uint8)
    for class_idx, rgb in class_rgb_values_rgb.items():
        rgb_mask[predicted_mask_resized == class_idx] = rgb
    
    return Image.fromarray(rgb_mask), predicted_mask_resized

def rgb_to_class_indices(bgr_mask):
    """Convert BGR mask to class indices based on class_rgb_values_bgr"""
    class_indices = np.zeros(bgr_mask.shape[:2], dtype=np.uint8)
    for class_idx, bgr in class_rgb_values_bgr.items():
        match = np.all(bgr_mask == bgr, axis=-1)
        class_indices[match] = class_idx
    return class_indices

def create_error_map_overlay(true_mask, predicted_mask, original_image, alpha=0.5):
    true_mask_array = np.array(true_mask)
    
    if len(true_mask_array.shape) == 3:  # RGB format (BGR order from OpenCV)
        print("True mask is RGB (BGR order)")
        unique_bgr = np.unique(true_mask_array.reshape(-1, true_mask_array.shape[-1]), axis=0)
        print("Unique BGR values in true mask:", unique_bgr)
        true_mask_indices = rgb_to_class_indices(true_mask_array)
        # Convert to RGB for display
        true_mask_rgb = np.zeros((*true_mask_indices.shape, 3), dtype=np.uint8)
        for class_idx, rgb in class_rgb_values_rgb.items():
            true_mask_rgb[true_mask_indices == class_idx] = rgb
    elif len(true_mask_array.shape) == 2:  # Grayscale/class indices
        print("True mask is grayscale/class indices")
        true_mask_indices = true_mask_array
        if np.max(true_mask_indices) > 4:
            print("Warning: True mask values exceed 4, assuming grayscale needs remapping")
            true_mask_indices = (true_mask_indices // 51).astype(np.uint8)
        true_mask_rgb = np.zeros((*true_mask_indices.shape, 3), dtype=np.uint8)
        for class_idx, rgb in class_rgb_values_rgb.items():
            true_mask_rgb[true_mask_indices == class_idx] = rgb
    
    print("True mask shape:", true_mask_indices.shape)
    print("Predicted mask shape:", predicted_mask.shape)
    print("True mask unique values (after conversion):", np.unique(true_mask_indices))
    print("Predicted mask unique values:", np.unique(predicted_mask))

    if true_mask_indices.shape != predicted_mask.shape:
        raise ValueError("Shape mismatch between true mask and predicted mask")

    error_map = np.zeros((*true_mask_indices.shape, 3), dtype=np.uint8)
    
    tp_mask = (true_mask_indices == predicted_mask) & (true_mask_indices > 0)
    error_map[tp_mask] = [0, 255, 0]  # TP: Green
    
    fn_mask = (true_mask_indices > 0) & (predicted_mask != true_mask_indices)
    error_map[fn_mask] = [255, 0, 0]  # FN: Red
    
    fp_mask = (predicted_mask > 0) & (predicted_mask != true_mask_indices)
    error_map[fp_mask] = [0, 0, 255]  # FP: Blue
    
    tn_mask = (true_mask_indices == 0) & (predicted_mask == 0)
    error_map[tn_mask] = [0, 0, 0]  # TN: Black
    
    print(f"TP pixels: {np.sum(tp_mask)}, FN pixels: {np.sum(fn_mask)}, "
          f"FP pixels: {np.sum(fp_mask)}, TN pixels: {np.sum(tn_mask)}")

    original_array = np.array(original_image)
    if len(original_array.shape) == 2:
        original_array = cv2.cvtColor(original_array, cv2.COLOR_GRAY2RGB)
    
    overlay = original_array.copy()
    mask_area = np.any(error_map != [0, 0, 0], axis=-1)
    overlay[mask_area] = (alpha * error_map[mask_area] + 
                         (1 - alpha) * original_array[mask_area]).astype(np.uint8)
    
    return Image.fromarray(overlay), Image.fromarray(true_mask_rgb)

def save_plot_in_vector_format(fig, save_path, format='eps'):
    """
    Save the plot in a vector format (e.g., EPS or SVG).
    
    Args:
        fig: The Matplotlib figure object to save.
        save_path: Path (including filename) where the plot will be saved.
        format: Format to save the plot ('eps' or 'svg'). Default is 'eps'.
    """
    if format not in ['eps', 'svg']:
        raise ValueError("Unsupported format. Use 'eps' or 'svg'.")
    
    # Save the figure in the specified vector format
    fig.savefig(save_path, format=format, bbox_inches='tight', dpi=300)
    print(f"Plot saved in {format.upper()} format at: {save_path}")


def visualize_four_plots(images_folder, masks_folder, image_size=(512, 512), alpha=0.5, save_path=None):
    """
    Visualize the original image, true mask, predicted mask, and error map overlay.
    Optionally save the plot in a vector format.
    
    Args:
        images_folder: Path to the folder containing input images.
        masks_folder: Path to the folder containing true masks.
        image_size: Size to which images are resized (default: (512, 512)).
        alpha: Transparency for the error map overlay (default: 0.5).
        save_path: Path to save the plot in vector format (optional).
    """
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not image_files:
        print("No images found in the images folder.")
        return

    random_image_file = random.choice(image_files)
    print(f"Selected random image: {random_image_file}")

    image_path = os.path.join(images_folder, random_image_file)
    mask_path = os.path.join(masks_folder, random_image_file)

    image = Image.open(image_path)
    true_mask = Image.open(mask_path)
    predicted_mask_img, predicted_mask_array = segment_image(image, image_size)
    overlay_image, true_mask_rgb = create_error_map_overlay(true_mask, predicted_mask_array, image, alpha)

    # Create the figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Original Image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # True Mask
    axes[1].imshow(true_mask_rgb)  # Display in RGB
    axes[1].set_title("True Mask")
    axes[1].axis('off')

    # Predicted Mask
    axes[2].imshow(predicted_mask_img)  # Display in RGB
    axes[2].set_title("Predicted Mask")
    axes[2].axis('off')

    # Error Map Overlay
    axes[3].imshow(overlay_image)
    axes[3].set_title(f"Error Map Overlay (alpha={alpha})")
    axes[3].axis('off')

    plt.tight_layout()

    # Save the plot in vector format if save_path is provided
    if save_path:
        save_plot_in_vector_format(fig, save_path, format='eps')  # Change format to 'svg' if needed

    plt.show()


# Example usage
images_folder = "H:/Research/Breast Cancer/multimodal_merged_set_1_2_3_BIRADS_rev/aug/181/aug/images"
masks_folder = "H:/Research/Breast Cancer/multimodal_merged_set_1_2_3_BIRADS_rev/aug/181/aug/masks"

# Define the path to save the plot in vector format
save_path = "H:/Research/Breast Cancer/multimodal_merged_set_1_2_3_BIRADS_rev/aug/cropped/512/Visualization/four_plot_visualization.eps"  # Change to 'four_plot_visualization.svg' for SVG format

# Visualize and save the plot
visualize_four_plots(images_folder, masks_folder, alpha=0.9, save_path=save_path)







#----------------- Model Interpretation GradCam Visualization (Original, Grad-cam Heatmap (specific class), Grad-cam Overlay) ---------------------------------# 

import random
import os
import numpy as np
import tensorflow as tf
import cv2
from keras.utils import normalize, load_img, img_to_array, array_to_img
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, target_class_idx=None):
    """
    Generate a Grad-CAM heatmap for the given image and model.
    
    Args:
        img_array: Preprocessed input image array
        model: The trained model
        last_conv_layer_name: Name of the last convolutional layer
        target_class_idx: Index of the target class (optional)
        
    Returns:
        heatmap: The Grad-CAM heatmap
    """
    # Ensure model.output is a single tensor
    if isinstance(model.output, list):
        segmentation_output = model.output[0]  # Use the first output tensor
    else:
        segmentation_output = model.output

    # Create a model that maps the input image to the activations of the last conv layer and the output predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, segmentation_output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        
        # Option 1: Use a specific pixel (e.g., center)
        center_pixel = preds[:, preds.shape[1] // 2, preds.shape[2] // 2, :]
        if target_class_idx is None:
            target_class_idx = tf.argmax(center_pixel[0])  # Use the top predicted class at the center pixel
        class_channel = center_pixel[:, target_class_idx]

        # Option 2: Aggregate over the entire image (uncomment below if needed)
        # class_channel = tf.reduce_mean(preds[:, :, :, target_class_idx])

    # Gradient of the output neuron (top predicted or chosen) with respect to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Vector where each entry is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array by "how important this channel is"
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap between 0 and 1 for visualization
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    """
    Save and display the Grad-CAM heatmap overlaid on the original image.
    
    Args:
        img_path: Path to the original image
        heatmap: The Grad-CAM heatmap
        cam_path: Path to save the superimposed image (default: "cam.jpg")
        alpha: Transparency of the heatmap overlay
    """
    # Load the original image
    img = load_img(img_path)
    img = img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = img_to_array(jet_heatmap)

    # Superimpose the heatmap on the original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad-CAM
    plt.figure(figsize=(10, 10))
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.title("Grad-CAM Overlay")
    plt.show()



def visualize_gradcam(image_path, model, last_conv_layer_name, target_class_idx=4):
    """
    Visualize the Grad-CAM heatmap overlay on the original image.
    
    Args:
        image_path: Path to the image file
        model: The trained model
        last_conv_layer_name: Name of the last convolutional layer
        target_class_idx: Index of the target class (default: 4 for Mass)
    """
    # Load and preprocess the image
    img_size = (512, 512)  # Adjust size to match your model's input
    img_array = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img_array)
    img_array = normalize(np.array([img_array], dtype=np.float32), axis=1)  # Normalize input

    # Get model predictions
    preds = model.predict(img_array)
    print(f"Predicted probabilities: {preds}")
    predicted_class = np.argmax(preds[0])
    print(f"Predicted class: {predicted_class}")

    # Generate the heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, target_class_idx=target_class_idx)

    # Create the Grad-CAM overlay
    def create_gradcam_overlay(img_path, heatmap, alpha=0.4):
        """
        Create a Grad-CAM overlay for visualization.
        
        Args:
            img_path: Path to the original image
            heatmap: The Grad-CAM heatmap
            alpha: Transparency of the heatmap overlay
            
        Returns:
            superimposed_img: The Grad-CAM overlay as an array
        """
        # Load the original image
        img = load_img(img_path)
        img = img_to_array(img)

        # Rescale heatmap to a range 0-255
        heatmap_rescaled = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = mpl.colormaps["jet"]
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap_rescaled]

        # Create an image with RGB colorized heatmap
        jet_heatmap = array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = img_to_array(jet_heatmap)

        # Superimpose the heatmap on the original image
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = np.uint8(superimposed_img)
        return superimposed_img

    # Generate the Grad-CAM overlay
    gradcam_overlay = create_gradcam_overlay(image_path, heatmap)

    # Plot the original image, heatmap, and Grad-CAM overlay in subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original Image
    axes[0].imshow(load_img(image_path))
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Grad-CAM Heatmap
    axes[1].matshow(heatmap, cmap='jet')
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis('off')

    # Grad-CAM Overlay
    axes[2].imshow(gradcam_overlay)
    axes[2].set_title("Grad-CAM Overlay")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


# Example usage
last_conv_layer_name = "conv2d_19"  # Explicitly use conv2d_19
print(f"Using layer {last_conv_layer_name} for GradCAM")

# Get a random image from your dataset
images_folder = "H:/Research/Breast Cancer/multimodal_merged_set_1_2_3_BIRADS_rev/aug/154/images"
image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

if image_files:
    random_image_file = random.choice(image_files)
    image_path = os.path.join(images_folder, random_image_file)
    print(f"Selected random image: {random_image_file}")
    
    # Visualize Grad-CAM
    visualize_gradcam(image_path, model, last_conv_layer_name, target_class_idx=4)
else:
    print("No images found in the folder.")    






#----------------- Model Interpretation GradCam Visualization of different convo layer activations (Original, conv1, conv2, conv3, Grad-cam Overlay) ---------------------------------# 

import random
import os
import numpy as np
import tensorflow as tf
import cv2
from keras.utils import normalize, load_img, img_to_array, array_to_img
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, target_class_idx=None):
    """
    Generate a Grad-CAM heatmap for the given image and model.
    
    Args:
        img_array: Preprocessed input image array
        model: The trained model
        last_conv_layer_name: Name of the last convolutional layer
        target_class_idx: Index of the target class (optional)
        
    Returns:
        heatmap: The Grad-CAM heatmap
    """
    # Ensure model.output is a single tensor
    if isinstance(model.output, list):
        segmentation_output = model.output[0]  # Use the first output tensor
    else:
        segmentation_output = model.output

    # Create a model that maps the input image to the activations of the last conv layer and the output predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, segmentation_output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        
        # Option 1: Use a specific pixel (e.g., center)
        center_pixel = preds[:, preds.shape[1] // 2, preds.shape[2] // 2, :]
        if target_class_idx is None:
            target_class_idx = tf.argmax(center_pixel[0])  # Use the top predicted class at the center pixel
        class_channel = center_pixel[:, target_class_idx]

        # Option 2: Aggregate over the entire image (uncomment below if needed)
        # class_channel = tf.reduce_mean(preds[:, :, :, target_class_idx])

    # Gradient of the output neuron (top predicted or chosen) with respect to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Vector where each entry is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array by "how important this channel is"
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap between 0 and 1 for visualization
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    """
    Save and display the Grad-CAM heatmap overlaid on the original image.
    
    Args:
        img_path: Path to the original image
        heatmap: The Grad-CAM heatmap
        cam_path: Path to save the superimposed image (default: "cam.jpg")
        alpha: Transparency of the heatmap overlay
    """
    # Load the original image
    img = load_img(img_path)
    img = img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = img_to_array(jet_heatmap)

    # Superimpose the heatmap on the original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad-CAM
    plt.figure(figsize=(10, 10))
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.title("Grad-CAM Overlay")
    plt.show()



def save_plot_in_vector_format(fig, save_path, format='eps'):
    """
    Save the plot in a vector format (e.g., EPS or SVG).
    
    Args:
        fig: The Matplotlib figure object to save.
        save_path: Path (including filename) where the plot will be saved.
        format: Format to save the plot ('eps' or 'svg'). Default is 'eps'.
    """
    if format not in ['eps', 'svg']:
        raise ValueError("Unsupported format. Use 'eps' or 'svg'.")
    
    # Save the figure in the specified vector format
    fig.savefig(save_path, format=format, bbox_inches='tight', dpi=300)
    print(f"Plot saved in {format.upper()} format at: {save_path}")


def visualize_gradcam_with_multiple_layers(image_path, model, layer_names, target_class_idx=4, save_path=None):
    """
    Visualize the Grad-CAM heatmap overlay on the original image with heatmaps from multiple layers.
    Optionally save the plot in a vector format.
    
    Args:
        image_path: Path to the image file
        model: The trained model
        layer_names: List of names of the convolutional layers to visualize
        target_class_idx: Index of the target class (default: 4 for Mass)
        save_path: Path to save the plot in vector format (optional)
    """
    # Load and preprocess the image
    img_size = (512, 512)  # Adjust size to match your model's input
    img_array = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img_array)
    img_array = normalize(np.array([img_array], dtype=np.float32), axis=1)  # Normalize input

    # Get model predictions
    preds = model.predict(img_array)
    print(f"Predicted probabilities: {preds}")
    predicted_class = np.argmax(preds[0])
    print(f"Predicted class: {predicted_class}")

    # Generate heatmaps for each layer
    heatmaps = []
    for layer_name in layer_names:
        heatmap = make_gradcam_heatmap(img_array, model, layer_name, target_class_idx=target_class_idx)
        heatmaps.append(heatmap)

    # Create the Grad-CAM overlay
    def create_gradcam_overlay(img_path, heatmap, alpha=0.4):
        """
        Create a Grad-CAM overlay for visualization.
        
        Args:
            img_path: Path to the original image
            heatmap: The Grad-CAM heatmap
            alpha: Transparency of the heatmap overlay
            
        Returns:
            superimposed_img: The Grad-CAM overlay as an array
        """
        # Load the original image
        img = load_img(img_path)
        img = img_to_array(img)

        # Rescale heatmap to a range 0-255
        heatmap_rescaled = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = mpl.colormaps["jet"]
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap_rescaled]

        # Create an image with RGB colorized heatmap
        jet_heatmap = array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = img_to_array(jet_heatmap)

        # Superimpose the heatmap on the original image
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = np.uint8(superimposed_img)
        return superimposed_img

    # Generate the Grad-CAM overlay for the last layer (conv2d_19)
    gradcam_overlay = create_gradcam_overlay(image_path, heatmaps[-1])

    # Plot the original image, heatmaps, and Grad-CAM overlay in subplots
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    # Original Image
    axes[0].imshow(load_img(image_path))
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Heatmap from conv2d_5
    axes[1].matshow(heatmaps[0], cmap='jet')
    axes[1].set_title("Heatmap (conv2d_5)")
    axes[1].axis('off')

    # Heatmap from conv2d_11
    axes[2].matshow(heatmaps[1], cmap='jet')
    axes[2].set_title("Heatmap (conv2d_11)")
    axes[2].axis('off')

    # Heatmap from conv2d_19
    axes[3].matshow(heatmaps[2], cmap='jet')
    axes[3].set_title("Heatmap (conv2d_19)")
    axes[3].axis('off')

    # Grad-CAM Overlay (using conv2d_19)
    axes[4].imshow(gradcam_overlay)
    axes[4].set_title("Grad-CAM Overlay")
    axes[4].axis('off')

    plt.tight_layout()

    # Save the plot in vector format if save_path is provided
    if save_path:
        save_plot_in_vector_format(fig, save_path, format='eps')  # Change format to 'svg' if needed

    plt.show()


# Example usage
layer_names = ["conv2d_5", "conv2d_11", "conv2d_19"]  # Layers to visualize
print(f"Using layers {layer_names} for GradCAM")

# Get a random image from your dataset
images_folder = "H:/Research/Breast Cancer/multimodal_merged_set_1_2_3_BIRADS_rev/aug/cropped/512/images"
image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

if image_files:
    random_image_file = random.choice(image_files)
    image_path = os.path.join(images_folder, random_image_file)
    print(f"Selected random image: {random_image_file}")
    
    # Define the path to save the plot in vector format
    save_path = "H:/Research/Breast Cancer/multimodal_merged_set_1_2_3_BIRADS_rev/aug/cropped/512/Visualization/heatmap_plot.eps"  # Change to 'output_plot.svg' for SVG format
    
    # Visualize Grad-CAM with multiple layers and save the plot
    visualize_gradcam_with_multiple_layers(image_path, model, layer_names, target_class_idx=4, save_path=save_path)
else:
    print("No images found in the folder.")






#----------------- Model Interpretation GradCam Visualization of different convo layer activations (Original, conv1, conv2, conv3, predicted mask) ---------------------------------# 

import random
import os
import numpy as np
import tensorflow as tf
import cv2
from keras.utils import normalize, load_img, img_to_array, array_to_img
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, target_class_idx=None):
    """
    Generate a Grad-CAM heatmap for the given image and model.
    
    Args:
        img_array: Preprocessed input image array
        model: The trained model
        last_conv_layer_name: Name of the last convolutional layer
        target_class_idx: Index of the target class (optional)
        
    Returns:
        heatmap: The Grad-CAM heatmap
    """
    # Ensure model.output is a single tensor
    if isinstance(model.output, list):
        segmentation_output = model.output[0]  # Use the first output tensor
    else:
        segmentation_output = model.output

    # Create a model that maps the input image to the activations of the last conv layer and the output predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, segmentation_output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        
        # Option 1: Use a specific pixel (e.g., center)
        center_pixel = preds[:, preds.shape[1] // 2, preds.shape[2] // 2, :]
        if target_class_idx is None:
            target_class_idx = tf.argmax(center_pixel[0])  # Use the top predicted class at the center pixel
        class_channel = center_pixel[:, target_class_idx]

        # Option 2: Aggregate over the entire image (uncomment below if needed)
        # class_channel = tf.reduce_mean(preds[:, :, :, target_class_idx])

    # Gradient of the output neuron (top predicted or chosen) with respect to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Vector where each entry is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array by "how important this channel is"
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap between 0 and 1 for visualization
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    """
    Save and display the Grad-CAM heatmap overlaid on the original image.
    
    Args:
        img_path: Path to the original image
        heatmap: The Grad-CAM heatmap
        cam_path: Path to save the superimposed image (default: "cam.jpg")
        alpha: Transparency of the heatmap overlay
    """
    # Load the original image
    img = load_img(img_path)
    img = img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = img_to_array(jet_heatmap)

    # Superimpose the heatmap on the original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad-CAM
    plt.figure(figsize=(10, 10))
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.title("Grad-CAM Overlay")
    plt.show()



def save_plot_in_vector_format(fig, save_path, format='eps'):
    """
    Save the plot in a vector format (e.g., EPS or SVG).
    
    Args:
        fig: The Matplotlib figure object to save.
        save_path: Path (including filename) where the plot will be saved.
        format: Format to save the plot ('eps' or 'svg'). Default is 'eps'.
    """
    if format not in ['eps', 'svg']:
        raise ValueError("Unsupported format. Use 'eps' or 'svg'.")
    
    # Save the figure in the specified vector format
    fig.savefig(save_path, format=format, bbox_inches='tight', dpi=300)
    print(f"Plot saved in {format.upper()} format at: {save_path}")


def visualize_gradcam_with_multiple_layers_and_mask(image_path, model, layer_names, target_class_idx=4, save_path=None):
    """
    Visualize the Grad-CAM heatmap overlay on the original image with heatmaps from multiple layers
    and the predicted mask. Optionally save the plot in a vector format.
    
    Args:
        image_path: Path to the image file
        model: The trained model
        layer_names: List of names of the convolutional layers to visualize
        target_class_idx: Index of the target class (default: 4 for Mass)
        save_path: Path to save the plot in vector format (optional)
    """
    # Load and preprocess the image
    img_size = (512, 512)  # Adjust size to match your model's input
    img_array = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img_array)
    img_array = normalize(np.array([img_array], dtype=np.float32), axis=1)  # Normalize input

    # Get model predictions
    preds = model.predict(img_array)
    print(f"Predicted probabilities: {preds}")
    predicted_class = np.argmax(preds[0])
    print(f"Predicted class: {predicted_class}")

    # Generate heatmaps for each layer
    heatmaps = []
    for layer_name in layer_names:
        heatmap = make_gradcam_heatmap(img_array, model, layer_name, target_class_idx=target_class_idx)
        heatmaps.append(heatmap)

    # Generate the predicted mask
    predicted_mask = np.argmax(preds[0], axis=-1)  # Assuming preds is of shape (1, H, W, C)
    predicted_mask = np.squeeze(predicted_mask)  # Remove batch dimension if present

    # Plot the original image, heatmaps, and predicted mask in subplots
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    # Original Image
    axes[0].imshow(load_img(image_path))
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Heatmap from conv2d_5
    axes[1].matshow(heatmaps[0], cmap='jet')
    axes[1].set_title("Heatmap (conv2d_5)")
    axes[1].axis('off')

    # Heatmap from conv2d_11
    axes[2].matshow(heatmaps[1], cmap='jet')
    axes[2].set_title("Heatmap (conv2d_11)")
    axes[2].axis('off')

    # Heatmap from conv2d_19
    axes[3].matshow(heatmaps[2], cmap='jet')
    axes[3].set_title("Heatmap (conv2d_19)")
    axes[3].axis('off')

    # Predicted Mask
    axes[4].imshow(predicted_mask, cmap='viridis')  # Use a colormap like 'viridis' for better visualization
    axes[4].set_title("Predicted Mask")
    axes[4].axis('off')

    plt.tight_layout()

    # Save the plot in vector format if save_path is provided
    if save_path:
        save_plot_in_vector_format(fig, save_path, format='eps')  # Change format to 'svg' if needed

    plt.show()


# Example usage
layer_names = ["conv2d_5", "conv2d_11", "conv2d_19"]  # Layers to visualize
print(f"Using layers {layer_names} for GradCAM")

# Get a random image from your dataset
images_folder = "H:/Research/Breast Cancer/multimodal_merged_set_1_2_3_BIRADS_rev/aug/cropped/512/images"
image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

if image_files:
    random_image_file = random.choice(image_files)
    image_path = os.path.join(images_folder, random_image_file)
    print(f"Selected random image: {random_image_file}")
    
    # Define the path to save the plot in vector format
    save_path = "H:/Research/Breast Cancer/multimodal_merged_set_1_2_3_BIRADS_rev/aug/cropped/512/Visualization/output_plot.eps"  # Change to 'output_plot.svg' for SVG format
    
    # Visualize Grad-CAM with multiple layers and predicted mask
    visualize_gradcam_with_multiple_layers_and_mask(image_path, model, layer_names, target_class_idx=4, save_path=save_path)
else:
    print("No images found in the folder.")




#----------------- Model Interpretation GradCam Visualization of different convo layer activations of different classes (Original, bottleneck 1, 2, 3, 4) ---------------------------------# 

# ========== Heatmap of different classes from bottleeneck layer 

import random
import os
import numpy as np
import tensorflow as tf
import cv2
from keras.utils import normalize, load_img, img_to_array, array_to_img
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, target_class_idx):
    """
    Generate a Grad-CAM heatmap for the given image and model.
    
    Args:
        img_array: Preprocessed input image array
        model: The trained model
        last_conv_layer_name: Name of the last convolutional layer
        target_class_idx: Index of the target class
        
    Returns:
        heatmap: The Grad-CAM heatmap
    """
    # Ensure model.output is a single tensor
    if isinstance(model.output, list):
        segmentation_output = model.output[0]  # Use the first output tensor
    else:
        segmentation_output = model.output

    # Create a model that maps the input image to the activations of the last conv layer and the output predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, segmentation_output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        
        # Use the output corresponding to the target class
        class_channel = preds[:, :, :, target_class_idx]

    # Gradient of the output neuron (target class) with respect to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Vector where each entry is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array by "how important this channel is"
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap between 0 and 1 for visualization
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    """
    Save and display the Grad-CAM heatmap overlaid on the original image.
    
    Args:
        img_path: Path to the original image
        heatmap: The Grad-CAM heatmap
        cam_path: Path to save the superimposed image (default: "cam.jpg")
        alpha: Transparency of the heatmap overlay
    """
    # Load the original image
    img = load_img(img_path)
    img = img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = img_to_array(jet_heatmap)

    # Superimpose the heatmap on the original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad-CAM
    plt.figure(figsize=(10, 10))
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.title("Grad-CAM Overlay")
    plt.show()



def save_plot_in_vector_format(fig, save_path, format='eps'):
    """
    Save the plot in a vector format (e.g., EPS or SVG).
    
    Args:
        fig: The Matplotlib figure object to save.
        save_path: Path (including filename) where the plot will be saved.
        format: Format to save the plot ('eps' or 'svg'). Default is 'eps'.
    """
    if format not in ['eps', 'svg']:
        raise ValueError("Unsupported format. Use 'eps' or 'svg'.")
    
    # Save the figure in the specified vector format
    fig.savefig(save_path, format=format, bbox_inches='tight', dpi=300)
    print(f"Plot saved in {format.upper()} format at: {save_path}")


def visualize_gradcam_for_multiple_classes(image_path, model, layer_name, target_class_indices, save_path=None):
    """
    Visualize Grad-CAM heatmaps for multiple target classes from a specific layer.
    Optionally save the plot in a vector format.
    
    Args:
        image_path: Path to the image file
        model: The trained model
        layer_name: Name of the convolutional layer to visualize (e.g., "conv2d_11")
        target_class_indices: List of target class indices to visualize (e.g., [1, 2, 3, 4])
        save_path: Path to save the plot in vector format (optional)
    """
    # Load and preprocess the image
    img_size = (512, 512)  # Adjust size to match your model's input
    img_array = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img_array)
    img_array = normalize(np.array([img_array], dtype=np.float32), axis=1)  # Normalize input

    # Get model predictions
    preds = model.predict(img_array)
    print(f"Predicted probabilities: {preds}")
    predicted_class = np.argmax(preds[0])
    print(f"Predicted class: {predicted_class}")

    # Generate heatmaps for each target class
    heatmaps = []
    for target_class_idx in target_class_indices:
        heatmap = make_gradcam_heatmap(img_array, model, layer_name, target_class_idx=target_class_idx)
        heatmaps.append(heatmap)

    # Plot the original image and heatmaps for each target class
    fig, axes = plt.subplots(1, len(target_class_indices) + 1, figsize=(20, 5))

    # Original Image
    axes[0].imshow(load_img(image_path))
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Heatmaps for each target class
    for i, (heatmap, target_class_idx) in enumerate(zip(heatmaps, target_class_indices)):
        axes[i + 1].matshow(heatmap, cmap='jet')
        axes[i + 1].set_title(f"Heatmap (Class {target_class_idx})")
        axes[i + 1].axis('off')

    plt.tight_layout()

    # Save the plot in vector format if save_path is provided
    if save_path:
        save_plot_in_vector_format(fig, save_path, format='eps')  # Change format to 'svg' if needed

    plt.show()

# Example usage
layer_name = "conv2d_12"  # Layer to visualize
target_class_indices = [1, 2, 3, 4]  # Target classes to visualize
print(f"Using layer {layer_name} for GradCAM with target classes {target_class_indices}")

# Get a random image from your dataset
images_folder = "H:/Research/Breast Cancer/multimodal_merged_set_1_2_3_BIRADS_rev/aug/cropped/512/images"
image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

if image_files:
    random_image_file = random.choice(image_files)
    image_path = os.path.join(images_folder, random_image_file)
    print(f"Selected random image: {random_image_file}")
    
    # Define the path to save the plot in vector format
    save_path = "H:/Research/Breast Cancer/multimodal_merged_set_1_2_3_BIRADS_rev/aug/cropped/512/Visualization/heatmap_plot_multiple_classes.eps"  # Change to 'output_plot.svg' for SVG format
    
    # Visualize Grad-CAM for multiple target classes and save the plot
    visualize_gradcam_for_multiple_classes(image_path, model, layer_name, target_class_indices, save_path=save_path)
else:
    print("No images found in the folder.")














#============================ Model Prediction (segmenation and tabular at once) ==================================================== 

#===== saving the seg_scaler ==============================

import os
import numpy as np
from keras.models import load_model, Model
from keras.utils import normalize
import cv2
from PIL import Image

# Load the segmentation model (assuming it's already defined)
model_path = "H:/Research/Breast Cancer/multimodal_merged_set_1_2_3_BIRADS_rev/aug/cropped/256/random_state_0/best_model1_t1_120_att_Epochs (1).keras"
custom_objects = {
    'combined_loss': combined_loss,
    'focal_loss': focal_loss,
    'soft_dice_loss': soft_dice_loss,
    'soft_dice_coefficient': lambda y_true, y_pred: 1 - soft_dice_loss(y_true, y_pred),
    'CustomMeanIoU': CustomMeanIoU
}
segmentation_model = load_model(model_path, custom_objects=custom_objects, compile=True)

# Extract the feature extractor (assuming 'conv2d_11' is the bottleneck layer with 512 filters)
feature_extractor = Model(inputs=segmentation_model.input, outputs=segmentation_model.get_layer('conv2d_11').output)

def create_reference_dataset(images_folder, num_samples=4831, image_size=(256, 256)):
    """
    Create a reference dataset from the images folder for scaling.
    
    Args:
        images_folder: Folder containing image files
        num_samples: Number of samples to use
        image_size: Size to resize images to
        
    Returns:
        Array of normalized images for reference
    """
    print(f"Creating reference dataset from {images_folder}...")
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_files:
        print("No images found in the folder.")
        return []
    
    # Limit to specified number of samples
    selected_files = image_files[:num_samples] if len(image_files) > num_samples else image_files
    print(f"Using {len(selected_files)} reference images.")
    
    sample_images = []
    for file in selected_files:
        img_path = os.path.join(images_folder, file)
        try:
            img = np.array(Image.open(img_path))
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_resized = cv2.resize(img, image_size)
            img_normalized = normalize(np.array([img_resized], dtype=np.float32), axis=1)[0]
            sample_images.append(img_normalized)
            print(f"Processed {file} for reference dataset.")
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    return np.array(sample_images) if sample_images else None

def save_seg_scaler(reference_data, save_path="H:/Research/Breast Cancer/multimodal_merged_set_1_2_3_BIRADS_rev/aug/cropped/256/random_state_0/seg_scaler_updated_4831_256.npz"):
    """
    Compute and save the mean and std of the features from the reference data.
    
    Args:
        reference_data: Array of normalized images for reference
        save_path: Path to save the scaler data
    """
    # Extract features from the reference data
    sample_features = []
    for img in reference_data:
        img_expanded = np.expand_dims(img, axis=0)
        feat = feature_extractor.predict(img_expanded)
        feat = feat.reshape(1, -1, 512).mean(axis=1)  # Global Average Pooling
        sample_features.append(feat)
    
    sample_features = np.vstack(sample_features)
    feature_mean = np.mean(sample_features, axis=0)
    feature_std = np.std(sample_features, axis=0) + 1e-10  # Add small epsilon to avoid division by zero
    
    # Save the mean and std to a file
    np.savez(save_path, mean=feature_mean, std=feature_std)
    print(f"Segmentation scaler saved to {save_path}")

# Example usage
images_folder = "H:/Research/Breast Cancer/multimodal_merged_set_1_2_3_BIRADS_rev/aug/cropped/256/random_state_0/train/images"
reference_data = create_reference_dataset(images_folder, num_samples=4831)
save_seg_scaler(reference_data, save_path="H:/Research/Breast Cancer/multimodal_merged_set_1_2_3_BIRADS_rev/aug/cropped/256/random_state_0/seg_scaler_updated_4831_256.npz")




# Multimodal Prediction 

import os
import cv2
import numpy as np
from keras.models import load_model, Model
from keras.utils import normalize
import tensorflow as tf
from keras.saving import register_keras_serializable
from PIL import Image
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

# Define custom loss and metric functions (from your segmentation code)
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

# Load the segmentation model
model_path = "H:/Research/Breast Cancer/multimodal_merged_set_1_2_3_BIRADS_rev/aug/cropped/256/random_state_0/best_model1_t1_120_att_Epochs (1).keras"
custom_objects = {
    'combined_loss': combined_loss,
    'focal_loss': focal_loss,
    'soft_dice_loss': soft_dice_loss,
    'soft_dice_coefficient': lambda y_true, y_pred: 1 - soft_dice_loss(y_true, y_pred),
    'CustomMeanIoU': CustomMeanIoU
}
segmentation_model = load_model(model_path, custom_objects=custom_objects, compile=True)

# Extract the feature extractor (assuming 'conv2d_11' is the bottleneck layer with 512 filters)
feature_extractor = Model(inputs=segmentation_model.input, outputs=segmentation_model.get_layer('conv2d_11').output)

# Load the tabular model
tabular_model_path = "H:/Research/Breast Cancer/multimodal_merged_set_1_2_3_BIRADS_rev/aug/cropped/256/random_state_0/tabular_prediction_model_raw_att_120.h5"
tabular_model = load_model(tabular_model_path, compile=True)

# Define target columns and output dimensions
target_columns = ['mass', 'mass_definition', 'mass_density', 'mass_shape', 'mass_calcification', 
                  'axilla_findings', 'calcification', 'calcification_type', 'acr_breast_density', 'BIRADS_CAT']
output_dims = {
    'mass': 2, 'mass_definition': 4, 'mass_density': 4, 'mass_shape': 4, 'mass_calcification': 3,
    'axilla_findings': 2, 'calcification': 2, 'calcification_type': 4, 'acr_breast_density': 4, 'BIRADS_CAT': 6
}

# Recreate label encoders
label_encoders = {}
unique_values = {
    'mass': [0, 1],
    'mass_definition': [0, 1, 2, 3],
    'mass_density': [0, 1, 2, 3],
    'mass_shape': [0, 1, 2, 3],
    'mass_calcification': [0, 1, 3],
    'axilla_findings': [0, 1],
    'calcification': [0, 1],
    'calcification_type': [0, 1, 2, 3],
    'acr_breast_density': [1, 2, 3, 4],
    'BIRADS_CAT': [1, 2, 3, 4, 5, 6]
}

for col in target_columns:
    le = LabelEncoder()
    le.fit(unique_values[col])
    label_encoders[col] = le

# Class RGB values for visualization
class_rgb_values = {
    0: [0, 0, 0],        # Background
    1: [153, 178, 199],  # Calcification
    2: [170, 110, 240],  # Axilla_Findings
    3: [216, 155, 8],    # Tissue
    4: [58, 237, 28]     # Mass
}

def extract_and_scale_features(image_normalized, reference_data=None):
    """
    Extract features from an image and scale them appropriately for tabular prediction.
    
    Args:
        image_normalized: Normalized image input (shape: [1, height, width, channels])
        reference_data: Training data to use for scaling reference (optional)
        
    Returns:
        Scaled features ready for tabular model prediction
    """
    # Extract features using the feature extractor
    features = feature_extractor.predict(image_normalized)  # Shape: (1, height, width, 512)
    features = features.reshape(1, -1, 512).mean(axis=1)  # Global Average Pooling
    
    # If we have training data available, use it to calibrate scaling
    if reference_data is not None and len(reference_data) > 0:
        # Extract features from a sample of training images
        sample_size = min(20, len(reference_data))  # Use up to 20 training samples
        sample_features = []
        
        for i in range(sample_size):
            sample_img = np.expand_dims(reference_data[i], axis=0)
            feat = feature_extractor.predict(sample_img)
            feat = feat.reshape(1, -1, 512).mean(axis=1)
            sample_features.append(feat)
            
        sample_features = np.vstack(sample_features)
        feature_mean = np.mean(sample_features, axis=0)
        feature_std = np.std(sample_features, axis=0) + 1e-10  # Add small epsilon to avoid division by zero
        
        # Scale using training data statistics
        features = (features - feature_mean) / feature_std
    else:
        # Fallback: perform basic standardization (less accurate but better than nothing)
        features = (features - np.mean(features)) / (np.std(features) + 1e-10)
    
    return features

def process_and_predict(image_path, mask_path=None, image_size=(256, 256), reference_data=None):
    """
    Process an image and predict both segmentation mask and tabular features.
    
    Args:
        image_path: Path to the input image
        mask_path: Path to the ground truth mask image (optional)
        image_size: Size to resize image to before prediction
        reference_data: Reference data for feature scaling
        
    Returns:
        Image, ground truth mask (if provided), predicted mask, and tabular predictions
    """
    # Load and preprocess the image
    image = Image.open(image_path)
    img = np.array(image)
    
    # Convert if not in BGR (OpenCV format)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    original_shape = img.shape[:2]
    img_resized = cv2.resize(img, image_size)
    img_normalized = normalize(np.array([img_resized], dtype=np.float32), axis=1)

    # Load ground truth mask if provided
    true_mask = None
    if mask_path and os.path.exists(mask_path):
        true_mask = Image.open(mask_path)
    
    # Predict segmentation mask
    predicted_mask = segmentation_model.predict(img_normalized)
    predicted_mask_class = np.argmax(predicted_mask, axis=-1)[0]
    predicted_mask_resized = cv2.resize(predicted_mask_class, (original_shape[1], original_shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)

    # Create RGB mask for visualization
    rgb_mask = np.zeros((*predicted_mask_resized.shape, 3), dtype=np.uint8)
    for class_idx, rgb in class_rgb_values.items():
        rgb_mask[predicted_mask_resized == class_idx] = rgb
    rgb_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_BGR2RGB)

    # Extract and scale features using our new function
    segmentation_features = extract_and_scale_features(img_normalized, reference_data)

    # Predict tabular features
    predicted_tabular_features = tabular_model.predict(segmentation_features)

    return image, true_mask, rgb_mask, predicted_tabular_features

def visualize_and_print_results(image, true_mask, predicted_mask, predicted_tabular_features):
    """
    Visualize the results and print tabular predictions.
    
    Args:
        image: Original input image
        true_mask: Ground truth mask (if available, can be None)
        predicted_mask: Predicted segmentation mask
        predicted_tabular_features: Predicted tabular features dictionary
    """
    plt.figure(figsize=(15, 5))

    # Original Image
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')

    # True Mask (if available)
    plt.subplot(1, 3, 2)
    if true_mask is not None:
        plt.title("True Mask")
        plt.imshow(true_mask)
    else:
        plt.title("True Mask (Not Available)")
        plt.text(0.5, 0.5, "N/A", ha='center', va='center')
    plt.axis('off')

    # Predicted Mask
    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(predicted_mask)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Print tabular predictions with confidence scores
    print("\n----- Tabular Predictions -----")
    for col in target_columns:
        if output_dims[col] > 1:  # Skip if not included in prediction
            print(f"\nFeature: {col}")
            y_pred_col = predicted_tabular_features[col][0]
            
            if output_dims[col] == 2:  # Binary classification
                y_pred_encoded = 1 if y_pred_col > 0.5 else 0
                confidence = y_pred_col[0] if y_pred_encoded == 1 else 1 - y_pred_col[0]  # Extract scalar value
                y_pred_raw = label_encoders[col].inverse_transform([y_pred_encoded])[0]
                print(f"Predicted: {y_pred_raw} (Confidence: {confidence:.2f})")
            else:  # Multi-class classification
                y_pred_encoded = np.argmax(y_pred_col)
                confidence = y_pred_col[y_pred_encoded]  # Extract scalar value
                y_pred_raw = label_encoders[col].inverse_transform([y_pred_encoded])[0]
                print(f"Predicted: {y_pred_raw} (Confidence: {confidence:.2f})")

def create_reference_dataset(images_folder, num_samples=20, image_size=(256, 256)):
    """
    Create a reference dataset from the images folder for scaling.
    
    Args:
        images_folder: Folder containing image files
        num_samples: Number of samples to use
        image_size: Size to resize images to
        
    Returns:
        Array of normalized images for reference
    """
    print(f"Creating reference dataset from {images_folder}...")
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_files:
        print("No images found in the folder.")
        return []
    
    # Limit to specified number of samples
    selected_files = image_files[:num_samples] if len(image_files) > num_samples else image_files
    print(f"Using {len(selected_files)} reference images.")
    
    sample_images = []
    for file in selected_files:
        img_path = os.path.join(images_folder, file)
        try:
            img = np.array(Image.open(img_path))
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_resized = cv2.resize(img, image_size)
            img_normalized = normalize(np.array([img_resized], dtype=np.float32), axis=1)[0]
            sample_images.append(img_normalized)
            print(f"Processed {file} for reference dataset.")
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    return np.array(sample_images) if sample_images else None

def visualize_random_image_and_predictions(images_folder, masks_folder, image_size=(256, 256)):
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not image_files:
        print("No images found in the images folder.")
        return

    random_image_file = random.choice(image_files)
    print(f"Selected random image: {random_image_file}")

    image_path = os.path.join(images_folder, random_image_file)
    mask_path = os.path.join(masks_folder, random_image_file)

    image, true_mask, predicted_mask, predicted_tabular_features = process_and_predict(image_path, mask_path, image_size)
    visualize_and_print_results(image, true_mask, predicted_mask, predicted_tabular_features)

# Example usage
images_folder = "H:/Research/Breast Cancer/multimodal_merged_set_1_2_3_BIRADS_rev/aug/cropped/256/random_state_0/train/images"
masks_folder = "H:/Research/Breast Cancer/multimodal_merged_set_1_2_3_BIRADS_rev/aug/cropped/256/random_state_0/train/masks"
visualize_random_image_and_predictions(images_folder, masks_folder)


























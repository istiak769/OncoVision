#======================= Image data loading and Precrossing ====================================


from keras.utils import normalize
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# Resizing images, if needed
SIZE_X = 512 
SIZE_Y = 512
n_classes = 5  # Number of classes for segmentation

# Paths to train and test folders
train_images_folder = 'N:/istiak/Breast Cancer/Final Training/multimodal/multimodal_set_1/Test/BIRADS_revised/train/images/*png'
train_masks_folder = 'N:/istiak/Breast Cancer/Final Training/multimodal/multimodal_set_1/Test/BIRADS_revised/train/masks/*png'
test_images_folder = 'N:/istiak/Breast Cancer/Final Training/multimodal/multimodal_set_1/Test/BIRADS_revised/test/images/*png'
test_masks_folder = 'N:/istiak/Breast Cancer/Final Training/multimodal/multimodal_set_1/Test/BIRADS_revised/test/masks/*png'

# Function to load and preprocess images/masks
def load_data(image_paths, mask_paths, size_x, size_y):
    images = []
    masks = []
    
    # Load and preprocess images
    for img_path in image_paths:
        img = cv2.imread(img_path, 0)  # Read as grayscale
        img = cv2.resize(img, (size_x, size_y))
        images.append(img)
    
    # Load and preprocess masks
    for mask_path in mask_paths:
        mask = cv2.imread(mask_path, 0)  # Read as grayscale
        mask = cv2.resize(mask, (size_x, size_y), interpolation=cv2.INTER_NEAREST)  # Nearest-neighbor interpolation
        masks.append(mask)
    
    # Convert lists to arrays
    images = np.array(images)
    masks = np.array(masks)
    
    return images, masks

# Get sorted file paths for train and test datasets
train_image_paths = sorted(glob.glob(train_images_folder))
train_mask_paths = sorted(glob.glob(train_masks_folder))
test_image_paths = sorted(glob.glob(test_images_folder))
test_mask_paths = sorted(glob.glob(test_masks_folder))

# Load train and test data
X_train, y_train = load_data(train_image_paths, train_mask_paths, SIZE_X, SIZE_Y)
X_test, y_test = load_data(test_image_paths, test_mask_paths, SIZE_X, SIZE_Y)


# Print shapes of raw data
print("Raw shapes:")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# Visualize a single random sample before label encoding
def visualize_random_sample(images, masks):
    # Randomly select one index
    idx = np.random.randint(0, len(images))
    
    # Create a figure with two subplots (image and mask)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Display the image
    axes[0].imshow(images[idx], cmap='gray')
    axes[0].set_title(f"Image {idx}")
    axes[0].axis('off')
    
    # Display the mask
    axes[1].imshow(masks[idx], cmap='cividis')  # Use 'jet' colormap to distinguish mask classes
    axes[1].set_title(f"Mask {idx}")
    axes[1].axis('off')
    
    # Show the plot
    plt.show()

# Call the function to visualize a random training sample
print("\nVisualizing a random training sample...")
visualize_random_sample(X_train, y_train)

# Call the function to visualize a random testing sample
print("\nVisualizing a random testing sample...")
visualize_random_sample(X_test, y_test)









# Encode labels for masks
def encode_labels(masks, n_classes):
    labelencoder = LabelEncoder()
    n, h, w = masks.shape
    masks_reshaped = masks.reshape(-1, 1)
    masks_reshaped_encoded = labelencoder.fit_transform(masks_reshaped)
    masks_encoded_original_shape = masks_reshaped_encoded.reshape(n, h, w)
    return masks_encoded_original_shape

# Encode train and test masks
y_train_encoded = encode_labels(y_train, n_classes)
y_test_encoded = encode_labels(y_test, n_classes)

# Print shapes after encoding
print("\nShapes after label encoding:")
print("y_train_encoded shape:", y_train_encoded.shape)
print("y_test_encoded shape:", y_test_encoded.shape)

# Expand dimensions for images (add channel axis)
X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)

# Normalize images
X_train = normalize(X_train, axis=1)
X_test = normalize(X_test, axis=1)

# Add channel axis for masks
y_train_input = np.expand_dims(y_train_encoded, axis=3)
y_test_input = np.expand_dims(y_test_encoded, axis=3)

# Print shapes after adding channel axis
print("\nShapes after adding channel axis:")
print("X_train shape:", X_train.shape)
print("y_train_input shape:", y_train_input.shape)
print("X_test shape:", X_test.shape)
print("y_test_input shape:", y_test_input.shape)

# Convert masks to categorical format
train_masks_cat = to_categorical(y_train_input, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train_input.shape[0], y_train_input.shape[1], y_train_input.shape[2], n_classes))

test_masks_cat = to_categorical(y_test_input, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test_input.shape[0], y_test_input.shape[1], y_test_input.shape[2], n_classes))

# Print shapes after converting to categorical format
print("\nShapes after converting to categorical format:")
print("y_train_cat shape:", y_train_cat.shape)
print("y_test_cat shape:", y_test_cat.shape)

# Print unique class values in the dataset
print("\nClass values in the training dataset are ... ", np.unique(y_train_encoded))  # 0 is the background/few unlabeled
print("Class values in the testing dataset are ... ", np.unique(y_test_encoded))  # 0 is the background/few unlabeled


# Visualize random samples after label encoding
def visualize_random_sample_after_encoding(images, masks, num_samples=1):
    indices = np.random.choice(len(images), num_samples, replace=False)
    for idx in indices:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Display the image
        axes[0].imshow(images[idx].squeeze(), cmap='gray')  # Squeeze removes the channel dimension for grayscale images
        axes[0].set_title(f"Image {idx}")
        axes[0].axis('off')
        
        # Display the encoded mask
        axes[1].imshow(masks[idx].squeeze(), cmap='cividis')  # Use 'jet' colormap to distinguish classes
        axes[1].set_title(f"Encoded Mask {idx}")
        axes[1].axis('off')
        
        plt.show()

# Call the function to visualize a random training sample after encoding
print("\nVisualizing a random training sample after encoding...")
visualize_random_sample_after_encoding(X_train, y_train_encoded)

# Call the function to visualize a random testing sample after encoding
print("\nVisualizing a random testing sample after encoding...")
visualize_random_sample_after_encoding(X_test, y_test_encoded)


IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3] 





#======== Tabular Data Loading and Preprocessing ==========================

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ======================== Step 1: Load and Preprocess Data ========================
# Define paths to the train and test Excel files
train_file_path = 'H:/Research/Breast Cancer/multimodal_merged_set_1_2_3_BIRADS_rev/aug/cropped/256/Test/train/report/multimodal_merged_aug_set_1_2_3_BIRADS_revised_train.xlsx'
test_file_path = 'H:/Research/Breast Cancer/multimodal_merged_set_1_2_3_BIRADS_rev/aug/cropped/256/Test/test/report/multimodal_merged_aug_set_1_2_3_BIRADS_revised_test.xlsx'

# Load the train and test data directly
train_data = pd.read_excel(train_file_path)
test_data = pd.read_excel(test_file_path)

# Drop the image_id column
train_data = train_data.drop(columns=['image_id'])
test_data = test_data.drop(columns=['image_id'])

print(train_data.shape)
print(test_data.shape)
print(train_data.info())
print(test_data.info())

test_data.head()
# All columns are target variables
target_columns = train_data.columns.tolist()

# Determine the number of unique classes for each target variable
output_dims = {col: len(train_data[col].unique()) for col in target_columns}

print("Target Variables:", target_columns)
print("Output Dimensions:", output_dims)

# Separate input (X) and outputs (y)
X_train_tabular = train_data[target_columns].copy()
X_test_tabular = test_data[target_columns].copy()

# Create dictionaries for multi-output targets
y_train_tabular = {col: train_data[col] for col in target_columns}
y_test_tabular = {col: test_data[col] for col in target_columns}


# Verify the re-encoded labels
for col in target_columns:
    print(f"\nColumn: {col}")
    print("Re-encoded Unique Values (Train):", np.unique(y_train_tabular[col]))  # Use np.unique for NumPy arrays
    print("Re-encoded Unique Values (Test):", np.unique(y_test_tabular[col]))    # Use np.unique for NumPy arrays


# Find unseen labels in the test set
for col in target_columns:
    train_unique = set(train_data[col].unique())  # Unique values in the training set
    test_unique = set(test_data[col].unique())    # Unique values in the test set
    
    # Identify unseen labels
    unseen_labels = test_unique - train_unique
    
    if unseen_labels:
        print(f"\nColumn: {col}")
        print("Unique Values (Train):", sorted(train_unique))
        print("Unique Values (Test):", sorted(test_unique))
        print("Unseen Labels:", sorted(unseen_labels))

  # Re-encode labels to ensure they match the expected number of classes
label_encoders = {}
for col in target_columns:
    le = LabelEncoder()
    y_train_tabular[col] = le.fit_transform(train_data[col])  # Encode training labels
    y_test_tabular[col] = le.transform(test_data[col])        # Encode test labels
    label_encoders[col] = le


for col in target_columns:
    print(f"\nColumn: {col}")
    print("Unique Values (Train):", train_data[col].unique())
    print("Unique Values (Test):", test_data[col].unique())




#================ Distribution for Imbalance ==================================
for col in target_columns:
    print(f"\nColumn: {col}")
    print("Unique Values (Train):", np.unique(train_data[col], return_counts=True))
    print("Unique Values (Test):", np.unique(test_data[col], return_counts=True))

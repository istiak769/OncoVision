# Unet Architecture with feature extraction from bottleneck layer
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform

def upsample_block(x, filters, kernel_size=(3, 3), padding='same', strides=1):
    x = Conv2DTranspose(filters, kernel_size, padding=padding, strides=strides)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)  # LeakyReLU activation
    return x

def multi_unet_model2(n_classes=5, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    # Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(s)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c5)

    # Extra layer with 512 channels (bottleneck)
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c5)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(0.3)(c6)
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c6)

    # Expansive path
    u6 = upsample_block(c6, 256, strides=(2, 2))
    u6 = concatenate([u6, c4])
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(u6)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c7)

    u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u7 = concatenate([u7, c3])
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(u7)
    c8 = BatchNormalization()(c8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c8)

    u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u8 = concatenate([u8, c2])
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(u8)
    c9 = BatchNormalization()(c9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c9)

    u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c9)
    u9 = concatenate([u9, c1], axis=3)
    c10 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(u9)
    c10 = BatchNormalization()(c10)
    c10 = Dropout(0.1)(c10)
    c10 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c10)

    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c10)

    # Create the full segmentation model
    model = Model(inputs=[inputs], outputs=[outputs])

    # Create a secondary model for feature extraction
    feature_extractor = Model(inputs=[inputs], outputs=[c6])  # Extract features from the bottleneck layer

    return model, feature_extractor


segmentation_model, feature_extractor = multi_unet_model2(n_classes, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)




# ====== Tabular MLP model with feature extractctor ================================= 

from tensorflow.keras.models import Model

def create_multi_output_model_with_feature_extractor(input_dim, output_dims):
    inputs = Input(shape=(input_dim,))
    
    # Shared layers
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Feature extraction layer
    feature_layer = Dense(64, activation='relu', name='feature_layer')(x)
    
    # Output layers for each target variable
    outputs = {}
    losses = {}  # Dictionary to store loss functions for each output
    metrics = {}  # Dictionary to store metrics for each output
    for col, num_classes in output_dims.items():
        if num_classes == 1:  # Constant variable (single class)
            outputs[col] = Dense(1, activation='linear', name=col)(feature_layer)  # Linear activation for constant output
            losses[col] = None  # No loss for constant variables
            metrics[col] = None  # No metric for constant variables
        elif num_classes == 2:  # Binary classification
            outputs[col] = Dense(1, activation='sigmoid', name=col)(feature_layer)
            losses[col] = 'binary_crossentropy'
            metrics[col] = 'accuracy'
        else:  # Multi-class classification
            outputs[col] = Dense(num_classes, activation='softmax', name=col)(feature_layer)
            losses[col] = 'sparse_categorical_crossentropy'
            metrics[col] = 'accuracy'
    
    # Create the full model
    full_model = Model(inputs=inputs, outputs=outputs)
    
    # Create a feature extractor model
    feature_extractor = Model(inputs=inputs, outputs=feature_layer)
    
    # Compile the full model
    full_model.compile(optimizer=Adam(learning_rate=0.001),
                       loss=losses,
                       metrics=metrics)
    
    return full_model, feature_extractor

# Create the multi-output model with a feature extractor
input_dim = X_train_tabular.shape[1]
multi_output_model, tabular_feature_extractor = create_multi_output_model_with_feature_extractor(input_dim, output_dims)


# ============= Final Late Independent Fusion Model ========================

def create_tabular_prediction_model(input_dim, output_dims):
    inputs = Input(shape=(input_dim,))
    
    # Shared layers
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Output layers for each target variable
    outputs = {}
    losses = {}
    metrics = {}
    for col, num_classes in output_dims.items():
        if num_classes == 1:  # Constant variable
            outputs[col] = Dense(1, activation='linear', name=col)(x)
            losses[col] = None
            metrics[col] = None
        elif num_classes == 2:  # Binary classification
            outputs[col] = Dense(1, activation='sigmoid', name=col)(x)
            losses[col] = 'binary_crossentropy'
            metrics[col] = 'accuracy'
        else:  # Multi-class classification
            outputs[col] = Dense(num_classes, activation='softmax', name=col)(x)
            losses[col] = 'sparse_categorical_crossentropy'
            metrics[col] = 'accuracy'
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=losses,
                  metrics=metrics)
    return model




# Define input dimension for segmentation features
segmentation_input_dim = train_features_segmentation.shape[1]

# Create the tabular prediction model
tabular_prediction_model = create_tabular_prediction_model(segmentation_input_dim, output_dims)









# ============ Feature Extracttion from Segmentation Model =========================

# Define segmentation data for feature extraction
X_train_segmentation = X_train
X_test_segmentation = X_test

# Generate segmentation features
train_features_segmentation = feature_extractor.predict(X_train_segmentation)
test_features_segmentation = feature_extractor.predict(X_test_segmentation)

print("Train Segmentation Feature Shape:", train_features_segmentation.shape)
print("Test Segmentation Feature Shape:", test_features_segmentation.shape) 



# =========== Feature Extraction from Tabular MLP Model ============================ 

# Generate tabular features
train_features_tabular = tabular_feature_extractor.predict(X_train_tabular)
test_features_tabular = tabular_feature_extractor.predict(X_test_tabular)

print("Train Tabular Feature Shape:", train_features_tabular.shape)
print("Test Tabular Feature Shape:", test_features_tabular.shape) 



# ============= Processing for Extrated Features ============================== 
# Example: Ensure alignment between segmentation and tabular data
assert len(X_train_segmentation) == len(X_train_tabular), "Mismatch in training set sizes"
assert len(X_test_segmentation) == len(X_test_tabular), "Mismatch in test set sizes"

# Example: Apply Global Average Pooling to segmentation features
train_features_segmentation = train_features_segmentation.reshape(len(train_features_segmentation), -1, 512).mean(axis=1)
test_features_segmentation = test_features_segmentation.reshape(len(test_features_segmentation), -1, 512).mean(axis=1)

print("Reduced Train Segmentation Feature Shape:", train_features_segmentation.shape)
print("Reduced Test Segmentation Feature Shape:", test_features_segmentation.shape)  # Make sure these matches with the shape of the dataset

# Ensure alignment between segmentation and tabular data
assert len(train_features_segmentation) == len(y_train_tabular[list(y_train_tabular.keys())[0]]), "Mismatch in training set sizes"
assert len(test_features_segmentation) == len(y_test_tabular[list(y_test_tabular.keys())[0]]), "Mismatch in test set sizes" 



# Normalize features (optional but recommended)
from sklearn.preprocessing import StandardScaler

# Normalize segmentation and tabular features
scaler_seg = StandardScaler()
train_features_segmentation = scaler_seg.fit_transform(train_features_segmentation)
test_features_segmentation = scaler_seg.transform(test_features_segmentation)

scaler_tab = StandardScaler()
train_features_tabular = scaler_tab.fit_transform(train_features_tabular)
test_features_tabular = scaler_tab.transform(test_features_tabular)

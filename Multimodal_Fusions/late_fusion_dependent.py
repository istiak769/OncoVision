# Attention based Unet Architecture with feature extraction from bottleneck layer
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose, BatchNormalization, Activation, LeakyReLU, Multiply, Add, GlobalAveragePooling2D, Reshape, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform

def attention_block(x, g, inter_channel):
    theta_x = Conv2D(inter_channel, (1, 1), strides=(1, 1))(x)
    phi_g = Conv2D(inter_channel, (1, 1), strides=(1, 1))(g)
    f = Activation('relu')(Add()([theta_x, phi_g]))
    psi_f = Conv2D(1, (1, 1), strides=(1, 1))(f)
    rate = Activation('sigmoid')(psi_f)
    att_x = Multiply()([x, rate])
    return att_x

def upsample_block(x, filters, kernel_size=(3, 3), padding='same', strides=1):
    x = Conv2DTranspose(filters, kernel_size, padding=padding, strides=strides)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)  # LeakyReLU activation
    return x

def multi_unet_model2(n_classes=5, IMG_HEIGHT=SIZE_X, IMG_WIDTH=SIZE_Y, IMG_CHANNELS=1):
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

    # Expansive path with attention gates
    u6 = upsample_block(c6, 256, strides=(2, 2))
    att6 = attention_block(c4, u6, 128)
    u6 = concatenate([u6, att6])
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(u6)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c7)

    u7 = upsample_block(c7, 128, strides=(2, 2))
    att7 = attention_block(c3, u7, 64)
    u7 = concatenate([u7, att7])
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(u7)
    c8 = BatchNormalization()(c8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c8)

    u8 = upsample_block(c8, 64, strides=(2, 2))
    att8 = attention_block(c2, u8, 32)
    u8 = concatenate([u8, att8])
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(u8)
    c9 = BatchNormalization()(c9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c9)

    u9 = upsample_block(c9, 32, strides=(2, 2))
    att9 = attention_block(c1, u9, 16)
    u9 = concatenate([u9, att9])
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



#=============== Dependept Late Fusion Model ================================= 

def create_tabular_prediction_model_raw(input_dim, output_dims):
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
tabular_prediction_model_raw = create_tabular_prediction_model_raw(segmentation_input_dim, output_dims)



#============= Segmentation Feature Extraction and Processing =============================

from sklearn.preprocessing import StandardScaler

# Define segmentation data for feature extraction
X_train_segmentation = X_train
X_test_segmentation = X_test

# Generate segmentation features
train_features_segmentation = feature_extractor.predict(X_train_segmentation)
test_features_segmentation = feature_extractor.predict(X_test_segmentation)
print("Train Segmentation Feature Shape:", train_features_segmentation.shape)
print("Test Segmentation Feature Shape:", test_features_segmentation.shape) 


# Apply Global Average Pooling to reduce dimensions
train_features_segmentation = train_features_segmentation.reshape(len(train_features_segmentation), -1, 512).mean(axis=1)
test_features_segmentation = test_features_segmentation.reshape(len(test_features_segmentation), -1, 512).mean(axis=1)
print("Reduced Train Segmentation Feature Shape:", train_features_segmentation.shape)
print("Reduced Test Segmentation Feature Shape:", test_features_segmentation.shape) 


# Normalize segmentation features
scaler_seg = StandardScaler()
train_features_segmentation = scaler_seg.fit_transform(train_features_segmentation)
test_features_segmentation = scaler_seg.transform(test_features_segmentation)






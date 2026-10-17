import tensorflow as tf
from tensorflow.keras import layers, Model

# --- Model 1: HDSV-Net (ඔබේ ප්‍රධාන ආකෘතිය) ---
def build_hdsv_net(input_shape=(64, 64, 1)):
    """HDSV-Net: Single-input Binary Classification"""
    inputs = layers.Input(shape=input_shape)
    
    # CNN සංරචකය
    x = layers.Conv2D(16, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    
    # GNN සංරචකය (Dense ලෙස සරල කර ඇත)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Yes/No Classification
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=output, name="HDSV_Net")
    return model

# --- Model 2: Siamese CNN (Baseline) ---
def build_siamese_cnn(input_shape=(64, 64, 1)):
    """Siamese CNN: Single-input Baseline"""
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(16, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=output, name="Siamese_CNN")
    return model

# --- Model 3: ResNet50 (Transfer Learning) ---
def build_resnet50(input_shape=(64, 64, 1)):
    input_tensor = layers.Input(shape=input_shape)
    x = layers.Conv2D(3, (3, 3), padding='same')(input_tensor) # Grayscale -> RGB
    base_model = tf.keras.applications.ResNet50(include_top=False, weights=None, input_tensor=x)
    x = layers.GlobalAveragePooling2D()(base_model.output)
    output = layers.Dense(1, activation='sigmoid')(x)
    return Model(inputs=input_tensor, outputs=output, name="ResNet50")

# --- Model 4: MobileNetV2 (Edge Deployment) ---
def build_mobilenetv2(input_shape=(64, 64, 1)):
    input_tensor = layers.Input(shape=input_shape)
    x = layers.Conv2D(3, (3, 3), padding='same')(input_tensor)
    base_model = tf.keras.applications.MobileNetV2(include_top=False, weights=None, input_tensor=x)
    x = layers.GlobalAveragePooling2D()(base_model.output)
    output = layers.Dense(1, activation='sigmoid')(x)
    return Model(inputs=input_tensor, outputs=output, name="MobileNetV2")

# --- Model 5: Vision Transformer (ViT) ---
def build_vit(input_shape=(64, 64, 1)):
    from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dense, Dropout
    
    patch_size = 8
    num_patches = (64 // patch_size) ** 2
    
    inputs = layers.Input(shape=input_shape)
    patches = layers.Conv2D(32, kernel_size=patch_size, strides=patch_size)(inputs)
    patches = layers.Reshape((num_patches, 32))(patches)
    
    x = MultiHeadAttention(num_heads=4, key_dim=32)(patches, patches)
    x = LayerNormalization()(x)
    x = layers.Flatten()(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    
    return Model(inputs=inputs, outputs=output, name="Vision_Transformer")
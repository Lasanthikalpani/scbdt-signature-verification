import tensorflow as tf
from tensorflow.keras import layers, Model

# --- Model 1: HDSV-Net (ඔබේ ප්‍රධාන ආකෘතිය) ---
def build_hdsv_net(input_shape=(128, 128, 1)):
    """HDSV-Net: CNN + GNN (Dense) + Siamese"""
    base = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(256, activation='relu'), # GNN සංරචකය (සරල කර ඇත)
        layers.Dense(128, activation='relu')
    ])

    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)

    emb_a = base(input_a)
    emb_b = base(input_b)

    # CBI₆: දුර ගණනය කිරීම
    dist = layers.Lambda(lambda x: tf.norm(x[0] - x[1], axis=1))([emb_a, emb_b])
    
    # Yes/No Classification
    output = layers.Dense(1, activation='sigmoid')(dist)

    model = Model(inputs=[input_a, input_b], outputs=output, name="HDSV_Net")
    return model

# --- Model 2: Siamese CNN (Baseline) ---
def build_siamese_cnn(input_shape=(128, 128, 1)):
    base = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu')
    ])

    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)

    emb_a = base(input_a)
    emb_b = base(input_b)

    dist = layers.Lambda(lambda x: tf.norm(x[0] - x[1], axis=1))([emb_a, emb_b])
    output = layers.Dense(1, activation='sigmoid')(dist)

    model = Model(inputs=[input_a, input_b], outputs=output, name="Siamese_CNN")
    return model

# --- Model 3: ResNet50 (Transfer Learning) ---
def build_resnet50(input_shape=(128, 128, 1)):
    # ResNet50 සඳහා 3-channel (RGB) අවශ්‍ය වේ
    input_tensor = layers.Input(shape=input_shape)
    x = layers.Conv2D(3, (3, 3), padding='same')(input_tensor) # Grayscale -> RGB
    base_model = tf.keras.applications.ResNet50(include_top=False, weights=None, input_tensor=x)
    x = layers.GlobalAveragePooling2D()(base_model.output)
    output = layers.Dense(1, activation='sigmoid')(x)
    return Model(inputs=input_tensor, outputs=output, name="ResNet50")

# --- Model 4: MobileNetV2 (Edge Deployment) ---
def build_mobilenetv2(input_shape=(128, 128, 1)):
    input_tensor = layers.Input(shape=input_shape)
    x = layers.Conv2D(3, (3, 3), padding='same')(input_tensor)
    base_model = tf.keras.applications.MobileNetV2(include_top=False, weights=None, input_tensor=x)
    x = layers.GlobalAveragePooling2D()(base_model.output)
    output = layers.Dense(1, activation='sigmoid')(x)
    return Model(inputs=input_tensor, outputs=output, name="MobileNetV2")

# --- Model 5: Vision Transformer (ViT) ---
def build_vit(input_shape=(128, 128, 1)):
    from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dense, Dropout
    
    # පේටන්ට් පැච් (Patch) ලෙස බෙදීම
    patch_size = 16
    num_patches = (128 // patch_size) ** 2
    
    inputs = layers.Input(shape=input_shape)
    patches = layers.Conv2D(64, kernel_size=patch_size, strides=patch_size)(inputs)
    patches = layers.Reshape((num_patches, 64))(patches)
    
    # Transformer Block (සරල කර ඇත)
    x = MultiHeadAttention(num_heads=4, key_dim=64)(patches, patches)
    x = LayerNormalization()(x)
    x = layers.Flatten()(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    
    return Model(inputs=inputs, outputs=output, name="Vision_Transformer")
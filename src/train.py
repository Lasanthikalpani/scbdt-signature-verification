import os
import tensorflow as tf
from src.data_loader import get_data_generators
from src.models import build_hdsv_net, build_siamese_cnn, build_resnet50, build_mobilenetv2, build_vit

def train_model(model_name, data_dir):
    # Data Loaders
    train_gen, val_gen = get_data_generators(data_dir, img_size=(64, 64))
    
    # Model Selection
    if model_name == "HDSV-Net":
        model = build_hdsv_net()
    elif model_name == "Siamese-CNN":
        model = build_siamese_cnn()
    elif model_name == "ResNet50":
        model = build_resnet50()
    elif model_name == "MobileNetV2":
        model = build_mobilenetv2()
    elif model_name == "ViT":
        model = build_vit()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=30,
        steps_per_epoch=len(train_gen),
        validation_steps=len(val_gen)
    )
    
    # Save Model (ෆෝල්ඩරය නොමැති නම් ස්වයංක්‍රීයව සාදන්න)
    os.makedirs("models", exist_ok=True)
    model.save(f"models/{model_name}.h5")
    print(f"✅ {model_name} පුහුණු කර models/ ෆෝල්ඩරයේ සුරකින ලදී.")
    
    return history

if __name__ == "__main__":
    # ආකෘතිය තෝරන්න (උදා: HDSV-Net)
    train_model("HDSV-Net", "data/")
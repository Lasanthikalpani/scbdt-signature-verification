import tensorflow as tf
from src.data_loader import get_data_generators
from src.models import build_hdsv_net, build_siamese_cnn, build_resnet50

def train_model(model_name, data_dir):
    # Data Loaders
    train_gen, val_gen = get_data_generators(data_dir)
    
    # Model Selection
    if model_name == "HDSV-Net":
        model = build_hdsv_net()
    elif model_name == "Siamese-CNN":
        model = build_siamese_cnn()
    else:
        model = build_resnet50()
    
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
        epochs=10,
        steps_per_epoch=len(train_gen),
        validation_steps=len(val_gen)
    )
    
    # Save Model
    model.save(f"models/{model_name}.h5")
    print(f"✅ {model_name} පුහුණු කර සුරකින ලදී.")
    return history

if __name__ == "__main__":
    # CEDAR දත්ත සමුදාය සඳහා
    train_model("HDSV-Net", "data/")
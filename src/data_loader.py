import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(data_dir, batch_size=32, img_size=(64, 64)):
    """
    Data Augmentation සහ Data Loading (අධීක්ෂක ඉල්ලීමට අනුව 5000+)
    """
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,           # භ්‍රමණය
        width_shift_range=0.1,       # තිරස් මාරුව
        height_shift_range=0.1,      # සිරස් මාරුව
        zoom_range=0.1,              # විශාලනය
        shear_range=0.1,             # ඇල කිරීම
        brightness_range=[0.8, 1.2], # දීප්තිය
        horizontal_flip=False,       # අත්සන් සඳහා භ්‍රමණය නොකරන්න
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        color_mode='grayscale'
    )

    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        color_mode='grayscale'
    )

    return train_generator, val_generator
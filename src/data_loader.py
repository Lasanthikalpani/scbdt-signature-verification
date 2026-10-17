import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def get_data_generators(data_dir, batch_size=32, img_size=(128, 128)):
    """
    Data Augmentation සහ Data Loading
    """
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1,
        brightness_range=[0.8, 1.2],
        validation_split=0.2,
        rescale=1./255
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
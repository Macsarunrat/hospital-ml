import tensorflow as tf
from tensorflow.keras import layers, models


def get_augmentation():
    """Data augmentation layer block."""
    return models.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.2),
        ],
        name="data_augmentation",
    )


def build_model(input_shape=(224, 224, 3)):
    """Build regression model based on EfficientNetB0 backbone.

    Initially, base_model weights are frozen for top-layer warm-up.
    """
    data_augmentation = get_augmentation()

    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    model = models.Sequential(
        [
            data_augmentation,
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="linear"),
        ],
        name="rice_regression_model",
    )

    return model, base_model


def unfreeze_for_finetuning(model, base_model):
    """Unfreeze base_model for fine-tuning while keeping BatchNormalization layers frozen."""
    base_model.trainable = True
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    return model

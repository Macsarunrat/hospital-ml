import tensorflow as tf
from tensorflow.keras import layers, models

from rice.src.augmentation import get_rice_augmentation
from rice.src.config import RiceConfig


def build_model(input_shape=(*RiceConfig.IMAGE_SIZE, RiceConfig.IMAGE_CHANNELS)):
    """สร้างโครงสร้างโมเดล Regression โดยใช้ Backbone ตามที่ระบุใน RiceConfig

    รอบแรก Freeze base_model ไว้สำหรับเทรนเฉพาะ Dense Head
    """
    data_augmentation = get_rice_augmentation()

    backbone_cls = getattr(tf.keras.applications, RiceConfig.BACKBONE, None)
    if backbone_cls is None:
        raise ValueError(
            f"Backbone '{RiceConfig.BACKBONE}' ไม่ถูกต้องหรือไม่รองรับใน tf.keras.applications"
        )

    base_model = backbone_cls(
        input_shape=input_shape,
        include_top=False,
        weights=RiceConfig.WEIGHTS,
    )
    base_model.trainable = False

    model = models.Sequential(
        [
            data_augmentation,
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(RiceConfig.DENSE_UNITS, activation="relu"),
            layers.Dropout(RiceConfig.DROPOUT_RATE),
            layers.Dense(1, activation="linear"),
        ],
        name=f"rice_{RiceConfig.BACKBONE.lower()}_regression_model",
    )

    return model, base_model


def unfreeze_for_finetuning(model, base_model):
    """ปลดล็อกเลเยอร์ของ Backbone สำหรับทำ Fine-tuning

    โดยล็อก BatchNormalization layers ไว้เพื่อป้องกันไม่ให้สถิติ Mean/Variance เสียหาย
    """
    base_model.trainable = True
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    return model

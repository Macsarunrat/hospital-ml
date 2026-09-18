import tensorflow as tf
from tensorflow.keras import layers, models

from rice.src.augmentation import get_rice_augmentation
from rice.src.config import RiceConfig


def build_base_regression_head(base_model, model_name, preprocessing_layer=None):
    """ประกอบ Regression Head มาตรฐานต่อท้าย Base Feature Extractor

    Parameters
    ----------
    base_model : tf.keras.Model
        Backbone Feature Extractor ที่เอา Top ออกแล้ว
    model_name : str
        ชื่อกำกับโมเดล
    preprocessing_layer : tf.keras.layers.Layer or callable, optional
        เลเยอร์แปลงข้อมูลก่อนเข้าโมเดล (เช่น สำหรับ ResNet ที่ต้องการ Lambda preprocess_input)
    """
    data_augmentation = get_rice_augmentation()

    pipeline_layers = [data_augmentation]
    if preprocessing_layer is not None:
        pipeline_layers.append(preprocessing_layer)

    pipeline_layers.extend(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(RiceConfig.DENSE_UNITS, activation="relu"),
            layers.Dropout(RiceConfig.DROPOUT_RATE),
            layers.Dense(1, activation="linear"),
        ]
    )

    model = models.Sequential(pipeline_layers, name=model_name)
    return model, base_model


def default_unfreeze_for_finetuning(model, base_model):
    """ปลดล็อกเลเยอร์ของ Backbone สำหรับ Fine-Tuning

    โดยล็อก BatchNormalization layers ไว้เพื่อป้องกันไม่ให้ค่าสถิติ Mean/Variance เสียหาย
    """
    base_model.trainable = True
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    return model

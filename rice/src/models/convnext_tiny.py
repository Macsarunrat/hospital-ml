import tensorflow as tf

from rice.src.config import RiceConfig
from rice.src.models.base import build_base_regression_head, default_unfreeze_for_finetuning

IMAGE_SIZE = (224, 224)


def build_model(input_shape=(*IMAGE_SIZE, RiceConfig.IMAGE_CHANNELS)):
    """สร้างโมเดล ConvNeXtTiny Regression

    ConvNeXt เป็น Modern Pure CNN ที่นำเทคนิคของ Vision Transformer มาปรับปรุง CNN
    มี Normalization ในตัว จึงไม่ต้องใส่ external preprocess_input
    """
    base_model = tf.keras.applications.ConvNeXtTiny(
        input_shape=input_shape,
        include_top=False,
        weights=RiceConfig.WEIGHTS,
    )
    base_model.trainable = False

    return build_base_regression_head(
        base_model=base_model,
        model_name="rice_convnexttiny_regression_model",
        preprocessing_layer=None,
    )


def unfreeze_for_finetuning(model, base_model):
    """ปลดล็อกเลเยอร์ของ ConvNeXtTiny พร้อมคง Freeze LayerNorm / BatchNorm"""
    base_model.trainable = True
    for layer in base_model.layers:
        if isinstance(layer, (tf.keras.layers.BatchNormalization, tf.keras.layers.LayerNormalization)):
            layer.trainable = False
    return model

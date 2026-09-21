import tensorflow as tf
from tensorflow.keras import layers

from rice.src.config import RiceConfig
from rice.src.models.base import build_base_regression_head, default_unfreeze_for_finetuning

IMAGE_SIZE = (224, 224)


def build_model(input_shape=(*IMAGE_SIZE, RiceConfig.IMAGE_CHANNELS)):
    """สร้างโมเดล DenseNet121 Regression

    DenseNet ใช้ Dense Connectivity เชื่อมโยงทุกเลเยอร์ เหมาะมากกับ Texture เม็ดข้าว
    ต้องการการสเกลพิกเซลด้วย densenet.preprocess_input
    """
    base_model = tf.keras.applications.DenseNet121(
        input_shape=input_shape,
        include_top=False,
        weights=RiceConfig.WEIGHTS,
    )
    base_model.trainable = False

    preprocess_layer = layers.Lambda(
        tf.keras.applications.densenet.preprocess_input,
        name="densenet_preprocess",
    )

    return build_base_regression_head(
        base_model=base_model,
        model_name="rice_densenet121_regression_model",
        preprocessing_layer=preprocess_layer,
    )


def unfreeze_for_finetuning(model, base_model):
    """ปลดล็อกเลเยอร์ของ DenseNet121 พร้อมคง Freeze BatchNormalization"""
    return default_unfreeze_for_finetuning(model, base_model)

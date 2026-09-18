import tensorflow as tf
from tensorflow.keras import layers

from rice.src.config import RiceConfig
from rice.src.models.base import build_base_regression_head, default_unfreeze_for_finetuning

IMAGE_SIZE = (224, 224)


def build_model(input_shape=(*IMAGE_SIZE, RiceConfig.IMAGE_CHANNELS)):
    """สร้างโมเดล ResNet50V2 Regression พร้อม Preprocessing เฉพาะตัว (สเกลพิกเซล [-1, 1])"""
    base_model = tf.keras.applications.ResNet50V2(
        input_shape=input_shape,
        include_top=False,
        weights=RiceConfig.WEIGHTS,
    )
    base_model.trainable = False

    # ResNet50V2 ต้องการ tf.keras.applications.resnet_v2.preprocess_input
    preprocess_layer = layers.Lambda(
        tf.keras.applications.resnet_v2.preprocess_input,
        name="resnet_v2_preprocess",
    )

    return build_base_regression_head(
        base_model=base_model,
        model_name="rice_resnet50v2_regression_model",
        preprocessing_layer=preprocess_layer,
    )


def unfreeze_for_finetuning(model, base_model):
    """ปลดล็อกเลเยอร์ของ ResNet50V2 พร้อมคง Freeze BatchNormalization"""
    return default_unfreeze_for_finetuning(model, base_model)

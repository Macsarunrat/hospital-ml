import tensorflow as tf

from rice.src.config import RiceConfig
from rice.src.models.base import build_base_regression_head, default_unfreeze_for_finetuning

IMAGE_SIZE = (224, 224)


def build_model(input_shape=(*IMAGE_SIZE, RiceConfig.IMAGE_CHANNELS)):
    """สร้างโมเดล EfficientNetB0 Regression"""
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights=RiceConfig.WEIGHTS,
    )
    base_model.trainable = False

    return build_base_regression_head(
        base_model=base_model,
        model_name="rice_efficientnetb0_regression_model",
        preprocessing_layer=None,  # EfficientNet มี rescaling ในตัว
    )


def unfreeze_for_finetuning(model, base_model):
    """ปลดล็อกเลเยอร์ของ EfficientNetB0 พร้อมคง Freeze BatchNormalization"""
    return default_unfreeze_for_finetuning(model, base_model)

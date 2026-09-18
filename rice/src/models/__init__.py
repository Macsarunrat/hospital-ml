from rice.src.models import (
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_v2_b0,
    resnet50_v2,
)

MODEL_REGISTRY = {
    "EfficientNetB0": {
        "builder": efficientnet_b0.build_model,
        "unfreezer": efficientnet_b0.unfreeze_for_finetuning,
        "image_size": efficientnet_b0.IMAGE_SIZE,
    },
    "EfficientNetB1": {
        "builder": efficientnet_b1.build_model,
        "unfreezer": efficientnet_b1.unfreeze_for_finetuning,
        "image_size": efficientnet_b1.IMAGE_SIZE,
    },
    "EfficientNetV2B0": {
        "builder": efficientnet_v2_b0.build_model,
        "unfreezer": efficientnet_v2_b0.unfreeze_for_finetuning,
        "image_size": efficientnet_v2_b0.IMAGE_SIZE,
    },
    "ResNet50V2": {
        "builder": resnet50_v2.build_model,
        "unfreezer": resnet50_v2.unfreeze_for_finetuning,
        "image_size": resnet50_v2.IMAGE_SIZE,
    },
}


def get_supported_models():
    """รายชื่อโมเดลทั้งหมดที่รองรับ"""
    return list(MODEL_REGISTRY.keys())


def get_model_image_size(model_name):
    """ดึงขนาดภาพ Input สำหรับโมเดลที่เลือก"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"โมเดล '{model_name}' ไม่ถูกต้อง เลือกรุ่นที่รองรับ: {get_supported_models()}"
        )
    return MODEL_REGISTRY[model_name]["image_size"]


def build_model_by_name(model_name=None):
    """สร้างโมเดลตามชื่อที่ระบุ (หากไม่ระบุ จะดึงจาก RiceConfig.BACKBONE)"""
    from rice.src.config import RiceConfig

    target_name = model_name or RiceConfig.BACKBONE
    if target_name not in MODEL_REGISTRY:
        raise ValueError(
            f"โมเดล '{target_name}' ไม่ถูกต้อง เลือกรุ่นที่รองรับ: {get_supported_models()}"
        )

    return MODEL_REGISTRY[target_name]["builder"]()


def unfreeze_model_by_name(model, base_model, model_name=None):
    """ปลดล็อกเลเยอร์ Backbone สำหรับ Fine-tuning ตามเทคนิคของโมเดลนั้น"""
    from rice.src.config import RiceConfig

    target_name = model_name or RiceConfig.BACKBONE
    if target_name not in MODEL_REGISTRY:
        raise ValueError(
            f"โมเดล '{target_name}' ไม่ถูกต้อง เลือกรุ่นที่รองรับ: {get_supported_models()}"
        )

    return MODEL_REGISTRY[target_name]["unfreezer"](model, base_model)

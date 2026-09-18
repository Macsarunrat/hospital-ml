"""โมดูลกลางสำหรับเข้าถึงโมเดล Rice Regression (รองรับ Backward Compatibility)"""
from rice.src.models import (
    build_model_by_name as build_model,
    get_model_image_size,
    get_supported_models,
    unfreeze_model_by_name as unfreeze_for_finetuning,
)

__all__ = [
    "build_model",
    "unfreeze_for_finetuning",
    "get_model_image_size",
    "get_supported_models",
]

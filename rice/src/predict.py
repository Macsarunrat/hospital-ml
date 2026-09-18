import os
import numpy as np
import tensorflow as tf
from rice.src.config import RiceConfig


def find_model_file(model_path=None):
    """ค้นหาไฟล์โมเดลอัตโนมัติ โดยให้ความสำคัญกับโฟลเดอร์ rice/output/ ก่อน"""
    if model_path and os.path.exists(model_path):
        return model_path

    candidates = [
        model_path,
        os.path.join("rice", "output", "best_rice_model.keras"),
        os.path.join("rice", "output", "Rice_regression.keras"),
        RiceConfig.BEST_MODEL_PATH,
        "best_rice_model.keras",
        "Rice_regression.keras",
        os.path.join("output", "best_rice_model.keras"),
        "/kaggle/working/best_rice_model.keras",
    ]

    for path in candidates:
        if path and os.path.exists(path):
            print(f"[Predict] Model found at: {path}")
            return path

    raise FileNotFoundError(
        "Model file not found. Please verify that 'best_rice_model.keras' exists in 'rice/output/'."
    )


def load_trained_model(model_path=None):
    """โหลดโมเดลที่เทรนแล้วขึ้นมาใช้งาน"""
    actual_path = find_model_file(model_path)
    print(f"[Predict] Loading model from {actual_path}...")
    return tf.keras.models.load_model(actual_path)


def preprocess_image(image_path_or_bytes, image_size=None):
    """แปลงรูปภาพ 1 รูปให้เป็น Tensor พร้อมส่งเข้าโมเดล"""
    target_size = image_size or RiceConfig.get_image_size()
    if isinstance(image_path_or_bytes, str):
        if not os.path.exists(image_path_or_bytes):
            raise FileNotFoundError(f"Image file not found: {image_path_or_bytes}")
        image_raw = tf.io.read_file(image_path_or_bytes)
    else:
        image_raw = image_path_or_bytes

    image = tf.image.decode_jpeg(image_raw, channels=RiceConfig.IMAGE_CHANNELS)
    image = tf.image.resize_with_pad(image, target_size[0], target_size[1])
    image = tf.expand_dims(image, axis=0)
    return image


def predict_rice_weight(image_path, model=None, model_path=None):
    """ทำนายน้ำหนักข้าวที่เหลือจากภาพเดี่ยว (กรัม)"""
    if model is None:
        model = load_trained_model(model_path)

    image_tensor = preprocess_image(image_path)
    predicted_weight = model.predict(image_tensor, verbose=0)[0][0]
    return float(np.round(predicted_weight, 2))


def estimate_rice_nutrition(
    image_path,
    standard_weight_g=RiceConfig.DEFAULT_STANDARD_WEIGHT_G,
    model=None,
    model_path=None,
):
    """ฟังก์ชันระดับคลินิกสำหรับพยาบาลและนักกำหนดอาหาร:

    ประเมินปริมาณการรับประทาน, พลังงาน, และโปรตีนที่คนไข้ได้รับจริงจากภาพถ่ายหลังทาน
    """
    if model is None:
        model = load_trained_model(model_path)

    remaining_g = predict_rice_weight(image_path, model=model)
    remaining_g = float(np.clip(remaining_g, 0.0, standard_weight_g))

    intake_g = standard_weight_g - remaining_g
    intake_percent = (intake_g / standard_weight_g) * 100.0

    ratio = intake_g / 100.0
    calories = ratio * RiceConfig.CALORIES_PER_100G
    protein = ratio * RiceConfig.PROTEIN_PER_100G
    carbs = ratio * RiceConfig.CARBS_PER_100G
    fat = ratio * RiceConfig.FAT_PER_100G

    return {
        "standard_weight_g": standard_weight_g,
        "remaining_weight_g": round(remaining_g, 2),
        "intake_weight_g": round(intake_g, 2),
        "intake_percent": round(intake_percent, 1),
        "protein_g": round(protein, 2),
        "calories_kcal": round(calories, 1),
        "carbs_g": round(carbs, 2),
        "fat_g": round(fat, 2),
    }


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) > 1:
        test_img = sys.argv[1]
        custom_model = sys.argv[2] if len(sys.argv) > 2 else None
        nutrition = estimate_rice_nutrition(test_img, model_path=custom_model)
        print(f"\nNutrition Assessment for: {test_img}")
        print(json.dumps(nutrition, indent=4, ensure_ascii=False))
    else:
        print("Usage:")
        print("  python -m rice.src.predict <path_to_image.jpg>")
        print("  python -m rice.src.predict <path_to_image.jpg> <path_to_model.keras>")

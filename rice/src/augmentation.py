import tensorflow as tf
from tensorflow.keras import layers, models


def get_rice_augmentation(name="rice_data_augmentation"):
    """บล็อกแต่งภาพ (Data Augmentation) เฉพาะสำหรับถ้วยข้าว

    หมุน, ซูม, พลิก, ปรับคอนทราสต์ เพื่อให้โมเดลทนต่อมุมกล้องและแสงไฟในโรงพยาบาล
    """
    return models.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.2),
        ],
        name=name,
    )

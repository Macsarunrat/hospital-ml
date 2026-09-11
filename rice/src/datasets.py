import os
from glob import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

from rice.src.config import RiceConfig


def find_image_files():
    """ค้นหาไฟล์ภาพถ้วยข้าวจาก Input paths บน Kaggle พร้อมระบบ Auto-fallback"""
    potential_patterns = [
        "/kaggle/input/datasets/macsarun/cropped/bowl/*.jpg",
        "/kaggle/input/cropped/bowl/*.jpg",
        "/kaggle/input/Cropped/bowl/*.jpg",
        "/kaggle/input/**/bowl/*.jpg",
    ]

    for pattern in potential_patterns:
        files = glob(pattern, recursive=True)
        if files:
            print(f"พบรูปภาพจำนวน {len(files)} รูป จากตำแหน่ง: {pattern}")
            return files

    raise FileNotFoundError(
        "ไม่พบไฟล์รูปภาพถ้วยข้าว กรุณาตรวจสอบว่าได้ Add Dataset 'macsarun/Cropped' เข้ามาใน Kaggle แล้วหรือยัง"
    )


def get_data_splits(
    test_size=RiceConfig.TEST_SPLIT,
    val_size=RiceConfig.VAL_SPLIT,
    random_state=RiceConfig.RANDOM_STATE,
):
    """อ่านข้อมูลและสกัด Label น้ำหนักจากชื่อภาพ แล้วแบ่งเป็น Train / Val / Test"""
    images = find_image_files()

    initial_label = []
    remaining_label = []
    image_names = []
    full_paths = []

    for filename in images:
        base_name = os.path.basename(filename)
        full_paths.append(filename)
        temp = base_name.split("_")
        initial = temp[1]
        initial_label.append(initial)
        remaining_jpg = temp[-1]
        remaining = remaining_jpg.split(".")[0]
        remaining_label.append(remaining)
        image_names.append(base_name)

    df = pd.DataFrame(
        {
            "image_name": image_names,
            "initial_label": initial_label,
            "remaining_label": remaining_label,
            "full_path": full_paths,
        }
    )

    all_paths = df["full_path"].values
    all_labels = df["remaining_label"].values.astype("float32")

    # แบ่ง Test dataset (10%)
    train_val_path, test_path, train_val_label, test_label = train_test_split(
        all_paths, all_labels, test_size=test_size, random_state=random_state
    )

    # แบ่ง Train (ประมาณ 72%) และ Val (ประมาณ 18%)
    train_path, val_path, train_label, val_label = train_test_split(
        train_val_path, train_val_label, test_size=val_size, random_state=random_state
    )

    print(
        f"แบ่งชุดข้อมูลเรียบร้อย -> Train: {len(train_path)}, Val: {len(val_path)}, Test: {len(test_path)}"
    )
    return train_path, val_path, test_path, train_label, val_label, test_label


def load_and_preprocessing(path, label):
    """ฟังก์ชันโหลดรูปภาพแบบ Pure TensorFlow เพื่อทำ Image Pipeline"""
    image_raw = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image_raw, channels=RiceConfig.IMAGE_CHANNELS)
    image = tf.image.resize_with_pad(
        image, RiceConfig.IMAGE_SIZE[0], RiceConfig.IMAGE_SIZE[1]
    )
    return image, label


def create_datasets(batch_size=None):
    """สร้าง tf.data.Dataset สำหรับ Train, Val, และ Test พร้อมคืน test_path สำหรับทำ Error Analysis"""
    actual_batch_size = batch_size or RiceConfig.BATCH_SIZE_PER_REPLICA

    train_path, val_path, test_path, train_label, val_label, test_label = (
        get_data_splits()
    )

    train_ds = tf.data.Dataset.from_tensor_slices((train_path, train_label))
    train_ds = train_ds.map(load_and_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=len(train_path)).batch(actual_batch_size)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_path, val_label))
    val_ds = val_ds.map(load_and_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(actual_batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((test_path, test_label))
    test_ds = test_ds.map(load_and_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(actual_batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, test_label, test_path

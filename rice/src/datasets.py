import os
from glob import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
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


def extract_metadata():
    """อ่านข้อมูลรูปภาพทั้งหมดและแปลงเป็น DataFrame"""
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
    return df


def get_data_splits(
    test_size=RiceConfig.TEST_SPLIT,
    val_size=RiceConfig.VAL_SPLIT,
    random_state=RiceConfig.RANDOM_STATE,
):
    """อ่านข้อมูลและสกัด Label น้ำหนักจากชื่อภาพ แล้วแบ่งเป็น Train / Val / Test แบบรอบเดียว"""
    df = extract_metadata()
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


def get_kfold_splits(
    n_splits=RiceConfig.NUM_FOLDS,
    test_size=RiceConfig.TEST_SPLIT,
    random_state=RiceConfig.RANDOM_STATE,
):
    """แบ่งชุดข้อมูลออกเป็น Hold-out Test Set และ K-Fold สำหรับ Train/Val"""
    df = extract_metadata()
    all_paths = df["full_path"].values
    all_labels = df["remaining_label"].values.astype("float32")

    # 1. แยก Hold-out Test Set (10%) สำหรับวัดผลรวมรอบสุดท้าย
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        all_paths, all_labels, test_size=test_size, random_state=random_state
    )

    print(f"แบ่ง Hold-out Test Set สำเร็จ: {len(test_paths)} รูป")
    print(f"ข้อมูลสำหรับ {n_splits}-Fold Cross-Validation: {len(train_val_paths)} รูป")

    # 2. แบ่ง Fold ด้วย KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_val_paths)):
        tr_paths = train_val_paths[train_idx]
        tr_labels = train_val_labels[train_idx]
        vl_paths = train_val_paths[val_idx]
        vl_labels = train_val_labels[val_idx]
        folds.append((tr_paths, tr_labels, vl_paths, vl_labels))

    return folds, test_paths, test_labels


def load_and_preprocessing(path, label):
    """ฟังก์ชันโหลดรูปภาพแบบ Pure TensorFlow เพื่อทำ Image Pipeline"""
    image_raw = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image_raw, channels=RiceConfig.IMAGE_CHANNELS)
    img_size = RiceConfig.get_image_size()
    image = tf.image.resize_with_pad(
        image, img_size[0], img_size[1]
    )
    return image, label


def create_dataset_from_slices(paths, labels, batch_size, shuffle=False):
    """สร้าง tf.data.Dataset จาก arrays ของ paths และ labels"""
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_and_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths))
    ds = ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def create_datasets(batch_size=None):
    """สร้าง tf.data.Dataset สำหรับ Train, Val, และ Test แบบ Single Split"""
    actual_batch_size = batch_size or RiceConfig.BATCH_SIZE_PER_REPLICA

    train_path, val_path, test_path, train_label, val_label, test_label = (
        get_data_splits()
    )

    train_ds = create_dataset_from_slices(train_path, train_label, actual_batch_size, shuffle=True)
    val_ds = create_dataset_from_slices(val_path, val_label, actual_batch_size, shuffle=False)
    test_ds = create_dataset_from_slices(test_path, test_label, actual_batch_size, shuffle=False)

    return train_ds, val_ds, test_ds, test_label, test_path


def create_kfold_datasets(batch_size=None):
    """สร้าง tf.data.Dataset สำหรับ K-Fold แต่ละ Fold และ Hold-out Test Dataset"""
    actual_batch_size = batch_size or RiceConfig.BATCH_SIZE_PER_REPLICA

    folds_data, test_paths, test_labels = get_kfold_splits()

    fold_datasets = []
    for fold_idx, (tr_paths, tr_labels, vl_paths, vl_labels) in enumerate(folds_data):
        train_ds = create_dataset_from_slices(tr_paths, tr_labels, actual_batch_size, shuffle=True)
        val_ds = create_dataset_from_slices(vl_paths, vl_labels, actual_batch_size, shuffle=False)
        fold_datasets.append({
            "fold": fold_idx + 1,
            "train_ds": train_ds,
            "val_ds": val_ds,
            "val_paths": vl_paths,
            "val_labels": vl_labels,
            "train_count": len(tr_paths),
            "val_count": len(vl_paths),
        })

    test_ds = create_dataset_from_slices(test_paths, test_labels, actual_batch_size, shuffle=False)

    return fold_datasets, test_ds, test_labels, test_paths

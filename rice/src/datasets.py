import os
from glob import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf


def find_image_files():
    """Find bowl images from Kaggle dataset input paths with fallback support."""
    potential_patterns = [
        "/kaggle/input/datasets/macsarun/cropped/bowl/*.jpg",
        "/kaggle/input/cropped/bowl/*.jpg",
        "/kaggle/input/Cropped/bowl/*.jpg",
        "/kaggle/input/**/bowl/*.jpg",
    ]

    for pattern in potential_patterns:
        files = glob(pattern, recursive=True)
        if files:
            print(f"Found {len(files)} images using pattern: {pattern}")
            return files

    raise FileNotFoundError(
        "No images found. Please ensure the dataset 'macsarun/Cropped' is added in Kaggle."
    )


def get_data_splits(test_size=0.1, val_size=0.2, random_state=42):
    """Load image paths and parse labels from filenames, then split into train/val/test."""
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

    # Test dataset (10%)
    train_val_path, test_path, train_val_label, test_label = train_test_split(
        all_paths, all_labels, test_size=test_size, random_state=random_state
    )

    # Train (approx 72%) + Val (approx 18%)
    train_path, val_path, train_label, val_label = train_test_split(
        train_val_path, train_val_label, test_size=val_size, random_state=random_state
    )

    print(
        f"Dataset split -> Train: {len(train_path)}, Val: {len(val_path)}, Test: {len(test_path)}"
    )
    return train_path, val_path, test_path, train_label, val_label, test_label


def load_and_preprocessing(path, label):
    """Read jpeg, decode, and resize with pad to 224x224."""
    image_raw = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image_raw, channels=3)
    image = tf.image.resize_with_pad(image, 224, 224)
    return image, label


def create_datasets(batch_size=32):
    """Build tf.data pipelines for Train, Validation, and Test."""
    train_path, val_path, test_path, train_label, val_label, test_label = (
        get_data_splits()
    )

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_path, train_label)
    )
    train_dataset = train_dataset.map(
        load_and_preprocessing, num_parallel_calls=tf.data.AUTOTUNE
    )
    train_dataset = train_dataset.shuffle(buffer_size=len(train_path)).batch(
        batch_size
    )
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_path, val_label))
    val_dataset = val_dataset.map(
        load_and_preprocessing, num_parallel_calls=tf.data.AUTOTUNE
    )
    val_dataset = val_dataset.batch(batch_size).prefetch(
        buffer_size=tf.data.AUTOTUNE
    )

    test_dataset = tf.data.Dataset.from_tensor_slices((test_path, test_label))
    test_dataset = test_dataset.map(
        load_and_preprocessing, num_parallel_calls=tf.data.AUTOTUNE
    )
    test_dataset = test_dataset.batch(batch_size).prefetch(
        buffer_size=tf.data.AUTOTUNE
    )

    return train_dataset, val_dataset, test_dataset, test_label

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from rice.src.datasets import create_datasets
from rice.src.model import build_model, unfreeze_for_finetuning


def main():
    print("=" * 60)
    print("Starting Rice Regression Training on Multi-GPU / T4x2")
    print("=" * 60)

    # 1. Distributed Multi-GPU Strategy Setup
    gpus = tf.config.list_physical_devices("GPU")
    print(f"Detected {len(gpus)} physical GPU(s):")
    for gpu in gpus:
        print(f"  -> {gpu}")

    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(
            f"Using MirroredStrategy with {strategy.num_replicas_in_sync} devices."
        )
    elif len(gpus) == 1:
        strategy = tf.distribute.get_strategy()
        print("Using Single GPU strategy.")
    else:
        strategy = tf.distribute.get_strategy()
        print("No GPU detected! Using CPU strategy.")

    # Scale Global Batch Size with number of replicas
    per_replica_batch_size = 32
    global_batch_size = per_replica_batch_size * strategy.num_replicas_in_sync
    print(
        f"Per-replica Batch Size: {per_replica_batch_size}, Global Batch Size: {global_batch_size}"
    )

    # 2. Output Paths (Save into /kaggle/working if on Kaggle)
    output_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
    best_model_path = os.path.join(output_dir, "best_rice_model.keras")
    final_model_path = os.path.join(output_dir, "Rice_regression.keras")

    # 3. Data Pipeline
    train_dataset, val_dataset, test_dataset, test_label = create_datasets(
        batch_size=global_batch_size
    )

    # 4. Build Model within Strategy Scope
    with strategy.scope():
        model, base_model = build_model(input_shape=(224, 224, 3))
        model.compile(
            optimizer="adam",
            metrics=["mae"],
            loss="mean_squared_error",
        )

    callbacks = [
        ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_mae",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_mae",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # 5. Phase 1: Train Top Layers
    print("\n" + "=" * 50)
    print("Phase 1: Training Top Layers (Freeze Backbone)")
    print("=" * 50)
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=100,
        callbacks=callbacks,
    )

    # 6. Phase 2: Fine-Tuning Backbone
    print("\n" + "=" * 50)
    print("Phase 2: Fine-Tuning EfficientNetB0")
    print("=" * 50)
    with strategy.scope():
        model = unfreeze_for_finetuning(model, base_model)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="mean_squared_error",
            metrics=["mae"],
        )

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=20,
        callbacks=callbacks,
    )

    # 7. Save Final Model
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    print(f"Best checkpoint saved to: {best_model_path}")

    # 8. Evaluation on Test Dataset
    print("\n" + "=" * 50)
    print("Evaluating Model on Test Dataset")
    print("=" * 50)
    predictions = model.predict(test_dataset).flatten()
    actual_labels = np.array(test_label)

    mae = mean_absolute_error(actual_labels, predictions)
    rmse = np.sqrt(mean_squared_error(actual_labels, predictions))
    r2 = r2_score(actual_labels, predictions)

    print(f"Test Evaluation Results:")
    print(f"  - MAE : {mae:.4f}")
    print(f"  - RMSE: {rmse:.4f}")
    print(f"  - R2  : {r2:.4f}")
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from rice.src.config import RiceConfig


def get_training_callbacks(best_model_path=None, custom_callbacks=None):
    """รวบรวม Callbacks มาตรฐานสำหรับการเทรน:

    1. ModelCheckpoint: บันทึกเฉพาะรอบที่ val_mae ต่ำที่สุด
    2. EarlyStopping: หยุดเมื่อ val_mae ไม่พัฒนาเกินจำนวนรอบที่กำหนด
    3. ReduceLROnPlateau: ลด Learning Rate อัตโนมัติเมื่อ Loss ชะลอตัว
    """
    model_path = best_model_path or RiceConfig.BEST_MODEL_PATH

    callbacks = [
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_mae",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_mae",
            patience=RiceConfig.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=RiceConfig.REDUCE_LR_FACTOR,
            patience=RiceConfig.REDUCE_LR_PATIENCE,
            min_lr=RiceConfig.MIN_LR,
            verbose=1,
        ),
    ]

    if custom_callbacks:
        callbacks.extend(custom_callbacks)

    return callbacks

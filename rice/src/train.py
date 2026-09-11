import tensorflow as tf

from common.logger import WandbLogger
from rice.src.callbacks import get_training_callbacks
from rice.src.config import RiceConfig
from rice.src.datasets import create_datasets
from rice.src.evaluate import evaluate_model
from rice.src.model import build_model, unfreeze_for_finetuning


def main():
    print("=" * 60)
    print("Starting Rice Regression Training (Multi-GPU T4x2)")
    print("=" * 60)

    # 1. Distributed Multi-GPU Strategy Setup (Kaggle T4 x 2)
    gpus = tf.config.list_physical_devices("GPU")
    print(f"ตรวจพบการ์ดจอ {len(gpus)} ใบ:")
    for gpu in gpus:
        print(f"  -> {gpu}")

    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"เปิดใช้งาน MirroredStrategy บนการ์ดจอ {strategy.num_replicas_in_sync} ใบสำเร็จ!")
    elif len(gpus) == 1:
        strategy = tf.distribute.get_strategy()
        print("ใช้งาน Single GPU")
    else:
        strategy = tf.distribute.get_strategy()
        print("ไม่พบการ์ดจอ กำลังใช้งาน CPU")

    # คำนวณ Global Batch Size (คูณตามจำนวนการ์ดจอ)
    global_batch_size = RiceConfig.BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    print(f"Batch Size รวมสำหรับทั้งระบบ: {global_batch_size} (ใบละ {RiceConfig.BATCH_SIZE_PER_REPLICA})")

    # 2. ตั้งค่า WandB Logger (จาก common/logger.py)
    logger = WandbLogger(
        project_name=RiceConfig.WANDB_PROJECT,
        task_name=RiceConfig.WANDB_TASK,
        config={
            "model": RiceConfig.BACKBONE,
            "global_batch_size": global_batch_size,
            "phase1_epochs": RiceConfig.PHASE1_EPOCHS,
            "phase2_epochs": RiceConfig.PHASE2_EPOCHS,
            "phase2_lr": RiceConfig.PHASE2_LR,
        },
    )

    # 3. เตรียมชุดข้อมูล (Data Pipeline: คืน test_paths สำหรับทำ Error Analysis)
    train_ds, val_ds, test_ds, test_labels, test_paths = create_datasets(batch_size=global_batch_size)

    # 4. สร้างและ Compile โมเดลภายใต้ Strategy Scope
    with strategy.scope():
        model, base_model = build_model()
        model.compile(
            optimizer=RiceConfig.PHASE1_OPTIMIZER,
            metrics=["mae"],
            loss=RiceConfig.PHASE1_LOSS,
        )

    # รวบรวม Callbacks (Checkpoint, EarlyStopping, ReduceLR + WandB)
    callbacks = get_training_callbacks(
        best_model_path=RiceConfig.BEST_MODEL_PATH,
        custom_callbacks=logger.get_callbacks(),
    )

    # 5. Phase 1: เทรนเฉพาะ Dense Head (Freeze Backbone)
    print("\n" + "=" * 50)
    print(f"Phase 1: Training Top Layers ({RiceConfig.PHASE1_EPOCHS} Epochs)")
    print("=" * 50)
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=RiceConfig.PHASE1_EPOCHS,
        callbacks=callbacks,
    )

    # 6. Phase 2: Fine-Tuning EfficientNetB0
    print("\n" + "=" * 50)
    print(f"Phase 2: Fine-Tuning Backbone ({RiceConfig.PHASE2_EPOCHS} Epochs, lr={RiceConfig.PHASE2_LR})")
    print("=" * 50)
    with strategy.scope():
        model = unfreeze_for_finetuning(model, base_model)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=RiceConfig.PHASE2_LR),
            loss=RiceConfig.PHASE1_LOSS,
            metrics=["mae"],
        )

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=RiceConfig.PHASE2_EPOCHS,
        callbacks=callbacks,
    )

    # 7. บันทึก Final Model
    model.save(RiceConfig.FINAL_MODEL_PATH)
    print(f"\nบันทึกโมเดลรอบสุดท้ายไปที่: {RiceConfig.FINAL_MODEL_PATH}")
    print(f"บันทึกโมเดลรอบที่ดีที่สุดไปที่: {RiceConfig.BEST_MODEL_PATH}")

    # 8. ประเมินผล พล็อตกราฟ และเซฟตาราง (ส่งต่อให้ evaluate.py พร้อมรูปวิเคราะห์ 12 เคส)
    eval_metrics = evaluate_model(
        model=model,
        test_dataset=test_ds,
        test_labels=test_labels,
        test_paths=test_paths,
        history1=history1,
        history2=history2,
    )

    # 9. บันทึกผลและภาพทั้งหมดขึ้น WandB
    logger.log_metrics(eval_metrics)
    logger.log_image("loss_curve", RiceConfig.LOSS_PLOT_PATH)
    logger.log_image("scatter_prediction", RiceConfig.SCATTER_PLOT_PATH)
    logger.log_image("error_distribution", RiceConfig.ERROR_DIST_PLOT_PATH)
    logger.log_image("worst_predictions", RiceConfig.WORST_PREDS_PLOT_PATH)
    logger.finish()

    print("\nกระบวนการเทรนและประเมินผลเสร็จสมบูรณ์เรียบร้อยแล้ว!")


if __name__ == "__main__":
    main()

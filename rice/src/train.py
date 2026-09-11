import os
import tensorflow as tf

from common.logger import WandbLogger
from rice.src.callbacks import get_training_callbacks
from rice.src.config import RiceConfig
from rice.src.datasets import create_datasets, create_kfold_datasets
from rice.src.evaluate import (
    evaluate_ensemble,
    evaluate_kfold_summary,
    evaluate_model,
)
from rice.src.metrics import calculate_metrics
from rice.src.model import build_model, unfreeze_for_finetuning


def train_single_fold(
    strategy,
    train_ds,
    val_ds,
    fold_name="",
    best_model_path=RiceConfig.BEST_MODEL_PATH,
    custom_callbacks=None,
):
    """เทรนโมเดล 1 รอบ (Phase 1 Freeze Backbone -> Phase 2 Fine-Tuning) ภายใต้ Strategy"""
    with strategy.scope():
        model, base_model = build_model()
        model.compile(
            optimizer=RiceConfig.PHASE1_OPTIMIZER,
            metrics=["mae"],
            loss=RiceConfig.PHASE1_LOSS,
        )

    callbacks = get_training_callbacks(
        best_model_path=best_model_path,
        custom_callbacks=custom_callbacks,
    )

    prefix = f"[{fold_name}] " if fold_name else ""
    print(f"\n{prefix}Phase 1: Training Top Layers ({RiceConfig.PHASE1_EPOCHS} Epochs)")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=RiceConfig.PHASE1_EPOCHS,
        callbacks=callbacks,
    )

    print(f"\n{prefix}Phase 2: Fine-Tuning Backbone ({RiceConfig.PHASE2_EPOCHS} Epochs, lr={RiceConfig.PHASE2_LR})")
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

    # โหลดน้ำหนักที่ดีที่สุดกลับมา
    if os.path.exists(best_model_path):
        model.load_weights(best_model_path)

    return model, history1, history2


def main():
    print("=" * 60)
    mode_text = f"{RiceConfig.NUM_FOLDS}-Fold Cross-Validation" if RiceConfig.USE_KFOLD else "Single Split"
    print(f"Starting Rice Regression Training (Multi-GPU T4x2) - Mode: {mode_text}")
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
            "use_kfold": RiceConfig.USE_KFOLD,
            "num_folds": RiceConfig.NUM_FOLDS if RiceConfig.USE_KFOLD else 1,
            "phase1_epochs": RiceConfig.PHASE1_EPOCHS,
            "phase2_epochs": RiceConfig.PHASE2_EPOCHS,
            "phase2_lr": RiceConfig.PHASE2_LR,
        },
    )

    if RiceConfig.USE_KFOLD:
        # ========================================================
        # โหมด K-Fold Cross-Validation
        # ========================================================
        fold_datasets, test_ds, test_labels, test_paths = create_kfold_datasets(
            batch_size=global_batch_size
        )

        trained_models = []
        fold_metrics_list = []

        for fold_info in fold_datasets:
            fold_idx = fold_info["fold"]
            print("\n" + "#" * 60)
            print(f"# กำลังดำเนินการเทรน Fold {fold_idx}/{RiceConfig.NUM_FOLDS} (Train: {fold_info['train_count']}, Val: {fold_info['val_count']})")
            print("#" * 60)

            fold_best_path = os.path.join(
                RiceConfig.OUTPUT_DIR, f"best_{RiceConfig.TASK_NAME}_model_fold_{fold_idx}.keras"
            )

            model, h1, h2 = train_single_fold(
                strategy=strategy,
                train_ds=fold_info["train_ds"],
                val_ds=fold_info["val_ds"],
                fold_name=f"Fold {fold_idx}",
                best_model_path=fold_best_path,
                custom_callbacks=logger.get_callbacks(),
            )

            # ประเมินผลเฉพาะ Fold บนชุด Validation ของ Fold นั้นๆ
            val_preds = model.predict(fold_info["val_ds"]).flatten()
            val_metrics = calculate_metrics(fold_info["val_labels"], val_preds)
            val_metrics["fold"] = fold_idx
            fold_metrics_list.append(val_metrics)

            print(
                f"[Fold {fold_idx} Val Score] MAE: {val_metrics['mae']:.2f}g | "
                f"RMSE: {val_metrics['rmse']:.2f} | R²: {val_metrics['r2']:.4f} | "
                f"Acc ±5g: {val_metrics['acc_within_5g']:.1f}%"
            )

            trained_models.append(model)

        # คำนวณ Mean ± Std และบันทึกตารางสรุป K-Fold
        summary_stats = evaluate_kfold_summary(fold_metrics_list)

        # รัน Ensemble ทำนายเฉลี่ยบน Hold-out Test Set
        ensemble_metrics = evaluate_ensemble(
            models=trained_models,
            test_dataset=test_ds,
            test_labels=test_labels,
            test_paths=test_paths,
        )

        # บันทึกผลรวมขึ้น WandB
        logger.log_metrics({
            "mean_val_mae": summary_stats["mean"]["mae"],
            "std_val_mae": summary_stats["std"]["mae"],
            "ensemble_test_mae": ensemble_metrics["mae"],
            "ensemble_test_rmse": ensemble_metrics["rmse"],
            "ensemble_test_r2": ensemble_metrics["r2"],
        })
        logger.log_image("ensemble_scatter", RiceConfig.ENSEMBLE_SCATTER_PATH)
        logger.log_image("ensemble_error_dist", RiceConfig.ENSEMBLE_ERROR_DIST_PATH)
        logger.log_image("ensemble_worst_preds", RiceConfig.ENSEMBLE_WORST_PREDS_PATH)

    else:
        # ========================================================
        # โหมด Single Split ปกติ
        # ========================================================
        train_ds, val_ds, test_ds, test_labels, test_paths = create_datasets(
            batch_size=global_batch_size
        )

        model, history1, history2 = train_single_fold(
            strategy=strategy,
            train_ds=train_ds,
            val_ds=val_ds,
            fold_name="Single Split",
            best_model_path=RiceConfig.BEST_MODEL_PATH,
            custom_callbacks=logger.get_callbacks(),
        )

        model.save(RiceConfig.FINAL_MODEL_PATH)
        print(f"\nบันทึกโมเดลรอบสุดท้ายไปที่: {RiceConfig.FINAL_MODEL_PATH}")
        print(f"บันทึกโมเดลรอบที่ดีที่สุดไปที่: {RiceConfig.BEST_MODEL_PATH}")

        eval_metrics = evaluate_model(
            model=model,
            test_dataset=test_ds,
            test_labels=test_labels,
            test_paths=test_paths,
            history1=history1,
            history2=history2,
        )

        logger.log_metrics(eval_metrics)
        logger.log_image("loss_curve", RiceConfig.LOSS_PLOT_PATH)
        logger.log_image("scatter_prediction", RiceConfig.SCATTER_PLOT_PATH)
        logger.log_image("error_distribution", RiceConfig.ERROR_DIST_PLOT_PATH)
        logger.log_image("worst_predictions", RiceConfig.WORST_PREDS_PLOT_PATH)

    logger.finish()
    print("\nกระบวนการเทรนและประเมินผลเสร็จสมบูรณ์เรียบร้อยแล้ว!")


if __name__ == "__main__":
    main()

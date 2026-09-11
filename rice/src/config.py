import os


class RiceConfig:
    """การตั้งค่า Hyperparameters และพาธต่าง ๆ สำหรับโมเดล Rice Regression"""

    # 1. Task Name
    TASK_NAME = "rice"

    # 2. Image & Data Settings
    IMAGE_SIZE = (224, 224)
    IMAGE_CHANNELS = 3
    BATCH_SIZE_PER_REPLICA = 32
    TEST_SPLIT = 0.1
    VAL_SPLIT = 0.2
    RANDOM_STATE = 42
    USE_KFOLD = True
    NUM_FOLDS = 5

    # 3. Model Architecture
    BACKBONE = "EfficientNetB0"
    WEIGHTS = "imagenet"
    DENSE_UNITS = 64
    DROPOUT_RATE = 0.3

    # 4. Training Phase 1: Top Layers
    PHASE1_EPOCHS = 100
    PHASE1_OPTIMIZER = "adam"
    PHASE1_LOSS = "mean_squared_error"

    # 5. Training Phase 2: Fine-tuning
    PHASE2_EPOCHS = 20
    PHASE2_LR = 1e-5

    # 6. Callbacks Settings
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_FACTOR = 0.2
    REDUCE_LR_PATIENCE = 5
    MIN_LR = 1e-6

    # 7. Output Paths (บน Kaggle ใช้ /kaggle/working เพื่อให้ดาวน์โหลดได้, ในเครื่องเก็บที่ rice/output)
    OUTPUT_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else os.path.join("rice", "output")
    BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, f"best_{TASK_NAME}_model.keras")
    FINAL_MODEL_PATH = os.path.join(OUTPUT_DIR, f"{TASK_NAME}_regression.keras")
    LOSS_PLOT_PATH = os.path.join(OUTPUT_DIR, f"{TASK_NAME}_loss_curve.png")
    SCATTER_PLOT_PATH = os.path.join(OUTPUT_DIR, f"{TASK_NAME}_scatter_prediction.png")
    ERROR_DIST_PLOT_PATH = os.path.join(OUTPUT_DIR, f"{TASK_NAME}_error_distribution.png")
    WORST_PREDS_PLOT_PATH = os.path.join(OUTPUT_DIR, f"{TASK_NAME}_worst_predictions.png")
    METRICS_JSON_PATH = os.path.join(OUTPUT_DIR, f"{TASK_NAME}_metrics_summary.json")
    EVAL_CSV_PATH = os.path.join(OUTPUT_DIR, f"{TASK_NAME}_test_predictions.csv")

    # K-Fold Artifact Paths
    KFOLD_SUMMARY_CSV_PATH = os.path.join(OUTPUT_DIR, f"{TASK_NAME}_kfold_summary.csv")
    KFOLD_METRICS_JSON_PATH = os.path.join(OUTPUT_DIR, f"{TASK_NAME}_kfold_metrics.json")
    ENSEMBLE_PRED_CSV_PATH = os.path.join(OUTPUT_DIR, f"{TASK_NAME}_ensemble_test_predictions.csv")
    ENSEMBLE_SCATTER_PATH = os.path.join(OUTPUT_DIR, f"{TASK_NAME}_ensemble_scatter.png")
    ENSEMBLE_ERROR_DIST_PATH = os.path.join(OUTPUT_DIR, f"{TASK_NAME}_ensemble_error_distribution.png")
    ENSEMBLE_WORST_PREDS_PATH = os.path.join(OUTPUT_DIR, f"{TASK_NAME}_ensemble_worst_predictions.png")

    # 8. WandB Logging
    WANDB_PROJECT = "hospital-ml"
    WANDB_TASK = f"{TASK_NAME}-regression-t4x2"

    # 9. Clinical Nutrition Standards (ตารางคุณค่าโภชนาการข้าวสวยสุก ต่อ 100 กรัม)
    DEFAULT_STANDARD_WEIGHT_G = 150.0  # น้ำหนักข้าวมาตรฐาน 1 ถ้วยของ รพ. (กรัม)
    CALORIES_PER_100G = 130.0          # พลังงาน (kcal / 100g)
    PROTEIN_PER_100G = 2.7             # โปรตีน (กรัม / 100g)
    CARBS_PER_100G = 28.2              # คาร์โบไฮเดรต (กรัม / 100g)
    FAT_PER_100G = 0.3                 # ไขมัน (กรัม / 100g)

import os


class RiceConfig:
    """การตั้งค่า Hyperparameters และพาธต่าง ๆ สำหรับโมเดล Rice Regression"""

    # 1. Image & Data Settings
    IMAGE_SIZE = (224, 224)
    IMAGE_CHANNELS = 3
    BATCH_SIZE_PER_REPLICA = 32
    TEST_SPLIT = 0.1
    VAL_SPLIT = 0.2
    RANDOM_STATE = 42

    # 2. Model Architecture
    BACKBONE = "EfficientNetB0"
    WEIGHTS = "imagenet"
    DENSE_UNITS = 64
    DROPOUT_RATE = 0.3

    # 3. Training Phase 1: Top Layers
    PHASE1_EPOCHS = 100
    PHASE1_OPTIMIZER = "adam"
    PHASE1_LOSS = "mean_squared_error"

    # 4. Training Phase 2: Fine-tuning
    PHASE2_EPOCHS = 20
    PHASE2_LR = 1e-5

    # 5. Callbacks Settings
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_FACTOR = 0.2
    REDUCE_LR_PATIENCE = 5
    MIN_LR = 1e-6

    # 6. Output Paths (บน Kaggle ใช้ /kaggle/working เพื่อให้ดาวน์โหลดได้, ในเครื่องเก็บที่ rice/output)
    OUTPUT_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else os.path.join("rice", "output")
    BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_rice_model.keras")
    FINAL_MODEL_PATH = os.path.join(OUTPUT_DIR, "Rice_regression.keras")
    LOSS_PLOT_PATH = os.path.join(OUTPUT_DIR, "loss_curve.png")
    SCATTER_PLOT_PATH = os.path.join(OUTPUT_DIR, "scatter_prediction.png")
    EVAL_CSV_PATH = os.path.join(OUTPUT_DIR, "test_predictions.csv")

    # 7. WandB Logging
    WANDB_PROJECT = "hospital-ml"
    WANDB_TASK = "rice-regression-t4x2"

    # 8. Clinical Nutrition Standards (ตารางคุณค่าโภชนาการข้าวสวยสุก ต่อ 100 กรัม)
    DEFAULT_STANDARD_WEIGHT_G = 150.0  # น้ำหนักข้าวมาตรฐาน 1 ถ้วยของ รพ. (กรัม)
    CALORIES_PER_100G = 130.0          # พลังงาน (kcal / 100g)
    PROTEIN_PER_100G = 2.7             # โปรตีน (กรัม / 100g)
    CARBS_PER_100G = 28.2              # คาร์โบไฮเดรต (กรัม / 100g)
    FAT_PER_100G = 0.3                 # ไขมัน (กรัม / 100g)

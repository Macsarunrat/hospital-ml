import os
import shutil
import subprocess
from glob import glob

# ============================================================
# 1. เลือกรหัสงานที่ต้องการเทรน (Task Configuration)
#    ตัวเลือก: "rice" | "before" | "after_yolo" | "after_reg"
# ============================================================
TASK = "rice"

print("=" * 60)
print(f"Hospital ML Training Runner - Target Task: '{TASK}'")
print("=" * 60)

# ============================================================
# 2. โหลดความลับ (Secrets) จาก Kaggle Private Dataset อัตโนมัติ
# ============================================================
token = ""
wandb_key = ""

secret_files = glob("/kaggle/input/**/.env", recursive=True)
if secret_files:
    print(f"Loading secrets from: {secret_files[0]}")
    with open(secret_files[0], "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if key == "GITHUB_TOKEN":
                    token = val
                elif key == "WANDB_API_KEY":
                    wandb_key = val
else:
    print("No private secret dataset detected. Running with public settings.")

# ============================================================
# 3. โคลน Repo กลางของทีม
# ============================================================
if token:
    repo_url = f"https://{token}@github.com/Macsarunrat/hospital-ml.git"
else:
    repo_url = "https://github.com/Macsarunrat/hospital-ml.git"

if not os.path.exists("hospital-ml"):
    subprocess.run(["git", "clone", repo_url], check=True)
else:
    subprocess.run(["git", "-C", "hospital-ml", "pull"], check=True)

os.chdir("hospital-ml")

# ตั้งค่าสภาพแวดล้อมให้ทุกโมดูลเห็นค่าเดียวกัน
if wandb_key:
    os.environ["WANDB_API_KEY"] = wandb_key
os.environ["TASK"] = TASK

# ============================================================
# 4. Dynamic Router: เลือกรันโมเดลตามงานที่ระบุ
# ============================================================
try:
    if TASK == "rice":
        print("[Task: Rice] Starting EfficientNetB0 Regression Training...")
        subprocess.run(["python", "-m", "rice.src.train"], check=True)

    elif TASK in ["before", "before_seg"]:
        print("[Task: Before] Starting YOLOv8s-seg Training...")
        script = "before.src.train" if os.path.exists("before/src/train.py") else "src.train_yolo"
        subprocess.run(["python", "-m", script], check=True)

    elif TASK == "after_yolo":
        print("[Task: After (Stage 1)] Starting YOLOv8n Detection Training...")
        script = "after.src.train_yolo" if os.path.exists("after/src/train_yolo.py") else "after.src.train"
        subprocess.run(["python", "-m", script], check=True)

    elif TASK == "after_reg":
        print("[Task: After (Stage 2)] Starting EfficientNetB0 Regression Training...")
        script = "after.src.train_reg" if os.path.exists("after/src/train_reg.py") else "after.src.train"
        subprocess.run(["python", "-m", script], check=True)

    else:
        raise ValueError(
            f"Unknown TASK: '{TASK}'. Supported options: 'rice', 'before', 'after_yolo', 'after_reg'"
        )

finally:
    # ========================================================
    # 5. Auto-Cleanup: ลบโฟลเดอร์ repo ทิ้งก่อน Kaggle แพ็ก Output
    # ========================================================
    working_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else ".."
    os.chdir(working_dir)
    shutil.rmtree("hospital-ml", ignore_errors=True)
    print("Cleaned up temporary repo. Only model weights and evaluation outputs remain.")
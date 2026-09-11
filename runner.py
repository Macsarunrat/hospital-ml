import os
import shutil
import subprocess

# 1. ดึงค่า Config จาก Environment หรือที่ถูก inject มาจาก push.py
token = globals().get("GITHUB_TOKEN", os.getenv("GITHUB_TOKEN", "")).strip()
wandb_key = globals().get("WANDB_API_KEY", os.getenv("WANDB_API_KEY", "")).strip()
task = globals().get("TASK", os.getenv("TASK", "rice")).strip().lower()

print("=" * 60)
print(f"Hospital ML Dynamic Runner - Task: '{task}'")
print("=" * 60)

# 2. โคลน Repo กลางของทีม (รองรับทั้ง Public และ Private)
if token:
    repo_url = f"https://{token}@github.com/Macsarunrat/hospital-ml.git"
else:
    repo_url = "https://github.com/Macsarunrat/hospital-ml.git"

if not os.path.exists("hospital-ml"):
    subprocess.run(["git", "clone", repo_url], check=True)
else:
    subprocess.run(["git", "-C", "hospital-ml", "pull"], check=True)

os.chdir("hospital-ml")

# 3. ตั้งค่าสภาพแวดล้อมให้ทุกโมดูลเห็นค่าเดียวกัน
if wandb_key:
    os.environ["WANDB_API_KEY"] = wandb_key
os.environ["TASK"] = task

# 4. Dynamic Router: สลับ Engine ระหว่าง TensorFlow และ Ultralytics YOLO
try:
    if task == "rice":
        print("🍚 [Task: Rice] กำลังเทรนโมเดลน้ำหนักข้าว (EfficientNetB0 Regression)...")
        subprocess.run(["python", "-m", "rice.src.train"], check=True)

    elif task in ["before", "before_seg"]:
        print("🍽️ [Task: Before] กำลังเทรน YOLOv8s-seg เต็มถาดอาหาร...")
        script = "before.src.train" if os.path.exists("before/src/train.py") else "src.train_yolo"
        subprocess.run(["python", "-m", script], check=True)

    elif task == "after_yolo":
        print("🥣 [Task: After (Stage 1)] กำลังเทรน YOLOv8n ครอบถ้วยอาหาร...")
        script = "after.src.train_yolo" if os.path.exists("after/src/train_yolo.py") else "after.src.train"
        subprocess.run(["python", "-m", script], check=True)

    elif task == "after_reg":
        print("🥣 [Task: After (Stage 2)] กำลังเทรน EfficientNetB0 Regression ถ้วยหลังทาน...")
        script = "after.src.train_reg" if os.path.exists("after/src/train_reg.py") else "after.src.train"
        subprocess.run(["python", "-m", script], check=True)

    else:
        raise ValueError(
            f"❌ ไม่รู้จัก TASK: '{task}' (ตัวเลือกที่รองรับ: 'rice', 'before', 'after_yolo', 'after_reg')"
        )

finally:
    # 5. Auto-Cleanup: ลบโฟลเดอร์ git repo ทิ้งก่อนที่ Kaggle จะแพ็ก Output
    working_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else ".."
    os.chdir(working_dir)
    shutil.rmtree("hospital-ml", ignore_errors=True)
    print("🧹 ทำความสะอาดโฟลเดอร์ชั่วคราวเรียบร้อย ผลลัพธ์พร้อมสำหรับดาวน์โหลด!")
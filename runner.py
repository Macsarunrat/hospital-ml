import os
import subprocess

# 1. โคลน Repo กลางของทีม
repo_url = "https://github.com/Macsarunrat/hospital-ml.git"

if not os.path.exists("hospital-ml"):
    subprocess.run(["git", "clone", repo_url], check=True)

os.chdir("hospital-ml")

# 2. ติดตั้ง Dependencies (หากต้องการ)
# subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)

# 3. เลือกคำสั่งเทรน (เอาคอมเมนต์ # ออกในบรรทัดที่ต้องการรัน)

# แบบ Rice Regression (TensorFlow Multi-GPU / T4 x 2):
# หมายเหตุ: TensorFlow จะตรวจพบ 2x T4 อัตโนมัติและใช้ MirroredStrategy จัดการ ไม่ต้องใช้ torchrun
subprocess.run(["python", "-m", "rice.src.train"], check=True)

# แบบ YOLO (ถ้ามี):
# subprocess.run(["python", "src/train_yolo.py"], check=True)
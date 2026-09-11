import os
import subprocess

# ดึงค่า Token และ Key (จะถูก inject อัตโนมัติจาก .env ตอนสั่ง python push.py)
token = globals().get("GITHUB_TOKEN", "").strip()
wandb_key = globals().get("WANDB_API_KEY", "").strip()

# 1. โคลน Repo กลางของทีม (รองรับทั้ง Public และ Private)
if token:
    repo_url = f"https://{token}@github.com/Macsarunrat/hospital-ml.git"
else:
    repo_url = "https://github.com/Macsarunrat/hospital-ml.git"

if not os.path.exists("hospital-ml"):
    subprocess.run(["git", "clone", repo_url], check=True)
else:
    subprocess.run(["git", "-C", "hospital-ml", "pull"], check=True)

os.chdir("hospital-ml")

# 2. ตั้งค่า WandB Key ให้ระบบมองเห็น
if wandb_key:
    os.environ["WANDB_API_KEY"] = wandb_key

# 3. รันเทรน Rice Regression บน GPU T4 x 2
subprocess.run(["python", "-m", "rice.src.train"], check=True)
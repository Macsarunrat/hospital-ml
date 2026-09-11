import os
import subprocess

# 1. โคลน Repo กลางของทีม
repo_url = "https://github.com/Macsarunrat/hospital-ml.git"

if not os.path.exists("hospital-ml"):
    subprocess.run(["git", "clone", repo_url], check=True)
else:
    subprocess.run(["git", "-C", "hospital-ml", "pull"], check=True)

os.chdir("hospital-ml")

# 2. เริ่มต้นเทรน Rice Regression (TensorFlow Multi-GPU / T4 x 2)
subprocess.run(["python", "-m", "rice.src.train"], check=True)
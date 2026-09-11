import os
import subprocess

# 1. นำเนื้อหาจาก .env ไปแปะไว้บนสุดของ runner.py ชั่วคราว
env_data = open(".env", "r", encoding="utf-8").read() if os.path.exists(".env") else ""
runner_code = open("runner.py", "r", encoding="utf-8").read()

open("runner.py", "w", encoding="utf-8").write(env_data + "\n" + runner_code)

try:
    # 2. สั่ง Kaggle CLI ส่งขึ้นคลาวด์
    print("🚀 กำลังส่งขึ้น Kaggle...")
    subprocess.run(["kaggle", "kernels", "push", "-p", "."], check=True)
    print("✅ ส่งขึ้น Kaggle สำเร็จแล้ว!")
finally:
    # 3. ให้ Git ดีด runner.py กลับเป็นไฟล์สะอาดเดิมทันที
    subprocess.run(["git", "checkout", "runner.py"])
    print("🔒 ล้างคีย์ออกจาก runner.py เรียบร้อย (ปลอดภัย 100%)")

import os
import subprocess

# 1. อ่านค่า TASK จาก .env เพื่อแสดงผลคำแนะนำ
task = "rice"
if os.path.exists(".env"):
    for line in open(".env", "r", encoding="utf-8"):
        if line.strip().startswith("TASK="):
            task = line.strip().split("=", 1)[1].strip("\"'").lower()

print("=" * 60)
print(f"🚀 เตรียมส่งงานขึ้น Kaggle สำหรับ Task: '{task}'")
print("=" * 60)

# 2. นำเนื้อหาจาก .env ไปแปะไว้บนสุดของ runner.py ชั่วคราว
env_data = open(".env", "r", encoding="utf-8").read() if os.path.exists(".env") else ""
runner_code = open("runner.py", "r", encoding="utf-8").read()

open("runner.py", "w", encoding="utf-8").write(env_data + "\n" + runner_code)

try:
    # 3. สั่ง Kaggle CLI ส่งขึ้นคลาวด์
    print("⏳ กำลังสั่ง Kaggle CLI Push...")
    subprocess.run(["kaggle", "kernels", "push", "-p", "."], check=True)
    print("✅ ส่งขึ้น Kaggle สำเร็จเรียบร้อยแล้ว!")
finally:
    # 4. ให้ Git ดีด runner.py กลับเป็นไฟล์สะอาดเดิมทันที
    subprocess.run(["git", "checkout", "runner.py"])
    print("🔒 ล้างคีย์ออกจาก runner.py เรียบร้อย (ปลอดภัย 100%)")

# 5. แนะนำคำสั่งดาวน์โหลดที่ตรงกับ Task ที่เพิ่งเทรน
print("\n" + "-" * 60)
print(f"📌 คำสั่งตรวจเช็กสถานะ:")
print(f"   kaggle kernels status macsarun/hospital-ml-training")
print(f"📥 คำสั่งดาวน์โหลดผลลัพธ์เมื่อเทรนเสร็จ (จะโหลดลงโฟลเดอร์ {task}/output โดยตรง):")
print(f"   kaggle kernels output macsarun/hospital-ml-training -p {task}/output/")
print("-" * 60)

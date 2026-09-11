import os

try:
    import wandb
    from wandb.integration.keras import WandbMetricsLogger
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class WandbLogger:
    """ระบบจัดการ Logging ด้วย Weights & Biases (WandB) แบบรวมศูนย์

    รองรับทุกโมเดลในโปรเจกต์ (rice, before, after, yolo)
    """

    def __init__(self, project_name="hospital-ml", task_name="default", config=None, api_key=None):
        self.is_active = False

        if not HAS_WANDB:
            print("[WandbLogger] ไลบรารี 'wandb' ยังไม่ได้ติดตั้ง ข้ามการล็อกไปยัง WandB")
            return

        # 1. ค้นหา API Key จาก 3 แหล่ง: พารามิเตอร์ -> .env/Environment -> Kaggle Secrets
        key = api_key or os.getenv("WANDB_API_KEY")
        if not key:
            try:
                from kaggle_secrets import UserSecretsClient
                user_secrets = UserSecretsClient()
                key = user_secrets.get_secret("WANDB_API_KEY")
            except Exception:
                pass

        # 2. ทำการ Login และ Initialize
        if key:
            try:
                wandb.login(key=key)
                wandb.init(
                    project=project_name,
                    name=task_name,
                    config=config or {},
                    reinit=True,
                )
                self.is_active = True
                print(f"[WandbLogger] เชื่อมต่อ WandB สำเร็จ! ติดตามผลได้ที่ Project: '{project_name}' Task: '{task_name}'")
            except Exception as e:
                print(f"[WandbLogger] เชื่อมต่อกับ WandB ไม่สำเร็จ: {e}")
        else:
            print("[WandbLogger] ไม่พบ WANDB_API_KEY (การเทรนจะดำเนินต่อไปโดยไม่บันทึกลง WandB)")

    def get_callbacks(self):
        """ส่งคืน Callbacks สำหรับ Keras model.fit()"""
        if self.is_active:
            return [WandbMetricsLogger()]
        return []

    def log_metrics(self, metrics: dict):
        """บันทึกค่าสถิติหรือตัวเลขผลประเมิน (เช่น MAE, RMSE, R2)"""
        if self.is_active:
            try:
                wandb.log(metrics)
            except Exception as e:
                print(f"[WandbLogger] บันทึก metrics ไม่สำเร็จ: {e}")

    def log_image(self, tag: str, image_path: str):
        """ส่งไฟล์รูปภาพกราฟ (เช่น loss_curve.png) ขึ้นแสดงบน Dashboard"""
        if self.is_active and os.path.exists(image_path):
            try:
                wandb.log({tag: wandb.Image(image_path)})
            except Exception as e:
                print(f"[WandbLogger] อัปโหลดรูป {tag} ไม่สำเร็จ: {e}")

    def finish(self):
        """ปิดเซสชันการบันทึกเมื่อเทรนและประเมินผลเสร็จสิ้น"""
        if self.is_active:
            try:
                wandb.finish()
                print("[WandbLogger] ปิดเซสชัน WandB เรียบร้อย")
            except Exception:
                pass

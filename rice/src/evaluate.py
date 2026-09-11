import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rice.src.metrics import calculate_metrics
from rice.src.config import RiceConfig


def plot_loss_curve(history1=None, history2=None, save_path=None):
    """วาดและบันทึกกราฟ Loss ตลอดทั้งสอง Phase"""
    save_path = save_path or RiceConfig.LOSS_PLOT_PATH

    loss1 = history1.history.get("loss", []) if history1 else []
    val_loss1 = history1.history.get("val_loss", []) if history1 else []

    loss2 = history2.history.get("loss", []) if history2 else []
    val_loss2 = history2.history.get("val_loss", []) if history2 else []

    total_train_loss = loss1 + loss2
    total_val_loss = val_loss1 + val_loss2

    if not total_train_loss:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(total_train_loss, "b-", label="Train Loss")
    plt.plot(total_val_loss, "r-", label="Val Loss")
    plt.title("Loss Curve (Phase 1 + Phase 2)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"[Evaluate] บันทึกกราฟ Loss Curve ไปที่: {save_path}")


def plot_scatter_prediction(actual_labels, predictions, save_path=None):
    """วาดและบันทึกกราฟ Scatter Plot เทียบ Actual vs Predicted"""
    save_path = save_path or RiceConfig.SCATTER_PLOT_PATH

    actual = np.array(actual_labels)
    pred = np.array(predictions)

    plt.figure(figsize=(7, 7))
    plt.scatter(actual, pred, color="blue", alpha=0.6, label="Predictions")

    min_val = min(min(actual), min(pred))
    max_val = max(max(actual), max(pred))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal 45°")

    plt.title("Actual vs Predicted Rice Weight")
    plt.xlabel("Actual Weight (grams)")
    plt.ylabel("Predicted Weight (grams)")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"[Evaluate] บันทึกกราฟ Scatter Plot ไปที่: {save_path}")


def save_prediction_table(actual_labels, predictions, save_path=None):
    """บันทึกตารางผลการทายรายภาพเป็น CSV"""
    save_path = save_path or RiceConfig.EVAL_CSV_PATH
    actual = np.array(actual_labels)
    pred = np.array(predictions)

    result_df = pd.DataFrame({
        "Actual_Weight_g": actual,
        "Predicted_Weight_g": np.round(pred, 2),
        "Error_g": np.round(np.abs(pred - actual), 2),
    })
    result_df.to_csv(save_path, index=False)
    print(f"[Evaluate] บันทึกตารางผลลัพธ์ CSV ไปที่: {save_path}")


def evaluate_model(model, test_dataset, test_labels, history1=None, history2=None):
    """รันการวัดผลโมเดล คำนวณสถิติ บันทึกกราฟ และสร้างตาราง CSV"""
    print("\n" + "=" * 50)
    print("เริ่มการ Evaluate บน Test Dataset")
    print("=" * 50)

    predictions = model.predict(test_dataset).flatten()
    actual = np.array(test_labels)

    # 1. คำนวณ Metrics
    metrics = calculate_metrics(actual, predictions)
    print("ผลการประเมินความแม่นยำ:")
    print(f"  - MAE  (คลาดเคลื่อนเฉลี่ย) : {metrics['mae']:.2f} กรัม")
    print(f"  - RMSE (ความแม่นยำรวม)   : {metrics['rmse']:.2f}")
    print(f"  - R²   (ความสัมพันธ์)     : {metrics['r2']:.4f}")
    print(f"  - แม่นยำในกรอบ ±5g      : {metrics['acc_within_5g']:.1f} %")
    print(f"  - แม่นยำในกรอบ ±10g     : {metrics['acc_within_10g']:.1f} %")

    # 2. บันทึกรูปกราฟและตาราง
    try:
        plot_loss_curve(history1, history2)
        plot_scatter_prediction(actual, predictions)
        save_prediction_table(actual, predictions)
    except Exception as e:
        print(f"[Evaluate] ไม่สามารถบันทึกกราฟได้: {e}")

    return metrics

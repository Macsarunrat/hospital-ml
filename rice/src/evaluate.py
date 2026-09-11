import json
import os
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rice.src.config import RiceConfig
from rice.src.metrics import calculate_metrics


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
    plt.savefig(save_path, bbox_inches="tight")
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
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Evaluate] บันทึกกราฟ Scatter Plot ไปที่: {save_path}")


def plot_error_distribution(actual_labels, predictions, save_path=None):
    """หัวข้อที่ 1: วาดกราฟการกระจายตัวของความคลาดเคลื่อน (Error Distribution Histogram)

    แสดงให้เห็นว่าโมเดลมี Bias ทายเกินหรือทายขาดหรือไม่
    """
    save_path = save_path or RiceConfig.ERROR_DIST_PLOT_PATH
    actual = np.array(actual_labels)
    pred = np.array(predictions)
    residuals = pred - actual  # ค่ายิ่งใกล้ 0 ยิ่งดี

    plt.figure(figsize=(8, 5))
    n, bins, patches = plt.hist(
        residuals, bins=25, color="skyblue", edgecolor="black", alpha=0.7, density=True
    )

    # วาดเส้นปกติ (Mean & Zero Line)
    mean_err = np.mean(residuals)
    std_err = np.std(residuals)
    plt.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero Error (Ideal)")
    plt.axvline(mean_err, color="green", linestyle="-", linewidth=2, label=f"Mean Error: {mean_err:+.2f}g")

    # แรเงาช่วงที่ยอมรับได้ในทางคลินิก (±5g, ±10g)
    plt.axvspan(-5, 5, color="green", alpha=0.1, label="Clinically Safe (±5g)")
    plt.axvspan(-10, 10, color="yellow", alpha=0.05, label="Acceptable (±10g)")

    plt.title("Error Distribution (Residuals = Predicted - Actual)")
    plt.xlabel("Error (grams)  [<0 = Underestimate, >0 = Overestimate]")
    plt.ylabel("Density")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Evaluate] บันทึกกราฟ Error Distribution ไปที่: {save_path}")


def save_metrics_summary(metrics, save_path=None):
    """หัวข้อที่ 2: บันทึกไฟล์สรุปตัวเลขสถิติทั้งหมดเป็น JSON"""
    save_path = save_path or RiceConfig.METRICS_JSON_PATH
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f"[Evaluate] บันทึกไฟล์สรุปสถิติ JSON ไปที่: {save_path}")


def plot_worst_predictions(test_paths, actual_labels, predictions, save_path=None, top_k=12):
    """หัวข้อที่ 3: รวมภาพเคสที่ AI ทายพลาดมากที่สุด (Top-K Worst Cases) แสดงเป็น Grid หลาย ๆ รูป

    เพื่อใช้ทำ Error Analysis หาสาเหตุว่าทำไมถึงทายรูปกลุ่มนี้ผิด
    """
    if not test_paths:
        return

    save_path = save_path or RiceConfig.WORST_PREDS_PLOT_PATH
    actual = np.array(actual_labels)
    pred = np.array(predictions)
    errors = np.abs(pred - actual)

    # จัดอันดับภาพที่ Error สูงสุด
    worst_indices = np.argsort(errors)[::-1][:top_k]

    # กำหนดขนาดตารางกริด (เช่น 12 รูป = 3 แถว x 4 คอลัมน์)
    cols = 4
    rows = int(np.ceil(len(worst_indices) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()

    for idx, sample_idx in enumerate(worst_indices):
        img_path = test_paths[sample_idx]
        act = actual[sample_idx]
        pr = pred[sample_idx]
        err = errors[sample_idx]

        try:
            img = Image.open(img_path)
            axes[idx].imshow(img)
            axes[idx].set_title(
                f"Rank #{idx+1} Worst\nActual: {act:.1f}g | Pred: {pr:.1f}g\nError: {err:.1f}g",
                fontsize=10,
                color="darkred",
                fontweight="bold",
            )
        except Exception:
            axes[idx].text(0.5, 0.5, "Image Error", ha="center")

        axes[idx].axis("off")

    # ปิดแกนที่เหลือ (ถ้ามี)
    for j in range(len(worst_indices), len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        f"Top {len(worst_indices)} Worst Predictions (Error Analysis)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Evaluate] บันทึกภาพเคสที่ทายพลาดมากที่สุด {len(worst_indices)} รูป ไปที่: {save_path}")


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


def evaluate_model(model, test_dataset, test_labels, test_paths=None, history1=None, history2=None):
    """รันการวัดผลโมเดล คำนวณสถิติ บันทึกกราฟ ตาราง และภาพเคสผิดพลาด"""
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

    # 2. บันทึกรูปกราฟและตารางทั้งหมด
    try:
        plot_loss_curve(history1, history2)
        plot_scatter_prediction(actual, predictions)
        plot_error_distribution(actual, predictions)             # 👈 หัวข้อ 1: Error Distribution
        save_metrics_summary(metrics)                            # 👈 หัวข้อ 2: Metrics JSON
        if test_paths:
            plot_worst_predictions(test_paths, actual, predictions, top_k=12)  # 👈 หัวข้อ 3: ตาราง 12 รูปทายพลาดสุด
        save_prediction_table(actual, predictions)
    except Exception as e:
        print(f"[Evaluate] คำเตือน: เกิดข้อผิดพลาดในการบันทึกภาพบางส่วน: {e}")

    return metrics

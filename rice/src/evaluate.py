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
        plot_error_distribution(actual, predictions)
        save_metrics_summary(metrics)
        if test_paths:
            plot_worst_predictions(test_paths, actual, predictions, top_k=12)
        save_prediction_table(actual, predictions)
    except Exception as e:
        print(f"[Evaluate] คำเตือน: เกิดข้อผิดพลาดในการบันทึกภาพบางส่วน: {e}")

    return metrics


def evaluate_kfold_summary(fold_metrics_list):
    """คำนวณ Mean และ Std ของแต่ละ Metric จากทุก Fold และบันทึกเป็นตาราง CSV และ JSON"""
    print("\n" + "=" * 60)
    print("สรุปผลการประเมิน K-Fold Cross-Validation (Mean ± Std)")
    print("=" * 60)

    # แปลง List of dicts เป็น DataFrame
    df = pd.DataFrame(fold_metrics_list)
    metric_cols = ["mae", "rmse", "r2", "acc_within_5g", "acc_within_10g"]

    summary_data = []
    summary_json = {
        "folds": fold_metrics_list,
        "mean": {},
        "std": {},
    }

    print(f"{'Fold':<8}{'MAE (g)':<12}{'RMSE (g)':<12}{'R²':<10}{'Acc ±5g (%)':<15}{'Acc ±10g (%)':<15}")
    print("-" * 72)

    for _, row in df.iterrows():
        f_idx = int(row["fold"]) if "fold" in row else _ + 1
        print(
            f"Fold {f_idx:<3} "
            f"{row['mae']:<12.2f}"
            f"{row['rmse']:<12.2f}"
            f"{row['r2']:<10.4f}"
            f"{row['acc_within_5g']:<15.1f}"
            f"{row['acc_within_10g']:<15.1f}"
        )

    print("-" * 72)
    mean_series = df[metric_cols].mean()
    std_series = df[metric_cols].std()

    print(
        f"{'Mean':<8}"
        f"{mean_series['mae']:<12.2f}"
        f"{mean_series['rmse']:<12.2f}"
        f"{mean_series['r2']:<10.4f}"
        f"{mean_series['acc_within_5g']:<15.1f}"
        f"{mean_series['acc_within_10g']:<15.1f}"
    )
    print(
        f"{'Std':<8}"
        f"±{std_series['mae']:<11.2f}"
        f"±{std_series['rmse']:<11.2f}"
        f"±{std_series['r2']:<9.4f}"
        f"±{std_series['acc_within_5g']:<14.1f}"
        f"±{std_series['acc_within_10g']:<14.1f}"
    )
    print("=" * 72)

    # บันทึกเป็น CSV
    df.to_csv(RiceConfig.KFOLD_SUMMARY_CSV_PATH, index=False)
    print(f"[Evaluate] บันทึกตารางสรุป K-Fold CSV ไปที่: {RiceConfig.KFOLD_SUMMARY_CSV_PATH}")

    # บันทึก JSON
    for col in metric_cols:
        summary_json["mean"][col] = round(float(mean_series[col]), 4)
        summary_json["std"][col] = round(float(std_series[col]), 4)

    with open(RiceConfig.KFOLD_METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=4, ensure_ascii=False)
    print(f"[Evaluate] บันทึกสรุปสถิติ K-Fold JSON ไปที่: {RiceConfig.KFOLD_METRICS_JSON_PATH}")

    return summary_json


def evaluate_ensemble(models, test_dataset, test_labels, test_paths=None):
    """รวมพลังทำนาย (Ensemble Mean Prediction) จากทุก Fold บน Hold-out Test Set"""
    print("\n" + "=" * 60)
    print(f"เริ่มการทำ Ensemble Inference จากโมเดลทั้งหมด {len(models)} ตัว บน Test Set")
    print("=" * 60)

    all_preds = []
    for idx, model in enumerate(models):
        preds = model.predict(test_dataset).flatten()
        all_preds.append(preds)
        print(f"  - Model Fold {idx+1} ทำนายเสร็จสิ้น")

    # หาค่าเฉลี่ยของผลทำนาย (Ensemble Average)
    ensemble_preds = np.mean(all_preds, axis=0)
    actual = np.array(test_labels)

    # คำนวณ Metrics รวม
    metrics = calculate_metrics(actual, ensemble_preds)
    print("\nผลการประเมิน Ensemble (Model Averaging):")
    print(f"  - Ensemble MAE  : {metrics['mae']:.2f} กรัม")
    print(f"  - Ensemble RMSE : {metrics['rmse']:.2f}")
    print(f"  - Ensemble R²   : {metrics['r2']:.4f}")
    print(f"  - แม่นยำในกรอบ ±5g  : {metrics['acc_within_5g']:.1f} %")
    print(f"  - แม่นยำในกรอบ ±10g : {metrics['acc_within_10g']:.1f} %")

    # บันทึกกราฟและตาราง Ensemble
    try:
        plot_scatter_prediction(actual, ensemble_preds, save_path=RiceConfig.ENSEMBLE_SCATTER_PATH)
        plot_error_distribution(actual, ensemble_preds, save_path=RiceConfig.ENSEMBLE_ERROR_DIST_PATH)
        if test_paths:
            plot_worst_predictions(
                test_paths, actual, ensemble_preds, top_k=12, save_path=RiceConfig.ENSEMBLE_WORST_PREDS_PATH
            )
        save_prediction_table(actual, ensemble_preds, save_path=RiceConfig.ENSEMBLE_PRED_CSV_PATH)
    except Exception as e:
        print(f"[Evaluate] คำเตือน: เกิดข้อผิดพลาดในการบันทึกผล Ensemble: {e}")

    return metrics

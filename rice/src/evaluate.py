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


def resolve_image_path(img_path):
    """ตรวจสอบและค้นหาตำแหน่งภาพอัตโนมัติหาก path เดิมไม่ตรง (ป้องกัน Kaggle path mismatch)"""
    if os.path.exists(img_path):
        return img_path

    filename = os.path.basename(img_path)
    search_dirs = [
        "/kaggle/input/datasets/macsarun/cropped/bowl",
        "/kaggle/input/cropped/bowl",
        "/kaggle/input/Cropped/bowl",
        "/kaggle/input",
        "rice/output",
    ]
    for directory in search_dirs:
        candidate = os.path.join(directory, filename)
        if os.path.exists(candidate):
            return candidate

    return img_path


def plot_worst_predictions(test_paths, actual_labels, predictions, save_path=None, top_k=12):
    """หัวข้อที่ 3: รวมภาพเคสที่ AI ทายพลาดมากที่สุด (Top-K Worst Cases) แสดงเป็น Grid หลาย ๆ รูป

    เพื่อใช้ทำ Error Analysis หาสาเหตุว่าทำไมถึงทายรูปกลุ่มนี้ผิด
    """
    if test_paths is None or len(test_paths) == 0:
        print("[Evaluate] คำเตือน: ไม่พบ test_paths ข้ามการบันทึก worst predictions")
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
    if hasattr(axes, "flatten"):
        axes = axes.flatten()
    else:
        axes = [axes]

    for idx, sample_idx in enumerate(worst_indices):
        raw_path = str(test_paths[sample_idx])
        img_path = resolve_image_path(raw_path)
        act = actual[sample_idx]
        pr = pred[sample_idx]
        err = errors[sample_idx]

        try:
            if os.path.exists(img_path):
                img = Image.open(img_path)
                axes[idx].imshow(img)
            else:
                axes[idx].text(0.5, 0.5, f"Image not found:\n{os.path.basename(img_path)}", ha="center", va="center")
            axes[idx].set_title(
                f"Rank #{idx+1} Worst\nActual: {act:.1f}g | Pred: {pr:.1f}g\nError: {err:.1f}g",
                fontsize=10,
                color="darkred",
                fontweight="bold",
            )
        except Exception as e:
            axes[idx].text(0.5, 0.5, f"Error: {e}", ha="center", va="center")

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
        if test_paths is not None and len(test_paths) > 0:
            plot_worst_predictions(test_paths, actual, predictions, top_k=12)
            plot_visual_gallery(test_paths, actual, predictions, n_samples=16)
        save_prediction_table(actual, predictions)
        stratified_df = calculate_stratified_errors(actual, predictions)
        plot_stratified_bars(stratified_df)
    except Exception as e:
        print(f"[Evaluate] คำเตือน: เกิดข้อผิดพลาดในการบันทึกภาพบางส่วน: {e}")

    return metrics


def calculate_stratified_errors(actual_labels, predictions, bins=None, labels=None, save_path=None):
    """วิเคราะห์ความแม่นยำตามช่วงน้ำหนัก (Stratified Error by Weight Bin)"""
    save_path = save_path or RiceConfig.STRATIFIED_CSV_PATH
    actual = np.array(actual_labels)
    pred = np.array(predictions)
    errors = np.abs(pred - actual)

    if bins is None:
        bins = [0, 30, 70, 110, 300]
    if labels is None:
        labels = ["0 - 30g (เหลือน้อย)", "31 - 70g (ปานกลาง)", "71 - 110g (ค่อนข้างเยอะ)", "> 110g (เกือบเต็ม)"]

    bin_assignments = pd.cut(actual, bins=bins, labels=labels, right=True, include_lowest=True)

    records = []
    print("\n" + "-" * 75)
    print(f"{'Weight Bin':<26}{'Samples':<10}{'MAE (g)':<12}{'RMSE (g)':<12}{'Acc ±5g (%)':<15}")
    print("-" * 75)

    for lab in labels:
        mask = bin_assignments == lab
        count = np.sum(mask)
        if count > 0:
            act_sub = actual[mask]
            pred_sub = pred[mask]
            err_sub = errors[mask]

            mae = np.mean(err_sub)
            rmse = np.sqrt(np.mean((pred_sub - act_sub) ** 2))
            acc_5g = np.mean(err_sub <= 5.0) * 100
            acc_10g = np.mean(err_sub <= 10.0) * 100
            mape = np.mean(err_sub / np.maximum(act_sub, 1.0)) * 100

            records.append({
                "weight_bin": lab,
                "sample_count": int(count),
                "mae": round(float(mae), 2),
                "rmse": round(float(rmse), 2),
                "mape": round(float(mape), 2),
                "acc_within_5g": round(float(acc_5g), 2),
                "acc_within_10g": round(float(acc_10g), 2),
            })
            print(f"{lab:<26}{count:<10}{mae:<12.2f}{rmse:<12.2f}{acc_5g:<15.1f}")
        else:
            records.append({
                "weight_bin": lab,
                "sample_count": 0,
                "mae": 0.0,
                "rmse": 0.0,
                "mape": 0.0,
                "acc_within_5g": 0.0,
                "acc_within_10g": 0.0,
            })

    print("-" * 75)
    df_stratified = pd.DataFrame(records)
    df_stratified.to_csv(save_path, index=False)
    print(f"[Evaluate] บันทึกตาราง Stratified Error ไปที่: {save_path}")
    return df_stratified


def plot_stratified_bars(stratified_df, save_path=None):
    """พล็อตกราฟแท่งเปรียบเทียบ MAE และความแม่นยำแบ่งตามช่วงน้ำหนัก"""
    save_path = save_path or RiceConfig.STRATIFIED_PLOT_PATH
    if stratified_df.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bins = stratified_df["weight_bin"]
    mae = stratified_df["mae"]
    acc_5g = stratified_df["acc_within_5g"]
    acc_10g = stratified_df["acc_within_10g"]

    # กราฟที่ 1: MAE by Bin
    bars1 = ax1.bar(bins, mae, color="#4C72B0", width=0.5)
    ax1.set_title("Mean Absolute Error (MAE) by Weight Range", fontsize=12, fontweight="bold")
    ax1.set_ylabel("MAE (grams)", fontsize=10)
    ax1.tick_params(axis="x", rotation=15)
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, f"{yval:.1f}g", ha="center", va="bottom", fontsize=9)

    # กราฟที่ 2: Accuracy within ±5g and ±10g
    x = np.arange(len(bins))
    width = 0.35
    bars2_1 = ax2.bar(x - width / 2, acc_5g, width, label="Acc within ±5g", color="#55A868")
    bars2_2 = ax2.bar(x + width / 2, acc_10g, width, label="Acc within ±10g", color="#C44E52")
    ax2.set_title("Clinical Accuracy (%) by Weight Range", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bins, rotation=15)
    ax2.set_ylim(0, 105)
    ax2.legend(loc="lower right")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Evaluate] บันทึกกราฟ Stratified Error Bars ไปที่: {save_path}")


def plot_visual_gallery(test_paths, actual_labels, predictions, save_path=None, n_samples=16):
    """รวมภาพผลการทำนายจริง (Clean Grid 4x4) แสดง Actual, Predicted และ Error โดยไม่มีกรอบสี"""
    if test_paths is None or len(test_paths) == 0:
        return

    save_path = save_path or RiceConfig.VISUAL_GALLERY_PATH
    actual = np.array(actual_labels)
    pred = np.array(predictions)
    errors = np.abs(pred - actual)

    total_items = len(test_paths)
    step = max(1, total_items // n_samples)
    selected_indices = list(range(0, total_items, step))[:n_samples]

    cols = 4
    rows = int(np.ceil(len(selected_indices) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    if hasattr(axes, "flatten"):
        axes = axes.flatten()
    else:
        axes = [axes]

    for idx, sample_idx in enumerate(selected_indices):
        raw_path = str(test_paths[sample_idx])
        img_path = resolve_image_path(raw_path)
        act = actual[sample_idx]
        pr = pred[sample_idx]
        err = errors[sample_idx]

        try:
            if os.path.exists(img_path):
                img = Image.open(img_path)
                axes[idx].imshow(img)
            else:
                axes[idx].text(0.5, 0.5, f"Image not found:\n{os.path.basename(img_path)}", ha="center", va="center")
            axes[idx].set_title(
                f"Actual: {act:.1f}g | Pred: {pr:.1f}g\nError: {err:.1f}g",
                fontsize=10,
                color="#222222",
                fontweight="normal",
            )
        except Exception as e:
            axes[idx].text(0.5, 0.5, f"Error: {e}", ha="center", va="center")

        axes[idx].axis("off")

    for j in range(len(selected_indices), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Sample Predictions on Test Set (Clean Gallery)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Evaluate] บันทึกภาพ Gallery ผลทำนายจริง {len(selected_indices)} รูป ไปที่: {save_path}")


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
        if test_paths is not None and len(test_paths) > 0:
            plot_worst_predictions(
                test_paths, actual, ensemble_preds, top_k=12, save_path=RiceConfig.ENSEMBLE_WORST_PREDS_PATH
            )
            plot_visual_gallery(
                test_paths, actual, ensemble_preds, n_samples=16, save_path=RiceConfig.VISUAL_GALLERY_PATH
            )
        save_prediction_table(actual, ensemble_preds, save_path=RiceConfig.ENSEMBLE_PRED_CSV_PATH)
        stratified_df = calculate_stratified_errors(actual, ensemble_preds)
        plot_stratified_bars(stratified_df)
    except Exception as e:
        print(f"[Evaluate] คำเตือน: เกิดข้อผิดพลาดในการบันทึกผล Ensemble: {e}")

    return metrics


def save_training_summary(extra_info=None, txt_path=None, json_path=None):
    """บันทึกไฟล์สรุปรายละเอียดการทดลองทั้งหมด (training_config_summary.txt และ JSON)

    ช่วยให้สามารถตรวจสอบย้อนหลังได้ 100% ว่าโมเดลนี้ใช้อะไรเทรน ตั้งค่า Hyperparameters ไว้อย่างไร
    """
    import datetime
    import platform
    import tensorflow as tf

    txt_path = txt_path or RiceConfig.TRAINING_SUMMARY_TXT_PATH
    json_path = json_path or RiceConfig.EXPERIMENT_MANIFEST_JSON_PATH

    info = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "task_name": RiceConfig.TASK_NAME,
        "backbone_model": RiceConfig.BACKBONE,
        "pretrained_weights": RiceConfig.WEIGHTS,
        "image_size": list(RiceConfig.get_image_size()),
        "image_channels": RiceConfig.IMAGE_CHANNELS,
        "image_preprocessing": "tf.image.resize_with_pad (Preserves Aspect Ratio with Zero Padding)",
        "pooling_layer": "GlobalAveragePooling2D",
        "dense_units": RiceConfig.DENSE_UNITS,
        "dense_activation": "relu",
        "dropout_rate": RiceConfig.DROPOUT_RATE,
        "output_units": 1,
        "output_activation": "linear (Continuous Weight in Grams)",
        "augmentation_techniques": [
            "RandomFlip(horizontal_and_vertical)",
            "RandomRotation(0.2)",
            "RandomZoom(0.2)",
            "RandomContrast(0.2)",
        ],
        "batch_size_per_replica": RiceConfig.BATCH_SIZE_PER_REPLICA,
        "mode": f"{RiceConfig.NUM_FOLDS}-Fold Cross-Validation" if RiceConfig.USE_KFOLD else "Single Split",
        "num_folds": RiceConfig.NUM_FOLDS if RiceConfig.USE_KFOLD else 1,
        "test_split_ratio": RiceConfig.TEST_SPLIT,
        "val_split_ratio": RiceConfig.VAL_SPLIT,
        "random_state": RiceConfig.RANDOM_STATE,
        "phase1_epochs": RiceConfig.PHASE1_EPOCHS,
        "phase1_optimizer": RiceConfig.PHASE1_OPTIMIZER,
        "phase1_loss": RiceConfig.PHASE1_LOSS,
        "phase1_backbone_trainable": False,
        "phase2_epochs": RiceConfig.PHASE2_EPOCHS,
        "phase2_optimizer": "adam",
        "phase2_lr": RiceConfig.PHASE2_LR,
        "phase2_batchnorm_trainable": False,
        "early_stopping_patience": RiceConfig.EARLY_STOPPING_PATIENCE,
        "early_stopping_monitor": "val_loss",
        "reduce_lr_patience": RiceConfig.REDUCE_LR_PATIENCE,
        "reduce_lr_factor": RiceConfig.REDUCE_LR_FACTOR,
        "reduce_lr_min": RiceConfig.MIN_LR,
        "reduce_lr_monitor": "val_loss",
        "wandb_project": RiceConfig.WANDB_PROJECT,
        "wandb_task": RiceConfig.WANDB_TASK,
    }

    if extra_info and isinstance(extra_info, dict):
        info.update(extra_info)

    # 1. เขียนไฟล์ข้อความ Text อ่านง่ายและละเอียด (training_config_summary.txt)
    lines = [
        "=" * 75,
        "HOSPITAL ML - TRAINING CONFIGURATION & EXPERIMENT SUMMARY",
        "=" * 75,
        f"Timestamp               : {info['timestamp']}",
        f"Platform / OS           : {info['platform']}",
        f"Python / TensorFlow     : Python {info['python_version']} | TensorFlow {info['tensorflow_version']}",
        f"Task Name               : {info['task_name']}",
        f"Backbone Architecture   : {info['backbone_model']}",
        f"Pretrained Weights      : {info['pretrained_weights']}",
        f"Input Image Resolution  : {info['image_size'][0]} x {info['image_size'][1]} (Channels: {info['image_channels']})",
        f"Image Resizing Method   : {info['image_preprocessing']}",
        "",
        "-" * 75,
        "REGRESSION HEAD ARCHITECTURE",
        "-" * 75,
        f"Pooling Layer           : {info['pooling_layer']}",
        f"Dense Hidden Layer      : {info['dense_units']} Units",
        f"Dense Activation        : {info['dense_activation']}",
        f"Dropout Regularization  : Rate = {info['dropout_rate']}",
        f"Output Layer            : {info['output_units']} Unit (Continuous Value in Grams)",
        f"Output Activation       : {info['output_activation']}",
        "",
        "-" * 75,
        "DATA AUGMENTATION PIPELINE",
        "-" * 75,
        f"Augmentation Layers     : 1. RandomFlip(horizontal_and_vertical)",
        f"                        : 2. RandomRotation(factor=0.2)",
        f"                        : 3. RandomZoom(height_factor=0.2, width_factor=0.2)",
        f"                        : 4. RandomContrast(factor=0.2)",
        "",
        "-" * 75,
        "DATASET & CROSS-VALIDATION STRATEGY",
        "-" * 75,
        f"Validation Strategy     : {info['mode']}",
        f"Number of Folds         : {info['num_folds']}",
        f"Hold-out Test Set Split : {int(info['test_split_ratio'] * 100)}% ({info.get('test_sample_count', 'N/A')} samples)",
        f"Train / Val Split Ratio : {int((1 - info['test_split_ratio']) * 100)}% ({info.get('train_val_sample_count', 'N/A')} samples across folds)",
        f"Random State / Seed     : {info['random_state']}",
        "",
        "-" * 75,
        "HARDWARE & DISTRIBUTED TRAINING",
        "-" * 75,
        f"Hardware Devices        : {info.get('hardware_devices', 'N/A')}",
        f"Batch Size Per Replica  : {info['batch_size_per_replica']}",
        f"Global Batch Size       : {info.get('global_batch_size', 'N/A')}",
        "",
        "-" * 75,
        "OPTIMIZATION & TWO-PHASE TRAINING",
        "-" * 75,
        f"Phase 1 (Top Layers)    : {info['phase1_epochs']} Epochs | Optimizer: {info['phase1_optimizer']} | Loss: {info['phase1_loss']}",
        f"  - Backbone Trainable  : {info['phase1_backbone_trainable']} (Frozen to preserve ImageNet features)",
        f"Phase 2 (Fine-Tuning)   : {info['phase2_epochs']} Epochs | Optimizer: {info['phase2_optimizer']} | LR: {info['phase2_lr']}",
        f"  - BatchNorm Trainable : {info['phase2_batchnorm_trainable']} (Frozen to stabilize Mean/Variance stats)",
        f"Early Stopping          : monitor={info['early_stopping_monitor']}, patience={info['early_stopping_patience']}, restore_best=True",
        f"Reduce LR on Plateau    : monitor={info['reduce_lr_monitor']}, factor={info['reduce_lr_factor']}, patience={info['reduce_lr_patience']}, min_lr={info['reduce_lr_min']}",
        "",
        "-" * 75,
        "LOGGING & EXPERIMENT TRACKING",
        "-" * 75,
        f"WandB Project           : {info['wandb_project']}",
        f"WandB Task / Run Name   : {info['wandb_task']}",
        f"WandB Run URL           : {info.get('wandb_url', 'N/A')}",
        "",
        "-" * 75,
        "FINAL EVALUATION METRICS (TEST SET ENSEMBLE)",
        "-" * 75,
        f"Ensemble MAE            : {info.get('ensemble_mae', 'N/A')} g",
        f"Ensemble RMSE           : {info.get('ensemble_rmse', 'N/A')}",
        f"Ensemble R² Score       : {info.get('ensemble_r2', 'N/A')}",
        f"Accuracy within ±5g     : {info.get('ensemble_acc_5g', 'N/A')} %",
        f"Accuracy within ±10g    : {info.get('ensemble_acc_10g', 'N/A')} %",
        "=" * 75,
    ]

    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[Evaluate] บันทึกไฟล์สรุปรายละเอียดการเทรนไปที่: {txt_path}")
    except Exception as e:
        print(f"[Evaluate] คำเตือน: ไม่สามารถบันทึก {txt_path}: {e}")

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=4, ensure_ascii=False)
        print(f"[Evaluate] บันทึก Manifest JSON ไปที่: {json_path}")
    except Exception as e:
        print(f"[Evaluate] คำเตือน: ไม่สามารถบันทึก {json_path}: {e}")

    return info


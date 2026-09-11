import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(actual_labels, predictions):
    """คำนวณสถิติความแม่นยำสำหรับการประมาณการน้ำหนัก:

    - MAE: ค่าคลาดเคลื่อนเฉลี่ย (กรัม)
    - RMSE: รากที่สองของค่าคลาดเคลื่อนกำลังสองเฉลี่ย
    - R²: สัมประสิทธิ์การตัดสินใจ (ความสัมพันธ์ระหว่างค่าจริงกับค่าทาย)
    - MAPE: เปอร์เซ็นต์ความคลาดเคลื่อนเฉลี่ย
    - Acc ±5g / ±10g: อัตราส่วนภาพที่ทายได้แม่นยำในกรอบ ±5g และ ±10g
    """
    actual = np.array(actual_labels).flatten()
    pred = np.array(predictions).flatten()

    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    r2 = r2_score(actual, pred)

    # คำนวณเปอร์เซ็นต์ความคลาดเคลื่อน (ตัดกรณีที่ actual == 0 ออกเพื่อกันหารด้วย 0)
    non_zero_mask = actual > 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((actual[non_zero_mask] - pred[non_zero_mask]) / actual[non_zero_mask])) * 100
    else:
        mape = 0.0

    errors = np.abs(actual - pred)
    acc_within_5g = np.mean(errors <= 5.0) * 100
    acc_within_10g = np.mean(errors <= 10.0) * 100

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape),
        "acc_within_5g": float(acc_within_5g),
        "acc_within_10g": float(acc_within_10g),
    }

    return metrics

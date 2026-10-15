
from pathlib import Path
import cv2
import numpy as np

LAB_DIR = Path(__file__).resolve().parents[1]

def calculate_iou(predicted_path, ground_truth_path):
    predicted = cv2.imread(
        str(predicted_path), cv2.IMREAD_GRAYSCALE
    )
    ground_truth = cv2.imread(
        str(ground_truth_path), cv2.IMREAD_GRAYSCALE
    )

    if predicted is None or ground_truth is None:
        raise FileNotFoundError("Не найдена одна из масок!")

    if predicted.shape != ground_truth.shape:
        raise ValueError("Размеры масок не совпадают!")

    predicted = predicted > 127
    ground_truth = ground_truth > 127

    intersection = np.logical_and(
        predicted, ground_truth
    ).sum()

    union = np.logical_or(
        predicted, ground_truth
    ).sum()

    iou = intersection / union if union > 0 else 1.0

    return iou

# IoU для дневной фотографии
iou_day = calculate_iou(
    LAB_DIR / "results" / "rgb_mask_day.png",
    LAB_DIR / "data" / "mask_day.png"
)

# IoU для вечерней фотографии
iou_evening = calculate_iou(
    LAB_DIR / "results" / "rgb_mask_evening.png",
    LAB_DIR / "data" / "mask_evening.png"
)

print("RGB segmentation results")
print("-" * 35)
print(f"Day IoU: {iou_day:.4f}")
print(f"Evening IoU: {iou_evening:.4f}")

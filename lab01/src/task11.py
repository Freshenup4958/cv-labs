
from pathlib import Path
import cv2
import numpy as np
import csv

LAB_DIR = Path(__file__).resolve().parents[1]

def calculate_iou(predicted_path, reference_path):
    predicted = cv2.imread(
        str(predicted_path), cv2.IMREAD_GRAYSCALE
    )
    reference = cv2.imread(
        str(reference_path), cv2.IMREAD_GRAYSCALE
    )

    if predicted is None or reference is None:
        raise FileNotFoundError("Одна из масок не найдена!")

    if predicted.shape != reference.shape:
        raise ValueError("Размеры масок не совпадают!")

    predicted = predicted > 127
    reference = reference > 127

    intersection = np.logical_and(predicted, reference).sum()
    union = np.logical_or(predicted, reference).sum()

    return intersection / union if union > 0 else 1.0

results = []

for color_space in ["rgb", "hsv"]:
    for frame in ["day", "evening"]:
        predicted_path = (
            LAB_DIR / "results" /
            f"{color_space}_mask_{frame}.png"
        )

        reference_path = (
            LAB_DIR / "data" /
            f"mask_{frame}.png"
        )

        iou = calculate_iou(
            predicted_path, reference_path
        )

        results.append([color_space.upper(), frame, iou])

# Вывод таблицы
print("Color Space | Frame   | IoU")
print("-" * 32)

for space, frame, iou in results:
    print(f"{space:<11} | {frame:<7} | {iou:.4f}")

# Сохранение CSV
output_path = LAB_DIR / "results" / "iou_comparison.csv"

with open(
    output_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:
    writer = csv.writer(file)
    writer.writerow(["Color Space", "Frame", "IoU"])
    writer.writerows(results)

print("\nТаблица сохранена:", output_path)

# Анализ устойчивости
rgb_day = results[0][2]
rgb_evening = results[1][2]
hsv_day = results[2][2]
hsv_evening = results[3][2]

print("\nИзменение IoU между кадрами:")
print(f"RGB: {abs(rgb_day - rgb_evening):.4f}")
print(f"HSV: {abs(hsv_day - hsv_evening):.4f}")

if abs(rgb_day - rgb_evening) < abs(hsv_day - hsv_evening):
    print("RGB показал меньшую разницу IoU между кадрами.")
elif abs(rgb_day - rgb_evening) > abs(hsv_day - hsv_evening):
    print("HSV показал меньшую разницу IoU между кадрами.")
else:
    print("Разница IoU между кадрами одинакова.")

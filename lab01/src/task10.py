
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

LAB_DIR = Path(__file__).resolve().parents[1]

# Загружаем фотографии
day = cv2.imread(str(LAB_DIR / "data" / "object_day.jpg"))
evening = cv2.imread(str(LAB_DIR / "data" / "object_evening.jpg"))

if day is None or evening is None:
    raise FileNotFoundError("Фотографии не найдены!")

# Преобразуем BGR в HSV
day_hsv = cv2.cvtColor(day, cv2.COLOR_BGR2HSV)
evening_hsv = cv2.cvtColor(evening, cv2.COLOR_BGR2HSV)

# Пороги жёлтого цвета в HSV
# OpenCV: H от 0 до 179
lower_hsv = np.array([15, 80, 70], dtype=np.uint8)
upper_hsv = np.array([40, 255, 255], dtype=np.uint8)

# Создаём маски
mask_day_hsv = cv2.inRange(
    day_hsv, lower_hsv, upper_hsv
)

mask_evening_hsv = cv2.inRange(
    evening_hsv, lower_hsv, upper_hsv
)

# Сохраняем результаты
cv2.imwrite(
    str(LAB_DIR / "results" / "hsv_mask_day.png"),
    mask_day_hsv
)

cv2.imwrite(
    str(LAB_DIR / "results" / "hsv_mask_evening.png"),
    mask_evening_hsv
)

# Отображаем фотографии и маски
fig, axes = plt.subplots(2, 2, figsize=(10, 9))

axes[0, 0].imshow(cv2.cvtColor(day, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Day - Original")

axes[0, 1].imshow(mask_day_hsv, cmap="gray", vmin=0, vmax=255)
axes[0, 1].set_title("Day - HSV Mask")

axes[1, 0].imshow(cv2.cvtColor(evening, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Evening - Original")

axes[1, 1].imshow(mask_evening_hsv, cmap="gray", vmin=0, vmax=255)
axes[1, 1].set_title("Evening - HSV Mask")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    LAB_DIR / "figures" / "hsv_comparison.png",
    dpi=300
)

plt.show()

print("HSV-пороги:")
print("Нижняя граница:", lower_hsv)
print("Верхняя граница:", upper_hsv)
print("Маски HSV сохранены!")

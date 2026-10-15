
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

LAB_DIR = Path(__file__).resolve().parents[1]

# Загружаем две фотографии
day = cv2.imread(str(LAB_DIR / "data" / "object_day.jpg"))
evening = cv2.imread(str(LAB_DIR / "data" / "object_evening.jpg"))

if day is None or evening is None:
    raise FileNotFoundError("Не найдены фотографии!")

# Переводим BGR в RGB
day_rgb = cv2.cvtColor(day, cv2.COLOR_BGR2RGB)
evening_rgb = cv2.cvtColor(evening, cv2.COLOR_BGR2RGB)

# Пороги для жёлтого цвета в RGB
lower_rgb = np.array([120, 80, 0], dtype=np.uint8)
upper_rgb = np.array([255, 255, 170], dtype=np.uint8)

# Автоматическое выделение цвета
mask_day_rgb = cv2.inRange(
    day_rgb, lower_rgb, upper_rgb
)

mask_evening_rgb = cv2.inRange(
    evening_rgb, lower_rgb, upper_rgb
)

# Сохраняем маски
cv2.imwrite(
    str(LAB_DIR / "results" / "rgb_mask_day.png"),
    mask_day_rgb
)

cv2.imwrite(
    str(LAB_DIR / "results" / "rgb_mask_evening.png"),
    mask_evening_rgb
)

# Отображение результатов
fig, axes = plt.subplots(2, 2, figsize=(10, 9))

axes[0, 0].imshow(day_rgb)
axes[0, 0].set_title("Day - Original")

axes[0, 1].imshow(mask_day_rgb, cmap="gray", vmin=0, vmax=255)
axes[0, 1].set_title("Day - RGB Mask")

axes[1, 0].imshow(evening_rgb)
axes[1, 0].set_title("Evening - Original")

axes[1, 1].imshow(mask_evening_rgb, cmap="gray", vmin=0, vmax=255)
axes[1, 1].set_title("Evening - RGB Mask")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    LAB_DIR / "figures" / "rgb_comparison.png",
    dpi=300
)

plt.show()

print("RGB-пороги:")
print("Нижняя граница:", lower_rgb)
print("Верхняя граница:", upper_rgb)
print("Маски сохранены!")

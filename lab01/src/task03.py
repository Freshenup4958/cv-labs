
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

LAB_DIR = Path(__file__).resolve().parents[1]

# Загружаем изображение
img = cv2.imread(str(LAB_DIR / "data" / "img.png"))

if img is None:
    raise FileNotFoundError("Изображение не найдено!")

# Неправильное увеличение яркости
# uint8 переполняется при значениях больше 255
incorrect = img + 50

# Правильное увеличение яркости
# Значения ограничиваются диапазоном 0–255
correct = cv2.add(img, 50)

# Преобразуем BGR в RGB для отображения
original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
incorrect_rgb = cv2.cvtColor(incorrect, cv2.COLOR_BGR2RGB)
correct_rgb = cv2.cvtColor(correct, cv2.COLOR_BGR2RGB)

# Строим сравнение
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(original_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(incorrect_rgb)
plt.title("Incorrect: uint8 + 50")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(correct_rgb)
plt.title("Correct: cv2.add")
plt.axis("off")

plt.tight_layout()

# Сохраняем рисунок для отчёта
plt.savefig(
    LAB_DIR / "figures" / "brightness_comparison.png",
    dpi=300
)

plt.show()

# Выводим числовой пример переполнения
pixel = np.array([250], dtype=np.uint8)
print("Исходное значение:", pixel[0])
print("Неправильное сложение:", (pixel + 50)[0])
print("Правильное сложение:", np.clip(
    pixel.astype(np.int16) + 50, 0, 255
)[0])


import matplotlib.pyplot as plt
from pathlib import Path
import cv2

# Путь к папке lab01
LAB_DIR = Path(__file__).resolve().parents[1]

# Чтение изображения
img = cv2.imread(str(LAB_DIR / "data" / "img.png"))

if img is None:
    raise FileNotFoundError("Изображение не найдено!")

# Характеристики изображения
print("Shape:", img.shape)
print("Dtype:", img.dtype)
print("Min:", img.min())
print("Max:", img.max())

# Перевод BGR в RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Создание двух изображений для сравнения
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Incorrect: BGR")
plt.axis("off")


# Второе изображение — правильные цвета
plt.subplot(1, 2, 2)
plt.imshow(img_rgb)
plt.title("Correct: RGB")
plt.axis("off")

plt.tight_layout()

# Сохранение результата
plt.savefig(
    LAB_DIR / "figures" / "bgr_rgb_comparison.png",
    dpi=300
)

# Показ изображений
plt.show()

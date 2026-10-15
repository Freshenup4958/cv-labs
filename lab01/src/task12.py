
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

LAB_DIR = Path(__file__).resolve().parents[1]

img = cv2.imread(str(LAB_DIR / "data" / "texture.jpg"))

if img is None:
    raise FileNotFoundError("Файл texture.jpg не найден!")

height, width = img.shape[:2]
new_size = (width // 4, height // 4)

# Способ 1: уменьшение без предварительного сглаживания
nearest = cv2.resize(
    img, new_size,
    interpolation=cv2.INTER_NEAREST
)

# Способ 2: сглаживание перед уменьшением
blurred = cv2.GaussianBlur(img, (5, 5), 1.5)

gaussian_nearest = cv2.resize(
    blurred, new_size,
    interpolation=cv2.INTER_NEAREST
)

# Переводим изображения в RGB
original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
nearest_rgb = cv2.cvtColor(nearest, cv2.COLOR_BGR2RGB)
gaussian_rgb = cv2.cvtColor(
    gaussian_nearest, cv2.COLOR_BGR2RGB
)

# Сравнение

# Сравнение увеличенных фрагментов

# Берём центральный участок каждого изображения
def crop_center(image, size=100):
    h, w = image.shape[:2]
    size = min(size, h, w)
    y = (h - size) // 2
    x = (w - size) // 2
    return image[y:y + size, x:x + size]

# Увеличиваем фрагменты без сглаживания
original_crop = crop_center(original_rgb, 400)
nearest_crop = crop_center(nearest_rgb, 100)
gaussian_crop = crop_center(gaussian_rgb, 100)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

images = [original_crop, nearest_crop, gaussian_crop]
titles = [
    "Original - crop",
    "INTER_NEAREST - 4x",
    "GaussianBlur + resize - 4x"
]

for ax, image, title in zip(axes, images, titles):
    ax.imshow(image, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    LAB_DIR / "figures" / "aliasing_comparison.png",
    dpi=300
)

plt.show()

# Сохраняем уменьшенные изображения
cv2.imwrite(
    str(LAB_DIR / "results" / "nearest.png"),
    nearest
)

cv2.imwrite(
    str(LAB_DIR / "results" / "gaussian_resize.png"),
    gaussian_nearest
)

print("Исходный размер:", width, "x", height)
print("Уменьшенный размер:", new_size)
print("Изображения сохранены!")

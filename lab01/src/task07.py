
from pathlib import Path
import cv2
import numpy as np

LAB_DIR = Path(__file__).resolve().parents[1]

img = cv2.imread(str(LAB_DIR / "data" / "object_evening.jpg"))

if img is None:
    raise FileNotFoundError("Фотография не найдена!")

# Уменьшаем изображение для удобного отображения
height, width = img.shape[:2]
scale = min(800 / width, 600 / height, 1.0)

new_width = int(width * scale)
new_height = int(height * scale)

small_img = cv2.resize(
    img,
    (new_width, new_height)
)

# Маска исходного размера
mask = np.zeros((height, width), dtype=np.uint8)

points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Переводим координаты в исходный размер
        original_x = int(x / scale)
        original_y = int(y / scale)

        points.append((original_x, original_y))
        print("Точка:", original_x, original_y)

cv2.namedWindow("Create Mask", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Create Mask", mouse_callback)

while True:
    display = small_img.copy()

    # Показываем точки на уменьшенном изображении
    small_points = [
        (int(x * scale), int(y * scale))
        for x, y in points
    ]

    for point in small_points:
        cv2.circle(display, point, 3, (0, 0, 255), -1)

    if len(small_points) > 1:
        cv2.polylines(
            display,
            [np.array(small_points, dtype=np.int32)],
            False,
            (0, 255, 0),
            2
        )

    cv2.imshow("Create Mask", display)

    key = cv2.waitKey(20) & 0xFF

    if key == ord("s") and len(points) >= 3:
        cv2.fillPoly(
            mask,
            [np.array(points, dtype=np.int32)],
            255
        )

        output = LAB_DIR / "data" / "mask_evening.png"
        cv2.imwrite(str(output), mask)

        print("Маска сохранена:", output)
        break

    elif key == ord("r"):
        points.clear()
        mask[:] = 0

    elif key == 27:
        break

cv2.destroyAllWindows()

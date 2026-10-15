
from pathlib import Path
import cv2

LAB_DIR = Path(__file__).resolve().parents[1]

# Фотография и маска, которую исправляем
photo_path = LAB_DIR / "data" / "object_evening.jpg"
mask_path = LAB_DIR / "data" / "mask_evening.png"

img = cv2.imread(str(photo_path))
mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

if img is None or mask is None:
    raise FileNotFoundError("Фото или маска не найдены!")

if img.shape[:2] != mask.shape:
    raise ValueError("Размеры фотографии и маски не совпадают!")

# Уменьшаем изображение для экрана
height, width = img.shape[:2]
scale = min(800 / width, 600 / height, 1.0)

size = (int(width * scale), int(height * scale))

# Кисть в пикселях исходной фотографии
brush_size = 12
drawing = False
brush_color = 255

def mouse_callback(event, x, y, flags, param):
    global drawing, brush_color

    # Левая кнопка — белая кисть
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        brush_color = 255

    # Правая кнопка — чёрная кисть
    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing = True
        brush_color = 0

    elif event in (
        cv2.EVENT_LBUTTONUP,
        cv2.EVENT_RBUTTONUP
    ):
        drawing = False
        return

    if drawing:
        original_x = min(int(x / scale), width - 1)
        original_y = min(int(y / scale), height - 1)

        cv2.circle(
            mask,
            (original_x, original_y),
            brush_size,
            brush_color,
            -1
        )

cv2.namedWindow("Mask Editor")
cv2.setMouseCallback("Mask Editor", mouse_callback)

print("Левая кнопка — белый цвет")
print("Правая кнопка — чёрный цвет")
print("+ / - — размер кисти")
print("S — сохранить")
print("ESC — выход")

while True:
    # Накладываем маску на фотографию
    overlay = img.copy()
    overlay[mask == 255] = (0, 255, 0)

    preview = cv2.addWeighted(
        img, 0.5, overlay, 0.5, 0
    )

    small_preview = cv2.resize(
        preview,
        size,
        interpolation=cv2.INTER_AREA
    )

    cv2.imshow("Mask Editor", small_preview)

    key = cv2.waitKey(20) & 0xFF

    if key == ord("s"):
        cv2.imwrite(str(mask_path), mask)
        print("Маска сохранена!")
        break

    elif key in (ord("+"), ord("=")):
        brush_size += 3
        print("Размер кисти:", brush_size)

    elif key == ord("-"):
        brush_size = max(1, brush_size - 3)
        print("Размер кисти:", brush_size)

    elif key == 27:
        print("Выход без сохранения")
        break

cv2.destroyAllWindows()

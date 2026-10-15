
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

LAB_DIR = Path(__file__).resolve().parents[1]

img = cv2.imread(str(LAB_DIR / "data" / "texture.jpg"))

if img is None:
    raise FileNotFoundError("Файл texture.jpg не найден!")

# Выбираем центральный участок 320x320
h, w = img.shape[:2]
size = min(320, h, w)
x = (w - size) // 2
y = (h - size) // 2

original = img[y:y + size, x:x + size]

# Уменьшаем участок в 8 раз
small = cv2.resize(
    original,
    (size // 8, size // 8),
    interpolation=cv2.INTER_AREA
)

methods = {
    "NEAREST": cv2.INTER_NEAREST,
    "LINEAR": cv2.INTER_LINEAR,
    "CUBIC": cv2.INTER_CUBIC,
    "LANCZOS4": cv2.INTER_LANCZOS4
}

results = []

fig, axes = plt.subplots(1, 5, figsize=(18, 5))

axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original")
axes[0].axis("off")

for i, (name, method) in enumerate(methods.items(), start=1):
    enlarged = cv2.resize(
        small,
        (size, size),
        interpolation=method
    )

    # PSNR относительно исходного изображения
    psnr = cv2.PSNR(original, enlarged)

    results.append([name, psnr])

    axes[i].imshow(
        cv2.cvtColor(enlarged, cv2.COLOR_BGR2RGB)
    )
    axes[i].set_title(f"{name}\nPSNR = {psnr:.2f} dB")
    axes[i].axis("off")

    cv2.imwrite(
        str(LAB_DIR / "results" / f"interpolation_{name.lower()}.png"),
        enlarged
    )

plt.tight_layout()

plt.savefig(
    LAB_DIR / "figures" / "interpolation_comparison.png",
    dpi=300
)

plt.show()

with open(
    LAB_DIR / "results" / "interpolation_psnr.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:
    writer = csv.writer(file)
    writer.writerow(["Method", "PSNR_dB"])
    writer.writerows(results)

print("Interpolation PSNR results:")
print("-" * 30)

for name, psnr in results:
    print(f"{name:<10}: {psnr:.2f} dB")

print("Результаты сохранены!")

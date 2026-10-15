
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

LAB_DIR = Path(__file__).resolve().parents[1]
RESULTS = LAB_DIR / "results"
FIGURES = LAB_DIR / "figures"

RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

# Исходное изображение
img = cv2.imread(str(LAB_DIR / "data" / "img.png"))

if img is None:
    raise FileNotFoundError("Изображение не найдено!")

# Значения качества JPEG
qualities = [95, 75, 50, 25, 10]
data = []

# Сохраняем PNG
png_path = RESULTS / "original.png"
cv2.imwrite(str(png_path), img)

png_size = png_path.stat().st_size / 1024
png_decoded = cv2.imread(str(png_path))
png_psnr = cv2.PSNR(img, png_decoded)

data.append(["PNG", png_size, png_psnr])

# Сохраняем JPEG с разным качеством
for quality in qualities:
    path = RESULTS / f"jpeg_{quality}.jpg"

    cv2.imwrite(
        str(path),
        img,
        [cv2.IMWRITE_JPEG_QUALITY, quality]
    )

    compressed = cv2.imread(str(path))
    size_kb = path.stat().st_size / 1024

    # Расчёт PSNR
    psnr = cv2.PSNR(img, compressed)

    data.append([f"JPEG {quality}", size_kb, psnr])

# Выводим результаты
print("Format       Size (KB)       PSNR (dB)")
print("-" * 45)

for row in data:
    print(f"{row[0]:<12} {row[1]:<15.2f} {row[2]:.2f}")

# Сохраняем таблицу в CSV
with open(
    RESULTS / "compression_results.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:
    writer = csv.writer(file)
    writer.writerow(["Format", "Size_KB", "PSNR_dB"])
    writer.writerows(data)

# График зависимости PSNR от размера
jpeg_data = data[1:]
sizes = [row[1] for row in jpeg_data]
psnr_values = [row[2] for row in jpeg_data]

plt.figure(figsize=(9, 6))
plt.plot(sizes, psnr_values, marker="o")

for i, quality in enumerate(qualities):
    plt.annotate(
        f"Q={quality}",
        (sizes[i], psnr_values[i]),
        xytext=(5, 5),
        textcoords="offset points"
    )

plt.xlabel("File size (KB)")
plt.ylabel("PSNR (dB)")
plt.title("JPEG compression: PSNR vs file size")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    FIGURES / "psnr_graph.png",
    dpi=300
)

plt.show()

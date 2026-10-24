
from pathlib import Path
from datetime import date
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"

calibration = np.load(
    RESULTS_DIR / "calibration_data.npz"
)

camera_matrix = calibration["camera_matrix"]
dist_coeffs = calibration["dist_coeffs"]
rms = float(calibration["rms"])
image_size = calibration["image_size"]

used_images = 22

output_path = BASE_DIR / "camera_params.yml"

fs = cv2.FileStorage(
    str(output_path),
    cv2.FILE_STORAGE_WRITE
)

if not fs.isOpened():
    raise RuntimeError("Не вдалося створити YAML-файл")

fs.write("camera_matrix", camera_matrix)
fs.write("dist_coeffs", dist_coeffs)
fs.write("image_width", int(image_size[0]))
fs.write("image_height", int(image_size[1]))
fs.write("rms", rms)
fs.write("used_images", used_images)
fs.write("date", date.today().isoformat())

fs.release()

print("Параметри камери успішно збережено.")
print(f"Файл: {output_path}")
print(f"Кількість кадрів: {used_images}")
print(f"Похибка RMS: {rms:.6f}")

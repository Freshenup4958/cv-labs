
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

calibration = np.load(
    RESULTS_DIR / "calibration_data.npz"
)

camera_matrix = calibration["camera_matrix"]
dist_coeffs = calibration["dist_coeffs"]

image = cv2.imread(
    str(DATA_DIR / "straight_line.jpg")
)

if image is None:
    raise FileNotFoundError(
        "Фотографію straight_line.jpg не знайдено"
    )

height, width = image.shape[:2]

calibration_size = tuple(
    calibration["image_size"]
)

if (width, height) != calibration_size:
    scale_x = width / calibration_size[0]
    scale_y = height / calibration_size[1]

    camera_matrix = camera_matrix.copy()
    camera_matrix[0, 0] *= scale_x
    camera_matrix[0, 2] *= scale_x
    camera_matrix[1, 1] *= scale_y
    camera_matrix[1, 2] *= scale_y

new_camera_matrix, roi = (
    cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        (width, height),
        1,
        (width, height)
    )
)

corrected = cv2.undistort(
    image,
    camera_matrix,
    dist_coeffs,
    None,
    new_camera_matrix
)

original_marked = image.copy()
corrected_marked = corrected.copy()

y = int(height * 0.41)

cv2.line(
    original_marked,
    (0, y),
    (width - 1, y),
    (0, 0, 255),
    4
)

cv2.line(
    corrected_marked,
    (0, y),
    (width - 1, y),
    (0, 0, 255),
    4
)

original_rgb = cv2.cvtColor(
    original_marked,
    cv2.COLOR_BGR2RGB
)

corrected_rgb = cv2.cvtColor(
    corrected_marked,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(original_rgb)
plt.title("До корекції дисторсії")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(corrected_rgb)
plt.title("Після корекції дисторсії")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "visual_check.png",
    dpi=200
)

cv2.imwrite(
    str(RESULTS_DIR / "straight_line_corrected.jpg"),
    corrected
)

plt.show()

print("Візуальну перевірку виконано.")
print("Результати збережено.")

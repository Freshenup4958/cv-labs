
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Шляхи до папок
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

calibration = np.load(
    RESULTS_DIR / "calibration_data.npz"
)

camera_matrix = calibration["camera_matrix"]
dist_coeffs = calibration["dist_coeffs"]

image_path = DATA_DIR / "calib_13.jpg"
image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(
        "Не вдалося завантажити фотографію"
    )

height, width = image.shape[:2]

corrected_images = {}

for alpha in [0, 1]:
    new_camera_matrix, roi = (
        cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            dist_coeffs,
            (width, height),
            alpha,
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

    corrected_images[alpha] = corrected

    output_path = (
        RESULTS_DIR /
        f"calib_13_alpha_{alpha}.jpg"
    )

    cv2.imwrite(str(output_path), corrected)

    print(
        f"alpha = {alpha}: "
        f"зображення збережено, ROI = {roi}"
    )

images = [
    image,
    corrected_images[0],
    corrected_images[1]
]

titles = [
    "Початкове зображення",
    "Корекція: alpha = 0",
    "Корекція: alpha = 1"
]

plt.figure(figsize=(15, 6))

for i, (img, title) in enumerate(
    zip(images, titles), start=1
):
    rgb = cv2.cvtColor(
        img, cv2.COLOR_BGR2RGB
    )

    plt.subplot(1, 3, i)
    plt.imshow(rgb)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "alpha_comparison.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("\nПорівняння завершено.")
print("Результати збережено.")

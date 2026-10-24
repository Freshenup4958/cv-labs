
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

PATTERN_SIZE = (9, 6)

object_points = np.zeros(
    (PATTERN_SIZE[0] * PATTERN_SIZE[1], 3),
    np.float32
)

object_points[:, :2] = np.mgrid[
    0:PATTERN_SIZE[0],
    0:PATTERN_SIZE[1]
].T.reshape(-1, 2)

criteria = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

errors = []
names = []

# Обробка фотографій
for image_path in sorted(DATA_DIR.glob("calib_*.jpg")):
    image = cv2.imread(str(image_path))

    if image is None:
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCornersSB(
        gray, PATTERN_SIZE
    )

    if not found:
        print(f"{image_path.name}: кути не знайдено")
        continue

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        corners,
        camera_matrix,
        dist_coeffs
    )

    if not success:
        continue

    projected_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs
    )

    difference = (
        corners.reshape(-1, 2) -
        projected_points.reshape(-1, 2)
    )

    error = np.sqrt(
        np.mean(np.sum(difference ** 2, axis=1))
    )

    errors.append(error)
    names.append(image_path.stem)

    print(f"{image_path.name}: похибка {error:.4f} пікс.")

if not errors:
    raise RuntimeError("Немає даних для аналізу")

plt.figure(figsize=(14, 6))
plt.bar(names, errors)

plt.axhline(
    np.mean(errors),
    linestyle="--",
    label=f"Середня похибка: {np.mean(errors):.3f}"
)

plt.xlabel("Фотографії")
plt.ylabel("Похибка, пікселі")
plt.title("Похибки перепроєктування")
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "reprojection_errors.png",
    dpi=200
)

plt.show()

print("\nРЕЗУЛЬТАТИ АНАЛІЗУ")
print(f"Кількість фотографій: {len(errors)}")
print(f"Середня похибка: {np.mean(errors):.4f}")
print(f"Максимальна похибка: {np.max(errors):.4f}")
print(f"Мінімальна похибка: {np.min(errors):.4f}")

print("\nНайбільша похибка:")
index = np.argmax(errors)
print(f"{names[index]}: {errors[index]:.4f} пікс.")

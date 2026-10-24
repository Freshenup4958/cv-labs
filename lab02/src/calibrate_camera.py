
from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKERBOARD = (9, 6)
SQUARE_SIZE = 15.5

EXCLUDED_IMAGES = {
    "calib_08.jpg",
    "calib_13.jpg"
}

object_points_template = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    np.float32
)

object_points_template[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

object_points_template *= SQUARE_SIZE

object_points = []
image_points = []
image_names = []

images = sorted(DATA_DIR.glob("calib_*.jpg"))
image_size = None

for image_path in images:
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"{image_path.name}: не вдалося відкрити")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCornersSB(
        gray,
        CHECKERBOARD,
        flags=cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if not found:
        print(f"{image_path.name}: кути не знайдено")
        continue

    current_size = (gray.shape[1], gray.shape[0])

    if image_size is None:
        image_size = current_size
    elif current_size != image_size:
        print(
            f"{image_path.name}: інший розмір зображення"
        )
        continue

    object_points.append(object_points_template.copy())
    image_points.append(corners.astype(np.float32))
    image_names.append(image_path.name)

    print(f"{image_path.name}: кути знайдено")

if len(object_points) < 3:
    raise RuntimeError(
        "Недостатньо зображень для калібрування"
    )

def calibrate(obj_points, img_points):
    return cv2.calibrateCamera(
        obj_points,
        img_points,
        image_size,
        None,
        None
    )

rms_all, matrix_all, dist_all, _, _ = calibrate(
    object_points,
    image_points
)

filtered_object_points = []
filtered_image_points = []
filtered_names = []

for name, obj, img in zip(
    image_names, object_points, image_points
):
    if name not in EXCLUDED_IMAGES:
        filtered_object_points.append(obj)
        filtered_image_points.append(img)
        filtered_names.append(name)

if len(filtered_object_points) < 3:
    raise RuntimeError(
        "Недостатньо фотографій після виключення"
    )

rms_filtered, matrix_filtered, dist_filtered, _, _ = (
    calibrate(
        filtered_object_points,
        filtered_image_points
    )
)

print("\nПОРІВНЯННЯ КАЛІБРУВАННЯ")

print("\nУсі розпізнані фотографії:")
print(f"Кількість: {len(object_points)}")
print(f"RMS: {rms_all:.6f}")

print("\nБез calib_08.jpg та calib_13.jpg:")
print(f"Кількість: {len(filtered_object_points)}")
print(f"RMS: {rms_filtered:.6f}")

print("\nРізниця RMS:")
print(f"{rms_all - rms_filtered:.6f}")

print("\nМАТРИЦЯ КАМЕРИ ПІСЛЯ ВИКЛЮЧЕННЯ:")
print(matrix_filtered)

print("\nКОЕФІЦІЄНТИ ДИСТОРСІЇ:")
print(dist_filtered)

np.savez(
    RESULTS_DIR / "calibration_all.npz",
    camera_matrix=matrix_all,
    dist_coeffs=dist_all,
    rms=rms_all,
    image_size=image_size,
    image_names=np.array(image_names)
)

np.savez(
    RESULTS_DIR / "calibration_filtered.npz",
    camera_matrix=matrix_filtered,
    dist_coeffs=dist_filtered,
    rms=rms_filtered,
    image_size=image_size,
    image_names=np.array(filtered_names)
)

print("\nОбидва результати збережено.")
print("Початковий calibration_data.npz не змінено.")

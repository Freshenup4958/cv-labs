
from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

CHECKERBOARD = (9, 6)
SQUARE_SIZE = 15.5

objp = np.zeros((54, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
objp *= SQUARE_SIZE

points_before = []
points_after = []
object_points = []

image_size = None

criteria = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

for path in sorted(DATA_DIR.glob("calib_*.jpg")):
    image = cv2.imread(str(path))

    if image is None:
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCornersSB(
        gray,
        CHECKERBOARD,
        flags=cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if not found:
        print(f"{path.name}: кути не знайдено")
        continue

    size = (gray.shape[1], gray.shape[0])

    if image_size is None:
        image_size = size

    if size != image_size:
        print(f"{path.name}: інший розмір зображення")
        continue

    corners_before = corners.astype(np.float32)

    corners_after = cv2.cornerSubPix(
        gray,
        corners_before.copy(),
        (5, 5),
        (-1, -1),
        criteria
    )

    object_points.append(objp.copy())
    points_before.append(corners_before)
    points_after.append(corners_after)

    print(f"{path.name}: кути оброблено")

if len(object_points) < 3:
    raise RuntimeError("Недостатньо фотографій")

rms_before, _, _, _, _ = cv2.calibrateCamera(
    object_points,
    points_before,
    image_size,
    None,
    None
)

rms_after, _, _, _, _ = cv2.calibrateCamera(
    object_points,
    points_after,
    image_size,
    None,
    None
)

print("\nПОРІВНЯННЯ ТОЧНОСТІ")

print(f"Кількість фотографій: {len(object_points)}")
print(f"RMS до уточнення: {rms_before:.6f}")
print(f"RMS після уточнення: {rms_after:.6f}")

if rms_after > 0:
    print(
        f"Відношення похибок: "
        f"{rms_before / rms_after:.3f}"
    )

print(
    f"Зміна RMS: "
    f"{rms_before - rms_after:.6f}"
)

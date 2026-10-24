
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKERBOARD = (9, 6)
SQUARE_SIZE = 15.5

FRAME_COUNTS = [5, 8, 10, 12, 15, 20]
REPEATS = 3

rng = np.random.default_rng(42)

objp = np.zeros((54, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
objp *= SQUARE_SIZE

object_points = []
image_points = []
valid_names = []
image_size = None

for path in sorted(DATA_DIR.glob("calib_*.jpg")):
    image = cv2.imread(str(path))

    if image is None:
        print(f"{path.name}: помилка зчитування")
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

    current_size = (gray.shape[1], gray.shape[0])

    if image_size is None:
        image_size = current_size

    if current_size != image_size:
        print(f"{path.name}: інший розмір зображення")
        continue

    object_points.append(objp.copy())
    image_points.append(corners.astype(np.float32))
    valid_names.append(path.name)

    print(f"{path.name}: додано")

print(f"\nУсього придатних кадрів: {len(valid_names)}")

if len(valid_names) < max(FRAME_COUNTS):
    raise RuntimeError(
        "Недостатньо фотографій для дослідження"
    )

results = []

for n in FRAME_COUNTS:
    rms_values = []
    fx_values = []

    print(f"\nКІЛЬКІСТЬ КАДРІВ: {n}")

    for repeat in range(REPEATS):
        indices = rng.choice(
            len(valid_names),
            size=n,
            replace=False
        )

        selected_obj = [
            object_points[i] for i in indices
        ]

        selected_img = [
            image_points[i] for i in indices
        ]

        rms, camera_matrix, dist_coeffs, _, _ = (
            cv2.calibrateCamera(
                selected_obj,
                selected_img,
                image_size,
                None,
                None
            )
        )

        fx = camera_matrix[0, 0]

        rms_values.append(rms)
        fx_values.append(fx)

        print(
            f"Підбірка {repeat + 1}: "
            f"RMS = {rms:.4f}, "
            f"fx = {fx:.2f}"
        )

    results.append([
        n,
        np.mean(rms_values),
        np.std(rms_values),
        np.mean(fx_values),
        np.std(fx_values)
    ])

results = np.array(results)

print("\nПІДСУМКОВІ РЕЗУЛЬТАТИ")
print(
    "N | Середня RMS | Розкид RMS | "
    "Середнє fx | Розкид fx"
)

for row in results:
    print(
        f"{int(row[0]):2d} | "
        f"{row[1]:11.4f} | "
        f"{row[2]:10.4f} | "
        f"{row[3]:10.2f} | "
        f"{row[4]:9.2f}"
    )

plt.figure(figsize=(9, 5))

plt.errorbar(
    results[:, 0],
    results[:, 1],
    yerr=results[:, 2],
    marker="o",
    capsize=5
)

plt.xlabel("Кількість кадрів")
plt.ylabel("Похибка RMS, пікселі")
plt.title("Залежність похибки від кількості кадрів")
plt.grid(True)

plt.savefig(
    FIGURES_DIR / "frame_count_rms.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.figure(figsize=(9, 5))

plt.errorbar(
    results[:, 0],
    results[:, 3],
    yerr=results[:, 4],
    marker="o",
    capsize=5
)

plt.xlabel("Кількість кадрів")
plt.ylabel("Фокусна відстань fx, пікселі")
plt.title("Залежність fx від кількості кадрів")
plt.grid(True)

plt.savefig(
    FIGURES_DIR / "frame_count_fx.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

np.savetxt(
    RESULTS_DIR / "frame_count_results.csv",
    results,
    delimiter=",",
    header="N,mean_rms,std_rms,mean_fx,std_fx",
    comments="",
    fmt="%.6f"
)

print("\nДослідження завершено.")
print("Графіки збережено у папці figures.")
print("Таблицю збережено у папці results.")

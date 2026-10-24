
from pathlib import Path
import cv2

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

CHECKERBOARD = (9, 6)

images = sorted(DATA_DIR.glob("*.jpg"))

print(f"Знайдено фотографій: {len(images)}")

successful = 0

for image_path in images:
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"{image_path.name}: помилка зчитування")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCornersSB(
        gray,
        CHECKERBOARD,
        flags=cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if found:
        successful += 1
        print(f"{image_path.name}: знайдено 54 кути")

        result_image = image.copy()

        cv2.drawChessboardCorners(
            result_image,
            CHECKERBOARD,
            corners,
            found
        )

        output_dir = DATA_DIR.parent / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"corners_{image_path.name}"
        cv2.imwrite(str(output_path), result_image)
    else:
        print(f"{image_path.name}: кути не знайдено")

print(f"\nУспішно оброблено: {successful} із {len(images)}")

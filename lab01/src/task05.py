
from pathlib import Path
from PIL import Image, ExifTags

LAB_DIR = Path(__file__).resolve().parents[1]
photo_path = LAB_DIR / "data" / "photo.jpg"

if not photo_path.exists():
    raise FileNotFoundError("Файл photo.jpg не найден!")

with Image.open(photo_path) as img:
    exif = img.getexif()

    # Получаем также данные из раздела Exif IFD
    exif_data = {}

    for tag_id, value in exif.items():
        tag_name = ExifTags.TAGS.get(tag_id, tag_id)
        exif_data[tag_name] = value

    if hasattr(exif, "get_ifd"):
        exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
        for tag_id, value in exif_ifd.items():
            tag_name = ExifTags.TAGS.get(tag_id, tag_id)
            exif_data[tag_name] = value

print("EXIF-данные фотографии:")
print("-" * 35)

focal = exif_data.get("FocalLength")
exposure = exif_data.get("ExposureTime")
iso = exif_data.get("ISOSpeedRatings")

print("Фокусное расстояние:", focal)
print("Выдержка:", exposure)
print("ISO:", iso)

if not exif_data:
    print("EXIF-данные отсутствуют.")

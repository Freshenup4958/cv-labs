
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

LAB_DIR = Path(__file__).resolve().parents[1]

img = cv2.imread(str(LAB_DIR / "data" / "texture.jpg"))

if img is None:
    raise FileNotFoundError("Изображение не найдено!")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_float = img_rgb.astype(np.float32) / 255.0

gammas = [0.5, 1.0, 2.2]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for i, gamma in enumerate(gammas):
    corrected = np.power(img_float, gamma)
    corrected_uint8 = np.clip(
        corrected * 255, 0, 255
    ).astype(np.uint8)

    axes[i].imshow(corrected_uint8)
    axes[i].set_title(f"Gamma = {gamma}")
    axes[i].axis("off")

    output = LAB_DIR / "results" / f"gamma_{gamma}.png"

    cv2.imwrite(
        str(output),
        cv2.cvtColor(corrected_uint8, cv2.COLOR_RGB2BGR)
    )

plt.tight_layout()

plt.savefig(
    LAB_DIR / "figures" / "gamma_comparison.png",
    dpi=300
)

plt.show()

x = np.linspace(0, 1, 256)

plt.figure(figsize=(8, 6))

for gamma in gammas:
    y = x ** gamma
    plt.plot(x, y, label=f"Gamma = {gamma}")

plt.xlabel("Input intensity")
plt.ylabel("Output intensity")
plt.title("Gamma correction curves")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    LAB_DIR / "figures" / "gamma_curves.png",
    dpi=300
)

plt.show()

print("Гамма-коррекция выполнена!")
print("Изображения и графики сохранены.")

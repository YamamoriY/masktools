from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
IMAGE_PATH = HERE / "image.jpg"
MASK_PATH = HERE / "mask.png"
OUTPUT_PATH = HERE / "masked.png"
THRESHOLD = 128


def main() -> None:
    image = Image.open(IMAGE_PATH).convert("RGB")
    mask = Image.open(MASK_PATH).convert("L").resize(image.size, Image.BILINEAR)

    mask_arr = np.array(mask) < THRESHOLD
    image_arr = np.array(image)

    out = np.zeros_like(image_arr)
    out[mask_arr] = image_arr[mask_arr]
    Image.fromarray(out, mode="RGB").save(OUTPUT_PATH)
    print(f"saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
IMAGE_PATH = HERE / "image.jpg"
MASKS = [
    ("mask_gemini.png", "masked_gemini.png"),
    ("mask_gpt.png", "masked_gpt.png"),
    ("mask_gpt5.png", "masked_gpt5.png"),
    ("mask_gpt5.4.png", "masked_gpt5.4.png"),
]
THRESHOLD = 128


def main() -> None:
    image = Image.open(IMAGE_PATH).convert("RGB")
    image_arr = np.array(image)

    for mask_name, output_name in MASKS:
        mask_path = HERE / mask_name
        output_path = HERE / output_name
        mask = Image.open(mask_path).convert("L").resize(image.size, Image.BILINEAR)

        mask_arr = np.array(mask) < THRESHOLD

        out = np.zeros_like(image_arr)
        out[mask_arr] = image_arr[mask_arr]
        Image.fromarray(out, mode="RGB").save(output_path)
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()

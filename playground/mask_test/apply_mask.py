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
    ("mask_gpt5.4_high.png", "masked_gpt5.4_high.png"),
    ("mask_gptimage2_low.png", "masked_gptimage2_low.png"),
    ("mask_gptimage2_medium.png", "masked_gptimage2_medium.png"),
    ("mask_gptimage2_high.png", "masked_gptimage2_high.png"),
]
THRESHOLD = 128


def main() -> None:
    image = Image.open(IMAGE_PATH).convert("RGB")
    image_arr = np.array(image)

    for mask_name, output_name in MASKS:
        mask_path = HERE / mask_name
        output_path = HERE / output_name
        output_inv_path = output_path.with_name(f"{output_path.stem}_inv{output_path.suffix}")
        mask = Image.open(mask_path).convert("L").resize(image.size, Image.BILINEAR)

        mask_arr = np.array(mask) < THRESHOLD
        mask_arr_inv = np.array(mask) >= THRESHOLD

        out = np.zeros_like(image_arr)
        out[mask_arr] = image_arr[mask_arr]
        Image.fromarray(out, mode="RGB").save(output_path)
        print(f"saved: {output_path}")

        out_inv = np.zeros_like(image_arr)
        out_inv[mask_arr_inv] = image_arr[mask_arr_inv]
        Image.fromarray(out_inv, mode="RGB").save(output_inv_path)
        print(f"saved: {output_inv_path}")


if __name__ == "__main__":
    main()

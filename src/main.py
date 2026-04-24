from lib.example_left_mask import ExampleLeftMask
from lib.example_right_mask import ExampleRightMask
from lib.example_bottom_mask import ExampleBottomMask
from lib.gemini_mask import GeminiMask
from lib.gptimage_mask import GptImageMask
from lib.person_mask import PersonMask
from lib.sky_mask_segformer_b5 import SkyMaskSegformerB5
from lib.background_mask_rmbg2 import BackgroundMaskRmbg2

def example():
    path = "data/testdata/input_01.jpg"
    right_mask = ExampleRightMask(path)
    left_mask = ExampleLeftMask(path)
    bottom_mask = ExampleBottomMask(path)
    (~left_mask - bottom_mask).export("right_top_mask")

def gemini_example():
    path = "data/testdata/input_01.jpg"
    mask = GeminiMask(
        path,
        prompt="Create a mask that extracts everything except the sky.",
        label="sky",
    )
    mask.export("gemini_sky_mask")

def gptimage_example():
    path = "data/testdata/input_03.jpg"
    mask = GptImageMask(
        path,
        prompt="空部分のマスクを白黒で生成してください。",
        label="sky",
    )
    mask.export("gptimage_sky_mask")

def main():
    paths = [
        "data/testdata/input_01.jpg",
        "data/testdata/input_02.jpg",
        "data/testdata/input_03.jpg",
        "data/testdata/input_04.jpg",
        "data/testdata/input_05.jpg",
    ]
    PROMPT = "Image A is the direct edit target. Create a precise black-and-white mask of only the sky region in the photo. Output a clean segmentation mask with pure white (#FFFFFF) for the sky / visible open sky area, and pure black (#000000) for everything else, including the person, glasses, hair, jacket, shoulder strap, trees, branches, leaves, trunks, ground, sign, fence, and all other non-sky areas. Do not keep any original colors or photo texture. The result should be a flat binary mask only, matching the original image framing and dimensions."
    for path in paths:
        mask = GptImageMask(
            path,
            prompt=PROMPT,
            label="sky",
            quality="medium",
            size="1024x1024",
        )
        mask.export(f"gptimage_sky_mask_medium_1024x1024", apply_inv=True)

if __name__ == "__main__":
    main()
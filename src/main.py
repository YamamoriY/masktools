from lib.example_left_mask import ExampleLeftMask
from lib.example_bottom_mask import ExampleBottomMask
from lib.gemini_mask import GeminiMask
from lib.gptimage_mask import GptImageMask
from lib.gptsky_mask import GptSkyMask
from lib.person_mask import PersonMask
from lib.sky_mask_segformer_b5 import SkyMaskSegformerB5
from lib.background_mask_rmbg2 import BackgroundMaskRmbg2
from lib.trunk_mask_yolov11 import TrunkMaskYolov11

def example():
    path = "data/testdata/input_01.jpg"
    left_mask = ExampleLeftMask(path)
    bottom_mask = ExampleBottomMask(path)
    (~left_mask - bottom_mask).export("right_top_mask")

def sky_person_mask():
    paths = [
        "data/testdata/input_01.jpg",
        "data/testdata/input_02.jpg",
        "data/testdata/input_03.jpg",
        "data/testdata/input_04.jpg",
        "data/testdata/input_05.jpg",
    ]
    # qualities = ["low", "medium", "high"]
    qualities = ["low"]
    for quality in qualities:
        for path in paths:
            skymask = GptSkyMask(
                path,
                quality=quality,
                size="1024x1024",
            )
            person_mask = PersonMask(path)
            mask = skymask | person_mask
            mask.export(f"gptsky_person_mask_{quality}", apply_inv=True)

def trunk_mask_yolov11():
    paths = [
        "data/testdata/input_01.jpg",
        "data/testdata/input_02.jpg",
        "data/testdata/input_03.jpg",
        "data/testdata/input_04.jpg",
        "data/testdata/input_05.jpg",
    ]
    for path in paths:
        mask = TrunkMaskYolov11(path, conf=0.25)
        mask.export("trunk_mask_yolov11")

def main():
    trunk_mask_yolov11()

if __name__ == "__main__":
    main()
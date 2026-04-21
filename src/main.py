from lib.example_left_mask import ExampleLeftMask
from lib.example_right_mask import ExampleRightMask
from lib.example_bottom_mask import ExampleBottomMask
from lib.person_mask import PersonMask
from lib.sky_mask_segformer_b5 import SkyMaskSegformerB5
from lib.background_mask_rmbg2 import BackgroundMaskRmbg2

def example():
    path = "data/testdata/input_01.jpg"
    right_mask = ExampleRightMask(path)
    left_mask = ExampleLeftMask(path)
    bottom_mask = ExampleBottomMask(path)
    (~left_mask - bottom_mask).export("right_top_mask")

def main():
    paths = [
        "data/testdata/input_01.jpg",
        "data/testdata/input_02.jpg",
        "data/testdata/input_03.jpg",
        "data/testdata/input_04.jpg",
        "data/testdata/input_05.jpg",
    ]
    for path in paths:
        background = BackgroundMaskRmbg2(path)
        background.export("background_mask")
        (~background).export("foreground_mask")

if __name__ == "__main__":
    main()
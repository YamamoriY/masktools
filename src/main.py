from lib.mask import GeneratedMask
from lib.left_half_mask import LeftHalfMask
from lib.right_half_mask import RightHalfMask
from lib.bottom_half_mask import BottomHalfMask
from lib.person_mask import PersonMask

def example():
    path = "data/testdata/input_01.jpg"
    right_mask = RightHalfMask(path)
    left_mask = LeftHalfMask(path)
    bottom_mask = BottomHalfMask(path)
    (~left_mask - bottom_mask).save("left_bottom_mask")
    left_mask.export("left_mask")

def main():
    paths = [
        "data/testdata/input_01.jpg",
        "data/testdata/input_02.jpg",
        "data/testdata/input_03.jpg",
        "data/testdata/input_04.jpg",
        "data/testdata/input_05.jpg",
    ]
    for path in paths:
        mask = PersonMask(path)
        mask.export("person_mask")

if __name__ == "__main__":
    main()

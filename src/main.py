from lib.mask import GeneratedMask

def main():
    paths = [
        "data/testdata/input_01.jpg",
        "data/testdata/input_02.jpg",
        "data/testdata/input_03.jpg",
        "data/testdata/input_04.jpg",
        "data/testdata/input_05.jpg",
        "data/testdata/input_06.jpg",
    ]
    for path in paths:
        mask = GeneratedMask("testmask", path)
        print(mask.mask.shape)

if __name__ == "__main__":
    main()

import torch
from torchvision.io import read_image

from .mask import GeneratedMask


class ExampleRightMask(GeneratedMask):
    def __init__(self, input_path: str):
        super().__init__("example_right_mask", input_path)

    def _generate(self) -> torch.Tensor:
        image = read_image(str(self.input_path))
        mask = torch.zeros_like(image)
        w = image.shape[-1]
        mask[..., w // 2:] = 255
        return mask

import torch
from torchvision.io import read_image

from .mask import GeneratedMask


class WhiteMask(GeneratedMask):
    def __init__(self, input_path: str):
        super().__init__("white_mask", input_path)

    def _generate(self) -> torch.Tensor:
        image = read_image(self.input_path)
        return torch.full_like(image, 255)

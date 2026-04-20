import torch
from torchvision.io import read_image

from .mask import GeneratedMask


class BottomHalfMask(GeneratedMask):
    def __init__(self, input_path: str):
        super().__init__("bottom_half_mask", input_path)

    def _generate(self) -> torch.Tensor:
        image = read_image(str(self.input_path))
        mask = torch.zeros_like(image)
        h = image.shape[-2]
        mask[..., h // 2:, :] = 255
        return mask

from pathlib import Path

import torch
from torchvision.io import read_image, write_jpeg


class Mask:
    def __init__(self, mask: torch.Tensor):
        self.mask = mask

    def __or__(self, other: "Mask") -> "Mask":
        return Mask(torch.maximum(self.mask, other.mask))

    def __sub__(self, other: "Mask") -> "Mask":
        return Mask(torch.clamp(self.mask - other.mask, min=0))

    def __mul__(self, other: "Mask") -> "Mask":
        return Mask(torch.minimum(self.mask, other.mask))


class GeneratedMask(Mask):
    name: str
    input_path: Path
    cache_path: Path
    def __init__(self, name: str, input_path: str):
        self.name = name
        self.input_path = Path(input_path)
        self.cache_path = self.input_path.parent / "tmp" / name / self.input_path.name

        if self.cache_path.exists():
            mask = read_image(str(self.cache_path))
        else:
            mask = self._generate()
            self._save(mask)
        super().__init__(mask)

    def _generate(self) -> torch.Tensor:
        image = read_image(self.input_path)  # str を渡さなければいけないかも？
        return torch.full_like(image, 255)

    def _save(self, tensor: torch.Tensor) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        write_jpeg(tensor, str(self.cache_path))

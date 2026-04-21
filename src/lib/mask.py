from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torchvision.io import read_image, write_jpeg


class Mask:
    def __init__(self, mask: torch.Tensor, input_path: Path | None = None):
        self.mask = mask
        self.input_path = input_path

    def __or__(self, other: "Mask") -> "Mask":
        return Mask(torch.maximum(self.mask, other.mask), self.input_path or other.input_path)

    def __sub__(self, other: "Mask") -> "Mask":
        return Mask(torch.clamp(self.mask - other.mask, min=0), self.input_path or other.input_path)

    def __mul__(self, other: "Mask") -> "Mask":
        return Mask(torch.minimum(self.mask, other.mask), self.input_path or other.input_path)

    def __invert__(self) -> "Mask":
        return Mask(255 - self.mask, self.input_path)

    def save(self, name: str) -> None:
        if self.input_path is None:
            raise ValueError("input_path is not set; cannot derive save location")
        save_path = self.input_path.parent / "output" / name / self.input_path.name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        write_jpeg(self.mask, save_path)

    def apply(self, name: str) -> None:
        if self.input_path is None:
            raise ValueError("input_path is not set; cannot derive save location")
        image = read_image(self.input_path)
        applied = (image.to(torch.float32) * self.mask.to(torch.float32) / 255).to(torch.uint8)
        save_path = self.input_path.parent / "output" / name / "apply" / self.input_path.name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        write_jpeg(applied, save_path)

    def export(self, name: str) -> None:
        self.save(name)
        self.apply(name)


class GeneratedMask(Mask, ABC):
    name: str
    input_path: Path
    cache_path: Path

    def __init__(self, name: str, input_path: str):
        self.name = name
        self.input_path = Path(input_path)
        self.cache_path = self.input_path.parent / "tmp" / name / self.input_path.name

        if self.cache_path.exists():
            mask = read_image(self.cache_path)
        else:
            mask = self._generate()
            self._save(mask)
        super().__init__(mask, self.input_path)

    @abstractmethod
    def _generate(self) -> torch.Tensor:
        ...

    def _save(self, tensor: torch.Tensor) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        write_jpeg(tensor, str(self.cache_path))

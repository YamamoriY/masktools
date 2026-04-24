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

    # マスクを保存
    def save(self, name: str) -> None:
        if self.input_path is None:
            raise ValueError("input_path is not set; cannot derive save location")
        save_path = self.input_path.parent / "output" / name / self.input_path.name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        write_jpeg(self.mask, save_path)

    # マスクを適用した画像を出力（確認用）
    def apply(
        self,
        name: str,
        apply_inv: bool = True,    # 反転させたマスクの画像も出力
        bg_color: tuple[int, int, int] = (180, 0, 180),
    ) -> None:
        if self.input_path is None:
            raise ValueError("input_path is not set; cannot derive save location")
        image = read_image(self.input_path).to(torch.float32)
        alpha = self.mask.to(torch.float32) / 255
        bg = torch.tensor(bg_color, dtype=torch.float32).view(3, 1, 1)
        applied = (image * alpha + bg * (1 - alpha)).to(torch.uint8)
        save_path = self.input_path.parent / "output" / name / "apply" / self.input_path.name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        write_jpeg(applied, save_path)
        if apply_inv:
            inv_alpha = 1 - alpha
            applied_inv = (image * inv_alpha + bg * alpha).to(torch.uint8)
            inv_save_path = self.input_path.parent / "output" / name / "apply_inv" / self.input_path.name
            inv_save_path.parent.mkdir(parents=True, exist_ok=True)
            write_jpeg(applied_inv, inv_save_path)

    # マスクの保存と適用（確認用）
    # 迷ったらこれ
    def export(
        self,
        name: str,
        apply_inv: bool = True,
        bg_color: tuple[int, int, int] | None = None,
    ) -> None:
        self.save(name)
        if bg_color is None:
            self.apply(name, apply_inv)
        else:
            self.apply(name, apply_inv, bg_color)


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

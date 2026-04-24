"""gpt-image-2 を使ったマスク生成。

OpenAI Images API の edit を呼び出して入力画像+プロンプトから白黒マスクを
返させる。playground/mask_test の比較で gpt-image-2 が実用水準の品質を
出すことを確認したので、GeneratedMask として組み込む。
"""

import io

import numpy as np
import torch
from PIL import Image
from torchvision.io import read_image

from .llmclient import gptimage
from .mask import GeneratedMask


class GptImageMask(GeneratedMask):
    def __init__(
        self,
        input_path: str,
        prompt: str,
        label: str,
        quality: str = "medium",
        size: str = "1024x1024",
    ):
        self._prompt = prompt
        self._quality = quality
        self._size = size
        super().__init__(f"{label}_mask_gptimage2_{quality}_{size}", input_path)

    def _generate(self) -> torch.Tensor:
        image = read_image(str(self.input_path))
        c, h, w = image.shape

        pngs = gptimage.edit(
            image=self.input_path,
            prompt=self._prompt,
            size=self._size,
            quality=self._quality,
        )
        if not pngs:
            raise RuntimeError("gpt-image-2 returned no images")

        mask_pil = Image.open(io.BytesIO(pngs[0])).convert("L").resize((w, h), Image.BILINEAR)
        binary = (np.array(mask_pil, dtype=np.uint8) >= 127).astype(np.uint8) * 255

        mask = torch.from_numpy(binary)
        return mask.unsqueeze(0).expand(c, -1, -1).contiguous()

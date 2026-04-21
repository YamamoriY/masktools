from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.io import read_image
from transformers import AutoModelForImageSegmentation

from .mask import GeneratedMask

_MODEL_ID = "briaai/RMBG-2.0"
_CACHE_DIR = Path("models/rmbg-2")
_INFERENCE_SIZE = 1024  # RMBG-2.0 の訓練解像度


class BackgroundMaskRmbg2(GeneratedMask):
    _model: AutoModelForImageSegmentation | None = None
    _device: str = "cuda" if torch.cuda.is_available() else "cpu"
    _transform = transforms.Compose([
        transforms.Resize((_INFERENCE_SIZE, _INFERENCE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def __init__(self, input_path: str):
        super().__init__("background_mask_rmbg2", input_path)

    @classmethod
    def _get_model(cls) -> AutoModelForImageSegmentation:
        if cls._model is None:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            model = AutoModelForImageSegmentation.from_pretrained(
                _MODEL_ID,
                trust_remote_code=True,
                cache_dir=str(_CACHE_DIR),
            )
            cls._model = model.to(cls._device).eval()
        return cls._model

    def _generate(self) -> torch.Tensor:
        image = read_image(str(self.input_path))
        c, h, w = image.shape

        model = self._get_model()
        pil = Image.open(str(self.input_path)).convert("RGB")
        input_tensor = self._transform(pil).unsqueeze(0).to(self._device)

        with torch.no_grad():
            foreground_prob = model(input_tensor)[-1].sigmoid()  # (1, 1, H, W) in [0, 1]

        foreground = F.interpolate(
            foreground_prob, size=(h, w), mode="bilinear", align_corners=False
        ).squeeze(0).squeeze(0)  # (h, w)

        background = ((1.0 - foreground) * 255.0).to(torch.uint8).cpu()
        return background.unsqueeze(0).expand(c, -1, -1).contiguous()

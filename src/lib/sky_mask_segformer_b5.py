from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.io import read_image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from .mask import GeneratedMask

_MODEL_ID = "nvidia/segformer-b5-finetuned-ade-640-640"
_CACHE_DIR = Path("models/segformer-b5-ade")
_SKY_CLASS_ID = 2  # ADE20K (150 classes, reduced labels): 2 == "sky"


class SkyMaskSegformerB5(GeneratedMask):
    _model: SegformerForSemanticSegmentation | None = None
    _processor: SegformerImageProcessor | None = None
    _device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, input_path: str):
        super().__init__("sky_mask_segformer_b5", input_path)

    @classmethod
    def _get_model(cls) -> tuple[SegformerForSemanticSegmentation, SegformerImageProcessor]:
        if cls._model is None:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cls._processor = SegformerImageProcessor.from_pretrained(
                _MODEL_ID, cache_dir=str(_CACHE_DIR)
            )
            model = SegformerForSemanticSegmentation.from_pretrained(
                _MODEL_ID, cache_dir=str(_CACHE_DIR)
            )
            cls._model = model.to(cls._device).eval()
        return cls._model, cls._processor

    def _generate(self) -> torch.Tensor:
        image = read_image(str(self.input_path))
        c, h, w = image.shape

        model, processor = self._get_model()
        pil = Image.open(str(self.input_path)).convert("RGB")
        inputs = processor(images=pil, return_tensors="pt").to(self._device)

        with torch.no_grad():
            logits = model(**inputs).logits  # (1, 150, H/4, W/4)

        upsampled = F.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )
        predicted = upsampled.argmax(dim=1).squeeze(0)
        sky = ((predicted == _SKY_CLASS_ID).to(torch.uint8) * 255).cpu()

        return sky.unsqueeze(0).expand(c, -1, -1).contiguous()

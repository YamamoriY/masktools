from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.io import read_image
from ultralytics import YOLO

from .mask import GeneratedMask

_WEIGHTS_PATH = Path("models/yolo11n-trunk/yolov11-trunk-seg.pt")

class TrunkMaskYolov11(GeneratedMask):
    _model: YOLO | None = None

    def __init__(self, input_path: str, conf: float = 0.25):
        self._conf = conf
        super().__init__(f"trunk_mask_yolov11_conf{conf}", input_path)

    @classmethod
    def _get_model(cls) -> YOLO:
        if cls._model is None:
            cls._model = YOLO(str(_WEIGHTS_PATH))
        return cls._model

    def _generate(self) -> torch.Tensor:
        image = read_image(str(self.input_path))
        c, h, w = image.shape

        model = self._get_model()
        results = model.predict(
            str(self.input_path),
            conf=self._conf,
            verbose=False,
        )
        result = results[0]

        mask_2d = torch.zeros((h, w), dtype=torch.uint8)
        if result.masks is not None:
            per_instance = result.masks.data.float()
            resized = F.interpolate(
                per_instance.unsqueeze(1),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            union = resized.amax(dim=0) > 0.5
            mask_2d = (union.to(torch.uint8) * 255).cpu()

        return mask_2d.unsqueeze(0).expand(c, -1, -1).contiguous()

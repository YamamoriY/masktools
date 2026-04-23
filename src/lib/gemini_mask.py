"""Gemini 2.5 のネイティブ segmentation 機能を使ったマスク生成。

ユーザーの当初要望は nano banana (gemini-2.5-flash-image) だったが、
nano banana は画像生成/編集モデルで出力解像度が 512/1K/2K/4K に限定され
入力画像との pixel 整合性が保証されない。そのため同じ Gemini 2.5 系列の
vision-language model (gemini-2.5-flash) が提供するネイティブ segmentation
機能を使い、box_2d + base64 PNG mask の JSON を取り出してピクセル精度の
マスクを生成する。
"""

import base64
import io
import json
import re
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from torchvision.io import read_image

from .mask import GeneratedMask

_MODEL_ID = "gemini-2.5-flash"
_JSON_INSTRUCTION = (
    "Output a JSON list of segmentation masks where each entry contains the 2D "
    'bounding box in the key "box_2d", the segmentation mask in key "mask", and '
    'the text label in the key "label". Use descriptive labels.'
)
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class GeminiMask(GeneratedMask):
    _client: genai.Client | None = None

    def __init__(self, input_path: str, prompt: str, label: str):
        self._prompt = prompt
        super().__init__(f"gemini_mask_{label}", input_path)

    @classmethod
    def _get_client(cls) -> genai.Client:
        if cls._client is None:
            load_dotenv()
            cls._client = genai.Client()
        return cls._client

    def _generate(self) -> torch.Tensor:
        image = read_image(str(self.input_path))
        c, h, w = image.shape

        pil = Image.open(str(self.input_path)).convert("RGB")
        full_prompt = f"{self._prompt}\n\n{_JSON_INSTRUCTION}"

        client = self._get_client()
        response = client.models.generate_content(
            model=_MODEL_ID,
            contents=[full_prompt, pil],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )

        items = self._parse_items(response.text)
        canvas = np.zeros((h, w), dtype=np.uint8)
        for item in items:
            self._paint_item(canvas, item, h, w)

        binary = (canvas >= 127).astype(np.uint8) * 255
        mask = torch.from_numpy(binary)
        return mask.unsqueeze(0).expand(c, -1, -1).contiguous()

    @staticmethod
    def _parse_items(raw: str) -> list[dict]:
        cleaned = _JSON_FENCE_RE.sub("", raw.strip()).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Gemini response was not valid JSON: {raw!r}") from e
        if not isinstance(data, list):
            raise RuntimeError(f"Gemini response was not a JSON list: {raw!r}")
        return data

    @staticmethod
    def _paint_item(canvas: np.ndarray, item: dict, h: int, w: int) -> None:
        box = item.get("box_2d")
        mask_b64 = item.get("mask")
        if box is None or mask_b64 is None:
            return

        y0 = int(box[0] / 1000 * h)
        x0 = int(box[1] / 1000 * w)
        y1 = int(box[2] / 1000 * h)
        x1 = int(box[3] / 1000 * w)
        y0, y1 = sorted((max(0, y0), min(h, y1)))
        x0, x1 = sorted((max(0, x0), min(w, x1)))
        if y1 <= y0 or x1 <= x0:
            return

        if "," in mask_b64:
            mask_b64 = mask_b64.split(",", 1)[1]
        png_bytes = base64.b64decode(mask_b64)
        mask_pil = Image.open(io.BytesIO(png_bytes)).convert("L")
        mask_pil = mask_pil.resize((x1 - x0, y1 - y0), Image.BILINEAR)
        region = np.array(mask_pil, dtype=np.uint8)

        canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], region)

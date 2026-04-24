"""gpt-image-2 を使った空(sky)専用マスク生成。

GptImageMask のロジックをそのまま流用し、空抽出用のプロンプトと
label を組み込んだ専用クラス。
"""

from .gptimage_mask import GptImageMask

_DEFAULT_PROMPT = (
    "Image A is the direct edit target. Create a precise black-and-white mask "
    "of only the sky region in the photo. Output a clean segmentation mask "
    "with pure white (#FFFFFF) for the sky / visible open sky area, and pure "
    "black (#000000) for everything else, including the person, glasses, hair, "
    "jacket, shoulder strap, trees, branches, leaves, trunks, ground, sign, "
    "fence, and all other non-sky areas. Do not keep any original colors or "
    "photo texture. The result should be a flat binary mask only, matching "
    "the original image framing and dimensions."
)


class GptSkyMask(GptImageMask):
    def __init__(
        self,
        input_path: str,
        prompt: str = _DEFAULT_PROMPT,
        quality: str = "medium",
        size: str = "1024x1024",
    ):
        super().__init__(
            input_path=input_path,
            prompt=prompt,
            label="sky",
            quality=quality,
            size=size,
        )

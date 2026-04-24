"""gpt-image-2 疎通確認スクリプト。

OpenAI Images API (`client.images.edit`) を直接呼び出して
gpt-image-2 によるマスク(白黒)生成の挙動を確認する。
"""

import base64
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

_MODEL = "gpt-image-2"


def _get_client() -> OpenAI:
    load_dotenv()
    return OpenAI()


def generate(
    prompt: str,
    *,
    size: str = "1024x1024",
    quality: str = "low",
    n: int = 1,
) -> list[bytes]:
    preview = prompt if len(prompt) <= 80 else prompt[:77] + "..."
    print(
        f"[LLM] gptimage.generate model={_MODEL} size={size} quality={quality} "
        f"n={n} prompt={preview!r}"
    )
    result = _get_client().images.generate(
        model=_MODEL,
        prompt=prompt,
        size=size,
        quality=quality,
        n=n,
    )
    return [base64.b64decode(d.b64_json) for d in result.data]


def edit(
    image: str | Path,
    prompt: str,
    *,
    mask: str | Path | None = None,
    size: str = "1024x1024",
    quality: str = "low",
) -> list[bytes]:
    preview = prompt if len(prompt) <= 80 else prompt[:77] + "..."
    print(
        f"[LLM] gptimage.edit model={_MODEL} image={image} mask={mask} "
        f"size={size} quality={quality} prompt={preview!r}"
    )
    with open(image, "rb") as image_file:
        kwargs: dict = {
            "model": _MODEL,
            "image": image_file,
            "prompt": prompt,
            "size": size,
            "quality": quality,
        }
        if mask is not None:
            with open(mask, "rb") as mask_file:
                kwargs["mask"] = mask_file
                result = _get_client().images.edit(**kwargs)
        else:
            result = _get_client().images.edit(**kwargs)
    return [base64.b64decode(d.b64_json) for d in result.data]


def main() -> None:
    out_dir = Path("data/tmp")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"edit: data/testdata/input_03.jpg (no mask)")
    images = edit(
        image="data/testdata/input_03.jpg",
        prompt="空部分のマスクを白黒で生成してください。",
        size="1024x1024",
        quality="medium",
    )
    for i, png in enumerate(images):
        path = out_dir / f"gptimage_edit_{i:02d}.png"
        path.write_bytes(png)
        print(f"  saved: {path} ({len(png)} bytes)")


if __name__ == "__main__":
    main()

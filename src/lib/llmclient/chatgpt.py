"""ChatGPT クライアント。

OpenAI Responses API を使用し、テキスト・画像を送信して
応答テキストおよび生成画像 (PNG bytes) を受け取る。
"""

import base64
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

_DEFAULT_MODEL = "gpt-4.1-mini"


@dataclass
class ChatResponse:
    text: str
    images: list[bytes] = field(default_factory=list)


class ChatGPT:
    _client: OpenAI | None = None

    def __init__(self, model: str = _DEFAULT_MODEL, instructions: str | None = None):
        self._model = model
        self._instructions = instructions

    @classmethod
    def _get_client(cls) -> OpenAI:
        if cls._client is None:
            load_dotenv()
            cls._client = OpenAI()
        return cls._client

    def ask(
        self,
        text: str,
        images: list[str | Path] | None = None,
        generate_images: bool = False,
    ) -> ChatResponse:
        content: list[dict] = [{"type": "input_text", "text": text}]
        for image in images or []:
            content.append({
                "type": "input_image",
                "image_url": self._to_image_url(image),
            })

        kwargs: dict = {
            "model": self._model,
            "input": [{"role": "user", "content": content}],
        }
        if self._instructions is not None:
            kwargs["instructions"] = self._instructions
        if generate_images:
            kwargs["tools"] = [{"type": "image_generation"}]

        response = self._get_client().responses.create(**kwargs)
        generated = [
            base64.b64decode(out.result)
            for out in response.output
            if out.type == "image_generation_call" and out.result
        ]
        return ChatResponse(text=response.output_text, images=generated)

    @staticmethod
    def _to_image_url(image: str | Path) -> str:
        image_str = str(image)
        if image_str.startswith(("http://", "https://", "data:")):
            return image_str
        path = Path(image_str)
        mime, _ = mimetypes.guess_type(path.name)
        mime = mime or "image/jpeg"
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{b64}"

def main() -> None:
    client = ChatGPT(model="gpt-5")
    response = client.ask(
        text=(
            "空部分のマスクを白黒で生成してください。"
        ),
        images=["data/testdata/input_03.jpg"],
        generate_images=True,
    )
    print(response.text)

    out_dir = Path("data/tmp")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, png_bytes in enumerate(response.images):
        path = out_dir / f"chatgpt_out_{i:02d}.png"
        path.write_bytes(png_bytes)
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
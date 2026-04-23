"""ChatGPT クライアント。

OpenAI Responses API を使用し、テキスト・画像を送信して応答テキストを受け取る。
Responses API は 2026 年時点で OpenAI が新規プロジェクトに推奨する API。
"""

import base64
import mimetypes
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

_DEFAULT_MODEL = "gpt-4.1-mini"


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

    def ask(self, text: str, images: list[str | Path] | None = None) -> str:
        content: list[dict] = [{"type": "input_text", "text": text}]
        for image in images or []:
            content.append({
                "type": "input_image",
                "image_url": self._to_image_url(image),
            })

        kwargs = {
            "model": self._model,
            "input": [{"role": "user", "content": content}],
        }
        if self._instructions is not None:
            kwargs["instructions"] = self._instructions

        response = self._get_client().responses.create(**kwargs)
        return response.output_text

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

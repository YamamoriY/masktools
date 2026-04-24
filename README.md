## 概要
画像に対するマスクを扱うユーティリティ。（3DGS の前処理のために作成） \
全てのマスクは抽象クラス `Mask` を継承し、マスク同士は演算を行うことができます。

## 使用例
### 実行法
```
uv run src/main.py
```

### 例
``` python
# left_mask と bottom_mask の結合例
def main():
    path = "data/testdata/input_01.jpg"
    left_mask = ExampleLeftMask(path)
    bottom_mask = ExampleBottomMask(path)
    (left_mask | bottom_mask).export("right_top_mask")

if __name__ == "__main__":
    main()
```

## 実装済みマスク

| クラス | 抽出対象 | 手法 / モデル | 備考 |
|--------|----------|---------------|------|
| `ExampleLeftMask` | 左半分 | -  | 演算サンプル |
| `ExampleBottomMask` | 下半分 | -  | 演算サンプル |
| `PersonMask` | 人物 | YOLO11n-seg | 中程度の精度 |
| `SkyMaskSegformerB5` | 空 | SegFormer-B5 (ADE20K) | エッジが甘い |
| `BackgroundMaskRmbg2` | 背景 | RMBG-2.0 | 前景/背景分離。空抽出には不向き |
| `GeminiMask` | プロンプト指定 | Gemini 2.5 Flash | `label` で対象指定 |
| `GptImageMask` | プロンプト指定 | gpt-image-2 | `quality` (low/medium/high) と `size` 指定可 |
| `GptSkyMask` | 空 | gpt-image-2 (空プロンプト固定) | `GptImageMask` のラッパ。実用品質 |
| `TrunkMaskYolov11` | 木の幹 | YOLO11n-seg (自家 FT) | `conf` 指定可 |

## マスクの演算
演算子は常に新しい `Mask` を返し、`self` は変更しない。結果は両辺の `input_path` を引き継ぐ。

| 演算子 | 意味 | 実装 |
|--------|------|------|
| `a \| b` | 和(union) | `torch.maximum(a, b)` |
| `a * b` | 積(intersection) | `torch.minimum(a, b)` |
| `a - b` | 差(difference) | `clamp(a - b, min=0)` |
| `~a` | 反転 | `255 - a` |

```python
combined = (WhiteMask(path) - ExampleLeftMask(path)) * ExampleBottomMask(path)
combined.export("right_bottom_quarter")  # output/right_bottom_quarter/{ファイル名} に保存
```

## マスクの作り方
`GeneratedMask` を継承して `_generate()` を実装する。

```python
from src.lib.mask import GeneratedMask

class MyMask(GeneratedMask):
    def __init__(self, input_path: str):
        super().__init__("my_mask", input_path)

    def _generate(self) -> torch.Tensor:
        # 白(255)/黒(0) の tensor を返す
        ...
```

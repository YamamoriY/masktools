## 概要
- 画像に対するマスクを扱うユーティリティ群。
- 抽出対象が白(255)、その他を黒(0) とする。
- マスクは `torch.Tensor` で保持し、元画像パス (`input_path`) と紐付く。

## マスクの作り方
組み込みの具象クラス(`GeneratedMask` を継承)を使うか、自分で継承クラスを作る。

### 組み込みクラス
...

### 自作する場合
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

## マスクの演算
演算子は常に新しい `Mask` を返し、`self` は変更しない。結果は両辺の `input_path` を引き継ぐ。

| 演算子 | 意味 | 実装 |
|--------|------|------|
| `a \| b` | 和(union) | `torch.maximum(a, b)` |
| `a * b` | 積(intersection) | `torch.minimum(a, b)` |
| `a - b` | 差(difference) | `clamp(a - b, min=0)` |
| `~a` | 反転 | `255 - a` |

```python
combined = (WhiteMask(path) - LeftHalfMask(path)) * BottomHalfMask(path)
combined.save("right_bottom_quarter")  # output/right_bottom_quarter/{ファイル名} に保存
```
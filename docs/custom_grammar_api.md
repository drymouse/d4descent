# カスタム文法を追加するための API リファレンス

## 概要

このプロジェクトは PyTorch を用いた勾配降下法と離散的な書き換え（rewrite）を組み合わせた最適化フレームワークです。
新しい文法を追加するには、以下の2つの主要な抽象クラスを実装します。

| 抽象クラス | 役割 |
|---|---|
| `ObjectCollection[ObjectT]` | オブジェクト群のバッチ表現・レンダリング・パラメータ管理 |
| `Task[ObjectT, RewriteSpecT, StateT]` | 初期化・損失計算・離散提案・可視化の統合 |

---

## ファイル構成

```
src/d4descent/
├── object_collection.py      # ObjectCollection 基底クラス
├── tasks/_base.py            # Task, TaskArgs 基底クラス
├── losses/
│   ├── _base.py              # LossArgs
│   ├── raster.py             # RasterLossMixin (画像 MSE 損失)
│   └── sds.py                # SDSLossMixin (Score Distillation Sampling)
├── objects/
│   ├── tree.py               # Tree / TreeCollection の実装例
│   ├── arclines.py           # Shape / ShapeCollection の実装例
│   └── ur.py                 # UR / URCollection の実装例
└── optimizer.py              # optimize() エントリポイント
```

---

## Step 1: オブジェクトクラスの定義

### 1-1. 単一オブジェクト (`ObjectT`)

単一のオブジェクトを表すデータクラスです。PyTorch テンソルをフィールドとして持ちます。

```python
from dataclasses import dataclass, field
import torch
from d4descent.context import Context

@dataclass
class MyPayload:
    """ユーザー定義のメタデータ。自由に拡張可能。"""
    pass

@dataclass
class MyObject:
    # --- 最適化対象のパラメータ (torch.Tensor) ---
    params: torch.Tensor          # 例: (n, 2) など
    id: int = field(default_factory=lambda: Context.get().gen_id())
    payload: MyPayload = field(default_factory=MyPayload)
```

**`id` は `Context.get().gen_id()` で自動生成します。**

### 1-2. 書き換え仕様 (`RewriteSpecT`)

離散最適化で構造を変化させる操作を表します。

```python
from dataclasses import dataclass
from enum import Enum

class MyRewriteType(Enum):
    AddElement = 0
    RemoveElement = 1
    ModifyElement = 2

@dataclass
class MyRewriteSpec:
    rewrite_type: MyRewriteType
    args: tuple   # 操作に必要な引数
```

---

## Step 2: `ObjectCollection` の実装

`src/d4descent/object_collection.py` にある `ObjectCollection[ObjectT]` を継承します。

```python
from typing import Self, Union
from dataclasses import dataclass
import torch
from d4descent.object_collection import ObjectCollection

@dataclass
class MyCollection(ObjectCollection[MyObject]):
    # バッチ化したパラメータ (grad が必要なものは Tensor として保持)
    params: torch.Tensor  # (n_objects, ...) 全オブジェクトのパラメータ
    ids: tuple[int, ...]
    payloads: tuple[MyPayload, ...]
    # 各オブジェクトの範囲: indices[i] = (start, end)
    indices: tuple[tuple[int, int], ...]
```

### 必須メソッド一覧

#### `__len__() -> int`
コレクション内のオブジェクト数を返します。

```python
def __len__(self) -> int:
    return len(self.ids)
```

#### `device() -> torch.device`
パラメータが存在するデバイスを返します。

```python
def device(self) -> torch.device:
    return self.params.device
```

#### `rasterize(positions) -> torch.Tensor`
各オブジェクトの **符号付き距離場 (SDF)** を計算します。これが最も重要なメソッドです。

```python
def rasterize(self, positions: torch.Tensor) -> torch.Tensor:
    """
    Args:
        positions: (..., 2) — クエリ座標

    Returns:
        sdf: (n_objects, ...) — 各オブジェクトのSDF値
             負=内側, 正=外側 (convex の場合の符号)
    """
    ...
```

`render()` / `render01()` は `rasterize()` を呼び出す形で `Renderable` 基底クラスに実装済みです。
`render01()` は SDF を [0, 1] にブラー付きで変換します（損失計算・可視化で使用）。

#### `parameters() -> list[torch.Tensor]`
勾配計算の対象となるテンソルのリストを返します。

```python
def parameters(self) -> list[torch.Tensor]:
    return [self.params]
```

#### `parameter_names() -> list[str]`
パラメータ名のリストを返します（デバッグ用）。

```python
def parameter_names(self) -> list[str]:
    return ["params"]
```

#### `per_object_grads() -> list[torch.Tensor]`
各オブジェクトの勾配をまとめた1次元テンソルのリストを返します。
離散最適化の提案評価に使います。

```python
def per_object_grads(self) -> list[torch.Tensor]:
    grads = []
    for s, e in self.indices:
        grad = self.params.grad[s:e].flatten()
        grads.append(grad)
    return grads
```

#### `requires_grad_(requires_grad) -> Self`
パラメータの `requires_grad` を設定します（最適化開始前に呼ばれます）。

```python
def requires_grad_(self, requires_grad: bool = True) -> Self:
    self.params = self.params.detach().clone().requires_grad_(requires_grad)
    return self
```

#### `clone() -> Self`
パラメータを detach & clone したコピーを返します。

```python
def clone(self) -> Self:
    return MyCollection(
        params=self.params.detach().clone(),
        ids=self.ids,
        payloads=self.payloads,
        indices=self.indices,
    )
```

#### `to(device) -> Self`
デバイス移動します。

```python
def to(self, device=None) -> Self:
    return MyCollection(
        params=self.params.to(device=device),
        ids=self.ids,
        payloads=self.payloads,
        indices=self.indices,
    )
```

#### `get_object(idx, detach) -> MyObject`
インデックス指定で単一オブジェクトを取り出します。

```python
def get_object(self, idx: int, detach: bool = True) -> MyObject:
    s, e = self.indices[idx]
    p = self.params[s:e]
    if detach:
        p = p.detach()
    return MyObject(params=p, id=self.ids[idx], payload=self.payloads[idx])
```

#### `from_object(object, **kwargs) -> Self` (classmethod)
単一オブジェクトからコレクションを作成します。

```python
@classmethod
def from_object(cls, object: MyObject, **kwargs) -> "MyCollection":
    return cls(
        params=object.params.detach().clone(),
        ids=(object.id,),
        payloads=(object.payload,),
        indices=((0, len(object.params)),),
    )
```

#### `cat(collections, **kwargs) -> Self` (classmethod)
複数のコレクションを結合します。

```python
@classmethod
def cat(cls, collections: list["MyCollection"], **kwargs) -> "MyCollection":
    offset = 0
    all_params = []
    all_ids = ()
    all_payloads = ()
    all_indices = []
    for c in collections:
        n = len(c.params)
        all_params.append(c.params)
        all_ids += c.ids
        all_payloads += c.payloads
        all_indices.extend((s + offset, e + offset) for s, e in c.indices)
        offset += n
    return cls(
        params=torch.cat(all_params, dim=0),
        ids=all_ids,
        payloads=all_payloads,
        indices=tuple(all_indices),
    )
```

### オプションメソッド

#### `project_to_valid_() -> Self`
勾配更新後にパラメータを有効な範囲にクランプします（例: 正のサイズを保証）。

```python
def project_to_valid_(self) -> Self:
    self.params.data.clamp_(min=0)
    return self
```

#### `scale_grads_() -> bool`
勾配をスケーリングします。`True` を返すとオプティマイザがスケーリング済みとみなします。

```python
def scale_grads_(self) -> bool:
    return False  # デフォルト: スケーリングなし
```

#### `get_sizes() -> list[int]`
各オブジェクトの要素数リストを返します（メトリクス用）。

```python
def get_sizes(self) -> list[int]:
    return [e - s for s, e in self.indices]
```

---

## Step 3: `Task` の実装

`src/d4descent/tasks/_base.py` にある `Task[ObjectT, RewriteSpecT, StateT]` を継承します。
`StateT` は最適化ステップをまたいで保持される状態の型です（不要なら `None`）。

```python
from d4descent.tasks._base import Task, RenderArgs, ExtraMetrics
from d4descent.losses.raster import RasterLossMixin, RasterLossArgs
import numpy as np

class MyTask(RasterLossMixin[MyObject, MyRewriteSpec, None],
             Task[MyObject, MyRewriteSpec, None]):
    def __init__(self, args: MyTaskArgs, render_args: RenderArgs,
                 raster_args: RasterLossArgs, target_img: torch.Tensor):
        Task.__init__(self, render_args)
        RasterLossMixin.__init__(self, raster_args, target_img)
        self.args = args
        self._device = target_img.device
```

### 必須メソッド一覧

#### `device() -> torch.device`

```python
def device(self) -> torch.device:
    return self._device
```

#### `get_collection_constructor() -> type[MyCollection]`
コレクションのクラスを返します。

```python
def get_collection_constructor(self) -> type[MyCollection]:
    return MyCollection
```

#### `initialize_object() -> MyObject`
最適化の開始点を作成します。

```python
def initialize_object(self) -> MyObject:
    return MyObject(
        params=torch.zeros((1, 2), device=self.device()),
    )
```

#### `_compute_losses(collection, state) -> tuple[Tensor, ExtraMetrics]`
連続最適化の損失を計算します。

- `RasterLossMixin` を使う場合: 自動的に `render01()` と MSE を計算するため **不要**
- `SDSLossMixin` を使う場合も同様

カスタム損失を使う場合は以下のように実装します:

```python
def _compute_losses(
    self, collection: ObjectCollection[MyObject], state: None
) -> tuple[torch.Tensor, ExtraMetrics]:
    imgs = collection.render01(
        self.render_args.size,
        self.render_args.lim,
        center_pixel=self.render_args.center_pixel,
        blur=self.render_args.blur,
    )  # (n_objects, size, size)

    loss = torch.mean((imgs - self.target_img).square().flatten(-2), dim=-1)  # (n_objects,)
    return loss, {}
```

#### `compute_simplicity(collection) -> list[float]`
各オブジェクトの複雑さを返します（離散最適化の正則化項）。

```python
def compute_simplicity(self, collection: ObjectCollection[MyObject]) -> list[float]:
    return [len(obj.params) * self.args.element_weight for obj in collection]
```

#### `make_proposals(obj) -> tuple[ObjectCollection, list[RewriteSpec]]`
離散的な書き換え提案を生成します。

```python
def make_proposals(self, obj: MyObject) -> tuple[ObjectCollection[MyObject], list[MyRewriteSpec]]:
    specs = obj.gen_rewrite_specs(self.args.rewrite_args)  # 例
    rewritten = [obj.apply_rewrite(spec) for spec in specs]
    return MyCollection.from_objects(rewritten), specs
```

#### `combine_proposals(base, proposals, base_loss, proposal_losses, proposal_specs, accept_parallel) -> tuple[MyObject, bool]`
損失が改善した提案を選択・統合します。

```python
def combine_proposals(
    self,
    base: MyObject,
    proposals: ObjectCollection[MyObject],
    base_loss: float,
    proposal_losses: list[float],
    proposal_specs: list[MyRewriteSpec],
    accept_parallel: bool = True,
) -> tuple[MyObject, bool]:
    candidates = [
        (base_loss - loss, i)
        for i, loss in enumerate(proposal_losses)
        if base_loss - loss > self.args.better_abs_eps
    ]
    candidates.sort(reverse=True)

    if not candidates:
        return base, False

    best_spec = proposal_specs[candidates[0][1]]
    return base.apply_rewrite(best_spec), True
```

#### `initialize_state() -> StateT`
最適化の状態を初期化します（不要なら `None`）。

```python
def initialize_state(self) -> None:
    return None
```

#### `cleanup(collection) -> ObjectCollection`
オブジェクトの後処理（不要な要素の削除など）を行います。

```python
def cleanup(self, collection: ObjectCollection[MyObject]) -> ObjectCollection[MyObject]:
    return collection  # 何もしない場合
```

#### `visualize(collection, step, loss, state) -> np.ndarray`
可視化用の画像を返します。

```python
def visualize(self, collection, step, loss, state) -> np.ndarray:
    from d4descent.visualizer import MPLVisualizer
    fig = MPLVisualizer(1, 1, 10.8, 10.8,
                        xlim=self.render_args.lim, ylim=self.render_args.lim)
    ax = fig[0]
    # ... 描画処理 ...
    return fig.get_image()
```

### オプションメソッド

#### `update_state_for_proposals(state, proposals) -> StateT`
提案評価の前に状態を更新します（提案用に SDS のテキスト埋め込みを拡張するなど）。

#### `step_state(state) -> StateT`
各ステップ後に状態を更新します（アニーリングなど）。

---

## Step 4: `TaskArgs` の定義

`TaskArgs` は `Task` のファクトリクラスです。YAML 設定と連携します。

```python
from dataclasses import dataclass, field
from typing import Optional
import torch
from d4descent.tasks._base import TaskArgs, RenderArgs, Task
from d4descent.losses._base import LossArgs
from d4descent.losses.raster import RasterLossArgs, RasterLossMixin
from d4descent.losses.sds import SDSLossArgs, SDSLossMixin

@dataclass
class MyTaskArgs(TaskArgs):
    element_weight: float = 1e-3
    better_abs_eps: float = 1e-8
    rewrite_args: MyRewriteArgs = field(default_factory=MyRewriteArgs)
    target_img: Optional[torch.Tensor] = None

    def create(
        self,
        render_args: RenderArgs,
        loss_args: LossArgs,
        device,
        target_img: Optional[torch.Tensor] = None,
    ) -> Task:
        if isinstance(loss_args, RasterLossArgs):
            assert target_img is not None
            self.target_img = target_img
            return MyRasterTask(self, render_args, loss_args, target_img)
        elif isinstance(loss_args, SDSLossArgs):
            return MySDSTask(self, render_args, loss_args, device=device)
        else:
            raise NotImplementedError(f"Unknown loss_args: {type(loss_args)}")
```

---

## Step 5: 損失ミックスイン

既存の損失を再利用するには多重継承を使います。

### `RasterLossMixin` — 画像 MSE 損失

`render01()` の出力と `target_img` の MSE を計算します。

```python
class MyRasterTask(RasterLossMixin[MyObject, MyRewriteSpec, None],
                   MyTask[None]):
    def __init__(self, args, render_args, raster_args, target_img):
        MyTask.__init__(self, args, render_args, target_img.device)
        RasterLossMixin.__init__(self, raster_args, target_img)
```

### `SDSLossMixin` — Score Distillation Sampling

テキストプロンプトで駆動する Stable Diffusion ベースの損失です。

```python
class MySDSTask(SDSLossMixin[MyObject, MyRewriteSpec, None],
                MyTask[None]):
    def __init__(self, args, render_args, sds_args, device):
        MyTask.__init__(self, args, render_args, device)
        SDSLossMixin.__init__(self, sds_args)
```

---

## Step 6: 最適化の実行

```python
from d4descent.optimizer import optimize, OptimizeArgs
from d4descent.tasks._base import RenderArgs
from d4descent.losses.raster import RasterLossArgs

render_args = RenderArgs(size=256, lim=(-1.5, 1.5))
loss_args = RasterLossArgs()
task_args = MyTaskArgs()

task = task_args.create(render_args, loss_args, device="cuda", target_img=target)
optimize_args = OptimizeArgs(n_steps=4000, lr=0.5, propose_every=50)

best_obj, best_loss, best_collection, metrics = optimize(task, optimize_args)
```

---

## クラス継承ツリー

```
Renderable (ABC)
└── ObjectCollection[ObjectT] (ABC)
    ├── TreeCollection
    ├── ShapeCollection (arclines)
    ├── URCollection (unionrect)
    └── MyCollection           ← 実装対象

Task[ObjectT, RewriteSpecT, StateT] (ABC)
├── RasterLossMixin            ← _compute_losses を提供
├── SDSLossMixin               ← _compute_losses を提供
├── TreeTask / TreeRasterTask / TreeSDSTask
├── ArcLinesTask / ...
└── MyTask / MyRasterTask / MySDSTask  ← 実装対象

TaskArgs (ABC)
├── TreeArgs
├── ArcLinesArgs
└── MyTaskArgs                 ← 実装対象
```

---

## 実装チェックリスト

### ObjectCollection

- [ ] `__len__`
- [ ] `device`
- [ ] `rasterize(positions) -> (n_objects, ...)` の SDF
- [ ] `parameters() -> list[Tensor]`
- [ ] `parameter_names() -> list[str]`
- [ ] `per_object_grads() -> list[Tensor]`
- [ ] `requires_grad_`
- [ ] `clone`
- [ ] `to`
- [ ] `get_object`
- [ ] `from_object` (classmethod)
- [ ] `cat` (classmethod)
- [ ] `project_to_valid_` (任意)

### Task

- [ ] `device`
- [ ] `get_collection_constructor`
- [ ] `initialize_object`
- [ ] `_compute_losses` (損失ミックスインを使う場合は不要)
- [ ] `compute_simplicity`
- [ ] `make_proposals`
- [ ] `combine_proposals`
- [ ] `initialize_state`
- [ ] `cleanup`
- [ ] `visualize`

### TaskArgs

- [ ] `create(render_args, loss_args, device, target_img) -> Task`

---

## 重要な型・定数

| 型 | 定義場所 | 説明 |
|---|---|---|
| `ObjectCollection[ObjectT]` | `object_collection.py` | コレクション基底クラス |
| `Task[O, R, S]` | `tasks/_base.py` | タスク基底クラス |
| `TaskArgs` | `tasks/_base.py` | タスクファクトリ基底クラス |
| `RenderArgs` | `tasks/_base.py` | `size`, `lim`, `blur` などのレンダリング設定 |
| `LossArgs` | `losses/_base.py` | 損失設定基底クラス |
| `RasterLossArgs` | `losses/raster.py` | 画像MSE損失の設定 |
| `SDSLossArgs` | `losses/sds.py` | SDS損失の設定 (prompt 必須) |
| `OptimizeArgs` | `optimizer.py` | 最適化ループ全体の設定 |
| `ExtraMetrics` | `tasks/_base.py` | `dict[str, tuple[float, ...]]` |

---

## SDF の符号規約

`rasterize()` が返す値は **符号付き距離場 (Signed Distance Field)** です。

- **負の値**: 点がオブジェクトの内側
- **正の値**: 点がオブジェクトの外側
- **0**: オブジェクトの境界

`render01()` はこれを `[0, 1]` にブラー付きで変換します（1 = 内側、0 = 外側）。

```python
# render01 の実装 (object_collection.py より)
vlim = blur * (lim[1] - lim[0]) / size
imgs = self.render(size, lim)         # SDF (負=内側)
imgs = (-imgs.clamp(-vlim, vlim) + vlim) / (2 * vlim)  # [0, 1]
```

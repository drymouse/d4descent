# d4descent コード解析

## 1. 概要

「Design for Descent: What Makes a Shape Grammar Easy to Optimize?」(SIGGRAPH Asia 2025) の実装。**形状文法(shape grammar)** で表現されたパラメトリックな2D図形を、**連続最適化(勾配降下)** と **離散的な文法書き換え(rewrite/proposal)** を交互に繰り返すことで目標(画像・トポロジー最適化・SDSプロンプト)に近づけるフレームワーク。

3つの組み込み文法:
- `arclines`(`src/d4descent/objects/arclines.py`) — 線分・円弧からなるループ図形
- `tree`(`tree.py`) — 階層的な枝分かれ構造
- `ur` (UnionRect, `ur.py`) — 回転矩形の和集合(穴あき対応)

3つの目的(loss):`raster`(画像へのラスタライズ一致)、`sds`(Stable DiffusionのScore Distillation Sampling)、`topopt`(トポロジー最適化)。

## 2. アーキテクチャ

```
Object (dataclass: Shape / Tree / UR)
   ├─ tensor パラメータ (xs, sizes, rots, ...)  ← Context.gen_id() で id 付与
   ├─ gen_rewrite_specs() : 離散書き換え候補を生成
   ├─ apply_rewrite() / apply_all_rewrites() : 書き換えを適用
   ├─ cleanup() : 退化したノードの除去・整理
   └─ visualize()

ObjectCollection[Object] (バッチ化されたテンソルの集合; arclines/tree/ur 毎に実装)
   ├─ parameters() / parameter_names() : Optimizer に渡す葉テンソル
   ├─ rasterize(positions) : 符号付き距離関数 (SDF) を計算 → render()/render01()
   ├─ from_object / cat / batchify : バッチ生成・分割
   └─ project_to_valid_() : パラメータを有効域へクランプ

Task[Object, RewriteSpec, State] (tasks/*.py。grammar × loss の組合せごとに具象クラス)
   ├─ TaskArgs.create(render_args, loss_args, device, target_img) で適切な Task を構築
   ├─ _compute_losses() (Loss Mixin から提供)
   ├─ compute_simplicity() : ノード数等に基づく簡潔性ペナルティ
   ├─ make_proposals_ex() : Object.gen_rewrite_specs/apply_rewrite を呼んで候補集合を生成
   └─ combine_proposals() : 候補の損失を比較し採用するか決定

Loss Mixin (losses/raster.py, sds.py, topopt.py)
   └─ Task に多重継承され _compute_losses() を実装(画像差分 / SDS / トポロジー感度解析)

optimizer.py: optimize(task, OptimizeArgs) … メインループ
```

`Context`(`context.py`)はグローバルなID発生器。`use_context()` で最適化中だけ有効化され、各Objectは生成時に一意な `id` を持つ。

## 3. 最適化アルゴリズムの挙動(`optimizer.py: optimize()`)

各 `step` (`0..n_steps`) で:

1. **クリーンアップ**: `step % cleanup_every == 0` で `task.cleanup()` を呼び、退化ノード(サイズ0の矩形など)を除去。
2. **書き換え判定 (`rewrite`)**:
   - `proposal_trigger == "step"`: `step % propose_every == 0` で発火
   - `proposal_trigger == "rel_loss"`: 損失の改善が `proposal_patience` ステップ停滞したら発火
3. **書き換えが発火した場合**:
   - 早期終了判定(`stopping_patience`, `stopping_eps`)。改善が一定回数見られなければ最適化を打ち切る。
   - 現在のオブジェクトから `make_proposals_ex()` で離散候補(`proposal_size` 個、0なら全候補)を生成。
   - 候補群を `batchify()` で `batch_param_count`(または `batch_size`)ごとにバッチ分割し、各バッチに対して `proposal_steps` 回の勾定降下を実行(`proposal_criterion`に応じ評価方法が変わる: `"loss"`=実際に数ステップ最適化して損失比較、`"grad"/"grad_only"`=勾配の内積で1次近似評価)。
   - `compute_simplicity()` を加味した損失で各候補をスコアリングし、`combine_proposals()` で改善する候補(複数可、`proposal_accept_parallel`)を元のオブジェクトに統合。
   - 採用後、学習率を `increase_lr_after_proposal` / `reset_lr_after_proposal` の設定に従って調整。
4. **連続最適化(通常ステップ)**:
   - `compute_losses()` → `loss.backward()` → `clip_grad`(`abs`/`rel` モード)→ `optimizer.step()` → `project_to_valid_()`。
   - 移動平均 (`MovingAverage`) で損失を平滑化し、スケジューラ (`ReduceLROnPlateau` / `AdaptiveLR` / `LinearLR` / `ExponentialLR`) を更新。
   - `task.step_state()` で外部状態(例: トポロジー最適化のPID制御器)を更新。
5. 各ステップの損失・簡潔性・学習率・タイムスタンプ・追加メトリクスを記録し、最後に `(best_object, best_loss, all_objects, all_metrics)` を返す。

### `OptimizeArgs` (主要パラメータ一覧)

| フィールド | 既定値 | 意味 |
|---|---|---|
| `n_steps` | 4000 | 総ステップ数 |
| `optimizer` | `"SGD"` | `"Adam"` / `"SGD"` |
| `scheduler` | `"ReduceLROnPlateau"` | `"none"` / `"ReduceLROnPlateau"` / `"AdaptiveLR"` / `"LinearLR"` / `"ExponentialLR"` |
| `lr` | 0.5 | 学習率(初期/最大値) |
| `clip_grad` / `clip_grad_mode` | 2.0 / `"abs"` | 勾配クリップ閾値とモード(`"rel"`は`lr`で割る) |
| `reduce_lr_factor`, `reduce_lr_patience`, `reduce_lr_min_lr` | 0.5, 2, 1e-4 | LR減衰系のパラメータ |
| `increase_lr_patience`, `increase_lr_after_proposal`, `reset_lr_after_proposal` | 2, True, False | 書き換え採用後のLR回復挙動(`AdaptiveLR`時) |
| `cleanup_every` | 10 | クリーンアップ間隔(ステップ) |
| `proposal_trigger` | `"step"` | 書き換えのトリガー方式 |
| `propose_every` | 50 | `trigger=="step"`時の書き換え間隔 |
| `proposal_rel_loss`, `proposal_patience` | 5e-3, 10 | `trigger=="rel_loss"`時の停滞判定 |
| `proposal_criterion` | `"loss"` | 候補評価方法: `"loss"` / `"grad"` / `"grad_only"` |
| `proposal_steps` | 2 | `"loss"`評価時、各候補に行う勾定降下回数 |
| `proposal_size` | 0 | サンプリングする候補数(0=全候補) |
| `proposal_clip_grad`, `proposal_accept_parallel` | True, True | 候補評価/採用時の挙動 |
| `batch_param_count` / `batch_size` | 8192 / None | 候補バッチ化基準(パラメータ数 or 件数) |
| `w_simplicity` | 1.0 | 簡潔性ペナルティの重み |
| `visualize_every` | 10 | 可視化コールバック間隔 |
| `enable_satisfy_constraints`, `satisfy_constraints_args` | False, — | 制約満足のための追加最適化(lr/steps/debug) |
| `stopping_eps`, `stopping_patience` | 5e-3, None | 早期終了条件 |

## 4. 学習パラメータをいじるための設定システム

### 4.1 `confify` (CLI/YAML 設定ライブラリ)

すべての設定はPythonの `dataclass` で定義され、CLIまたはYAMLから流し込まれる(`scripts/optimize_shc.py` のような `read_config_from_cli(Args)`)。

- `--a.b.c value` … ドット区切りパスでフィールドを上書き(値は文字列としてパースされ、型に応じて int/float/bool/Enum 等に解釈)
- `---a.b.c path.yaml` … 指定パスにYAMLファイルの内容をマージ(`configs/tasks/*.yaml`, `configs/losses/*.yaml` をこの形で読み込む)
- ポリモーフィックなフィールド(`TaskArgs`, `LossArgs` などABCを継承する抽象型)はYAML/dict中の `$type: module.path.ClassName` キーで具象クラスを解決する(`confify/parser.py` の `_parse_impl`)。これが `configs/tasks/ur.yaml` の `$type: d4descent.tasks.ur.URArgs` のような書き方の正体。

### 4.2 トップレベル設定 (`scripts/optimize_shc.py: Args`)

```python
task: TaskArgs            # ---task configs/tasks/{arclines,tree,ur}.yaml
loss: LossArgs             # ---loss configs/losses/{raster,sds,topopt*}.yaml
save_path: Path
render: RenderArgs         # size, lim, center_pixel, blur
target_points_path: Optional[Path]   # 目標形状(.shc ファイル)
optim: OptimizeArgs        # 上記パラメータ群
device: str
restart: bool
skip / until: int          # バッチ処理の範囲指定
```

`scripts/optimize_pngs.py` / `optimize_prompts.py` も同様の構成で、目標が画像/プロンプトに変わるだけ。

### 4.3 実験生成スクリプト (`runs/_rungen_*.py`)

`confify.builder.CLIBuilder` を使い、`b.add(key, value)` でベース引数、`b.add_sweep_set({suffix: {key: value, ...}, ...})` でスイープ集合(直積で全組合せのジョブ名・引数列を生成)を構築。`b.build()` の結果から `runs/_generated/<grammar>/<JobName>.sh` を生成する(これが `git status` にある `runs/_generated/...` の出自)。

例 (`runs/_rungen_ur.py`):
```python
b.add("--task.rewrite_args.merge_threshold", 100)
b.add_sweep_set({"UR-F": {}, "UR-1": {"--task.rewrite_args.add_rect_weight": 0}, ...})
```
→ `UR-F_*`, `UR-1_*` などの組合せ分シェルスクリプトが生成され、それぞれ `uv run python scripts/optimize_shc.py --task.rewrite_args... ...` を実行する。

### 4.4 文法固有のパラメータ(例: `URArgs` / `URRewriteArgs`)

- `URRewriteArgs`: `split_h_weight`, `split_v_weight`, `add_rect_weight`, `remove_rect_weight`, `merge_weight`, `add_hole_weight`(各書き換え種別の重み=出現確率の比重)、`rect_scale`, `hole_scale`, `merge_threshold`, `remove_threshold`
- `URArgs`: `node_weight`(簡潔性ペナルティ係数)、`cleanup_*`(退化除去/分割/統合の閾値・戦略)、`ur_args.theta_scale`/`offset_scale`

`arclines`/`tree` も同様に `*RewriteArgs`(各書き換え操作の重み)と `*CollectionArgs`(座標スケール等)を持つ。

## 5. 自作の文法(grammar)を追加する方法

新しい形状文法を追加するには、既存の `ur.py`(最も単純な例)をテンプレートに、以下を実装する。

### Step 1: `src/d4descent/objects/<grammar>.py` — Object 定義

```python
@dataclass
class MyShape:
    params: torch.Tensor  # 1つ以上のテンソル(勾定降下対象)
    id: int = field(default_factory=lambda: Context.get().gen_id())
    payload: MyPayload = field(default_factory=MyPayload)

    def visualize(self, ax, ...): ...
    def cleanup(self, ...) -> "MyShape": ...                  # 退化要素の除去
    def gen_rewrite_specs(self, args, num_rewrites, lim, ...) -> list[MyRewrite]: ...  # 離散候補の生成
    def apply_rewrite(self, spec: MyRewrite, ...) -> "MyShape": ...        # 単一候補の評価用(複製+1書き換え)
    def apply_all_rewrites(self, specs: list[MyRewrite], scores: list[float], ...) -> "MyShape": ...  # 複数候補を1つに統合
```

書き換え種別は `Enum` + `@dataclass` の階層(`URRewriteType` / `URRewriteSplitH` 等)で表現するのが既存パターン。

### Step 2: 同ファイル — `*RewriteArgs`, `*CollectionArgs` dataclass

各書き換え操作の重み・スケールパラメータをまとめる(confifyでCLI上書き可能にするため)。

### Step 3: `MyShapeCollection(ObjectCollection[MyShape])`

`ObjectCollection`(`object_collection.py`)の抽象メソッドをすべて実装する:
- `from_object`, `from_objects`(継承可), `cat` — バッチ生成・結合
- `get_object` — インデックスから単体取得
- `parameters()`, `parameter_names()` — Optimizerに渡す葉テンソル
- `per_object_grads()` — オブジェクト単位の勾定(プロポーザル評価で使用)
- `requires_grad_()`, `clone()`, `to()`, `device()`
- `rasterize(positions)` — SDFを計算、`(n_shapes, ...)` を返す(これが `render()`/`render01()` の基盤になり、loss計算で使われる)
- 任意: `project_to_valid_()`(パラメータのクランプ)、`scale_grads_()`(勾定スケーリング)

**省力化したい場合**: バッチ化が単純(オブジェクトごとに別々のテンソル数で良い)なら `StdObject` / `StdCollection`(`object_collection.py` 末尾)という素朴な実装を継承すれば、`Collection` クラス自体を1から書く必要はない(`torch.jit.fork` で各オブジェクトを並列rasterizeする)。

### Step 4: `src/d4descent/tasks/<grammar>.py` — Task 定義

```python
@dataclass
class MyGrammarArgs(TaskArgs):
    rewrite_args: MyRewriteArgs = field(default_factory=MyRewriteArgs)
    collection_args: MyCollectionArgs = field(default_factory=MyCollectionArgs)
    node_weight: float = 1e-6   # compute_simplicity 用

    def create(self, render_args, loss_args, device, target_img=None) -> "Task":
        if isinstance(loss_args, RasterLossArgs):
            return MyGrammarRasterTask(self, render_args, loss_args, target_img)
        elif isinstance(loss_args, SDSLossArgs):
            return MyGrammarSDSTask(self, render_args, loss_args, device)
        elif isinstance(loss_args, TopoptArgs):
            return MyGrammarTopoptTask(self, render_args, loss_args, device)
        raise NotImplementedError

class MyGrammarTask(Task[MyShape, MyRewrite, StateT]):
    def device(self) -> torch.device: ...
    def get_collection_constructor(self) -> type[MyShapeCollection]: ...
    def initialize_object(self) -> MyShape: ...                 # 初期形状
    def compute_simplicity(self, collection) -> list[float]: ...
    def make_proposals_ex(self, obj, num_proposals) -> tuple[Collection, list[Spec]]:
        specs = obj.gen_rewrite_specs(...)
        return Collection.from_objects([obj.apply_rewrite(s, ...) for s in specs]), specs
    def combine_proposals(self, base, proposals, base_loss, proposal_losses, specs, accept_parallel) -> tuple[MyShape, bool]:
        # base_loss と proposal_losses を比較し、改善する書き換えを base.apply_all_rewrites() で統合
        ...
    def cleanup(self, collection) -> Collection: ...

# loss は Mixin との多重継承で取得(RasterLossMixin / SDSLossMixin / TopoptXXXMixin)
class MyGrammarRasterTask(RasterLossMixin[MyShape, MyRewrite, None], MyGrammarTask[None]):
    def initialize_state(self) -> None: return None
    def visualize(self, collection, step, loss, state) -> np.ndarray: ...  # MPLVisualizer でPNG/フレーム生成
```

`Task`(`tasks/_base.py`)の抽象メソッド一式 (`device`, `get_collection_constructor`, `initialize_object`, `_compute_losses`, `compute_simplicity`, `make_proposals`, `combine_proposals`, `initialize_state`, `cleanup`, `visualize`) を満たす必要がある。`_compute_losses` はLoss Mixin側が提供するので、Task自身は通常実装不要。

トポロジー最適化に対応させたい場合は `TopoptMixin`(`losses/topopt.py`)を継承し、`initialize_object`(境界条件別の初期形状)と `compute_constraints` を実装する。

### Step 5: `configs/tasks/<grammar>.yaml` を追加

```yaml
$type: d4descent.tasks.<grammar>.MyGrammarArgs
```
(必要ならネストしたフィールドの既定上書きも同ファイルに追記可)

### Step 6: 実行用スクリプト

- 既存の `scripts/optimize_shc.py`(目標図形ファイル `.shc` に対して最適化)や `optimize_pngs.py`(画像目標)、`optimize_prompts.py`(SDSプロンプト目標)はグラマー非依存(`TaskArgs.create()` で分岐)なので、**そのまま再利用できる**。CLIで `---task configs/tasks/<grammar>.yaml` を指すだけで良い。
- スイープ実験をしたい場合は `runs/_rungen_<grammar>.py` を `runs/_rungen_ur.py` を参考に作成し、`CLIBuilder` でパラメータ・スイープを定義してシェルスクリプトを生成する。

### 留意点

- 各 Object は生成時に `Context.get().gen_id()` でユニークIDを取得する(最適化ループ内は `Context(save_objects=True)` が `use_context()` で有効化されている)。
- `rasterize()` は符号付き距離関数(負=内側)を返す設計を守ること。`render01()` がこれをブラー付きの0–1占有率画像に変換し、`raster`/`sds` lossが使う。`topopt` lossは生のSDF (`render()`) を直接使う。
- `compute_simplicity()` は連続最適化の損失には加えず、書き換え候補の採否判定(離散側)でのみ `w_simplicity` 重み付きで加算される。

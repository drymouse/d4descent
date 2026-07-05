# City文法 × SDS損失 実装指示書

City文法（`src/d4descent/objects/city.py` / `src/d4descent/tasks/city.py`、道路幅は単一種類）を、
画像一致（RasterLoss）の代わりに **SDS損失**（Stable Diffusion のテキストプロンプト）で最適化できるようにする。
「手設計の都市らしさ損失 vs 学習済み拡散モデルの事前分布」という比較実験が目的。発表は明日なので、
**新規コードは最小限**（既存の `URSDSTask` パターンの移植のみ）とし、既存ファイルの挙動は一切変えない。

参照すべき既存実装（この3つを見れば迷わない）:

- `src/d4descent/tasks/ur.py:201`〜 `URSDSTask` — SDS タスクの正確なテンプレート（継承順・`__init__` の呼び方）
- `src/d4descent/tasks/ur.py:45`〜 `URArgs.create` — `loss_args` の型で分岐するディスパッチ
- `src/d4descent/losses/sds.py` — `SDSLossMixin`。`render01` → `(1-img)`（白地に黒線）→ `compute_sds_loss` まで全部やってくれる

---

## Step 1: `src/d4descent/tasks/city.py` の変更（これが唯一のライブラリ変更）

### 1-a. import 追加

```python
from ..losses.sds import SDSLossMixin, SDSLossArgs
```

### 1-b. `CityArgs.create`（city.py:59〜）に分岐追加

`ur.py:52-61` と同型。既存の Raster 分岐は触らない:

```python
if isinstance(loss_args, RasterLossArgs):
    ...  # 既存のまま
elif isinstance(loss_args, SDSLossArgs):
    return CitySDSTask(self, render_args, loss_args, device)
else:
    raise NotImplementedError(...)
```

`scripts/optimize_prompts.py` は `task.create(render, loss, device, None)` と **target_img=None** で呼ぶので、
SDS 分岐が target_img を要求しないことが重要（既存の assert は Raster 分岐内にあるのでそのままで良い）。

### 1-c. `CitySDSTask` クラス新設

`CityRasterTask`（city.py:389〜）の直後に置く。書くべきメソッドは4つだけ:

```python
class CitySDSTask(SDSLossMixin[CityNetwork, CityRewrite, None], CityTask[None]):
    def __init__(self, args: CityArgs, render_args: RenderArgs,
                 sds_args: SDSLossArgs, device: Union[str, torch.device]):
        CityTask.__init__(self, args, render_args, device)
        SDSLossMixin.__init__(self, sds_args)   # ここで SD がロードされる（GPU必須・初回~5GB DL）

    def initialize_state(self) -> None:
        return None

    def compute_losses(self, collection, state):
        # CityRasterTask.compute_losses (city.py:424-432) と同じ構造:
        # SDSLossMixin._compute_losses + size_weight * 建設コスト
        losses, xtra = self._compute_losses(collection, state)
        assert isinstance(collection, CityNetworkCollection)
        if self.args.size_weight != 0.0:
            losses = losses + self.args.size_weight * collection.get_construction_costs()
        return losses, xtra

    def visualize(self, collection, step, loss, state) -> np.ndarray:
        # CityRasterTask.visualize (city.py:434-448) をコピーし、
        # self.target_img を使う imshow の1行（city.py:441）だけ削除する。
        # ★SDSタスクに target_img は存在しないため、消し忘れると AttributeError で落ちる
        ...
```

注意点:

- **継承順は `SDSLossMixin` が先**（`URSDSTask` と同じ）。`_compute_losses` が Mixin 側で解決される必要がある。
- `get_add_anywhere_targets` はオーバーライドしない。基底 `CityTask` が `None` を返し、
  AddAnywhere は `lim` 全体から一様サンプルになる（SDS ではキャンバス全体が対象なのでこれで正しい。
  `_precompute_add_anywhere_targets` は target_img 前提なので使えない）。
- **`__main__` の抽象メソッドチェック（city.py:452-453）に `CitySDSTask` を追加しないこと**。
  コンストラクタが Stable Diffusion 本体（約5GB）をロードしてしまう。

## Step 2: 実行スクリプト `runs/_rungen_city_prompts.py` 新設

`runs/_rungen_ur_prompts.py` をコピーして以下を差し替える。それ以外の構造（CLIBuilder、
`_generated/` への .sh 出力）はそのまま:

| 箇所 | 値 |
|---|---|
| ジョブ名 | `CT-F_SDS` |
| `---` | `configs/sds_600.yaml`（既存。SDSLossArgs + render.size 600） |
| `---task` | `configs/tasks/city.yaml` |
| UR固有の行 | `add_hole_weight` / `ur_args.*` / `cleanup_strategy` / `node_weight` の行は**全部削除** |
| gen_dir | `city_prompts` |
| 呼ぶスクリプト | `scripts/optimize_prompts.py`（無変更で流用） |

optim 設定は UR の値を出発点にし、時間短縮のため steps だけ削る:

```python
b.add("--optim.proposal_trigger", "step")
b.add("--optim.propose_every", 75)
b.add("--optim.proposal_size", 32)   # UR は 64。VRAM が苦しければさらに 16 へ
b.add("--optim.scheduler", "none")
b.add("--optim.lr", 0.01)
b.add("--optim.batch_size", 4)
b.add("--optim.n_steps", 800)        # UR は 1500。まず 800 で回して映像を確保する
```

### プロンプト（まずこの4本。1本目が本命）

```python
b.add("--prompts", [
    "street map of a city",
    "road network of manhattan, grid streets",
    "radial street map of paris",
    "street map of a medieval european city",
])
b.add("--prompt_suffix", "top-down map, black lines on white background")
```

`SDSLossMixin` は `1 - render01`（= 白地に黒の道路）を SD に渡すので、「black lines on white background」
という suffix はレンダリングの見た目と一致する。出力が塊状に崩れる場合は
`--loss.neg_prompt "photo, 3d render, shading, buildings"` を足す。

### City 固有のパラメータ

- `--task.cost_weight`: デフォルト 1e-4 は **Raster の MSE スケール（~1e-2）向けに調整された値**。
  SDS の損失値はスケールが全く違う（数十〜数百になり得る）ので、そのままだと単純さ罰則が実質ゼロになり
  エッジ数が爆発する恐れがある。**まず 1e-4 で1本回してログの loss とエッジ数 E を見て、
  E が数百を超えて増え続けるなら 1e-3 → 1e-2 と上げる**（sweep するなら `b.add_sweep_set` で
  `{1e-4, 1e-3}` の2値）。
- `--task.city_collection_args.width`: デフォルト 0.05（lim=(-1.5,1.5)・600px で約10px幅の線）。
  SD の VAE が認識するには十分な太さなので**変更不要**。細くしたい場合も 0.03 未満にしない
  （VAE で潰れて勾配が消える）。
- `--task.decimate_dense`: まずデフォルト（True, cell 0.1）のまま。プロンプトが格子模様を要求しているのに
  中央が間引かれてスカスカになる場合のみ `false` を試す。
- `--render.blur`: `sds_600.yaml` 側の設定に任せる。UR は `1/math.sqrt(2)` を明示していたので、
  同じ行 `b.add("--render.blur", 1 / math.sqrt(2))` を残しておく。

## Step 3: 動作確認と実行手順

1. **配線チェック（GPU不要・数秒）**: `uv run python -c "from d4descent.tasks.city import CitySDSTask"` が通ること。
   ※インスタンス化はしない（SD ロードが走るため）。
2. `uv run python runs/_rungen_city_prompts.py` で `.sh` が生成されること。
3. **GPU マシン上で** 生成された `.sh` に `--optim.n_steps 20 --restart true` を付けて smoke test
   （SD のダウンロード込みで初回10分程度）。`visualize` が落ちないこと・loss が出力されることを確認。
4. 本番実行。1プロンプトあたりの所要時間は smoke test の 20 step から線形外挿して見積もる。
   出力は `output/CT-F_SDS.../<prompt名>/` に `last.png` と `video.mp4`（発表にそのまま使える）。

### ⚠ 計算資源（最重要の制約）

**この開発マシンには NVIDIA GPU が無い**（確認済み）。SDS は Stable Diffusion 2.1 base（fp16 で VRAM 約5GB＋
提案バッチ分）を毎ステップ回すので CPU では現実的に不可能。選択肢:

- 研究室の GPU サーバ / Slurm（`scripts/optimize_prompts.py` は Slurm のシグナルハンドラ登録済みで、
  そのまま投げられる想定の作りになっている）
- Google Colab（無料 T4 16GB で足りる）: リポジトリを clone → `uv sync` → 生成済み `.sh` を実行

VRAM 不足（OOM）が出たら効く順に: `--optim.proposal_size 16` → `--render.size 512`（`configs/sds_600.yaml` を
コピーして `sds_512.yaml` を作る）→ `--optim.batch_size 2`。

## 既知のリスクと対処（発表前にハマりやすい順）

1. **visualize の target_img 消し忘れ** → smoke test の最初の可視化で即落ちる。Step 1-c 参照。
2. **エッジ数の爆発**（cost_weight が SDS スケールに対して小さすぎる）→ 1e-3〜1e-2 に上げる。
3. **SDS 損失は確率的**（呼ぶたびに乱数 t とノイズが変わる）。提案の採否（`better_abs_eps=1e-8`）が
   ノイズを拾って書き換えが暴れることがあるが、論文の UR/ArcLines も同条件で動いているので
   まずはそのまま。明らかに毎ステップ構造が振動するなら `--task.better_abs_eps 0.1` 程度に上げる。
4. **地図らしくならない**（塊・べた塗りになる）→ neg_prompt 追加、`guidance_scale`（`--loss.sd.guidance_scale`、
   デフォルト100）を 50 に下げる、プロンプトを "map" 系に寄せる、の順で試す。

## 発表用の実験メニュー（優先順）

- **E1（必須・保険）**: "street map of a city" 1本、800 steps。`video.mp4` が「線1本から都市地図が育つ」
  映像になるのでこれだけでも発表が成立する。
- **E2（本命の主張）**: プロンプト4本の最終画像を横並びに。「格子(Manhattan) / 放射(Paris) / 中世」の
  様式差が出れば、"CityEngine が道路パターンをハードコードしていたものをテキストで指定できる" という主張になる。
- **E3（あれば）**: 既存の RasterLoss 版 City の結果（`runs/_rungen_city_pngs.py` の出力）と並べ、
  「画像一致（外形指定） vs 事前分布（様式指定）」の対比スライドを1枚。追加実行は不要、手持ちの結果を使う。

時間が無いときは E1 だけ確保 → 残り時間で E2 のプロンプトを1本ずつ追加、の順で回すこと
（`optimize_prompts.py` はプロンプトごとに完結して保存するので途中打ち切りでも成果物は残る）。

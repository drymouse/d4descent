# 道路網文法（Road Grammar）実装計画

## 背景・参考文献

- Parish & Müller, *Procedural Modeling of Cities* (CityEngine, SIGGRAPH 2001)
  - 拡張L-system: `ideal successor` → `globalGoals`（大域目標がパラメータを決定）→ `localConstraints`（局所制約で調整・FAILED判定）
  - 道路は highway（人口密度ピークを結ぶ）と street（highway 間を人口密度に沿って埋める）の2種
  - self-sensitive L-system: 道路端が既存道路に交差/接近 → 交差点生成・延長（＝本文法の「スナップ」に相当）
- Petrasch, *Prozedurale Städtegenerierung mit Hilfe von L-Systemen* (TU Dresden, 2008)
  - 上記手法のドイツ語での再実装レポート。道路網→街区→敷地→建物の一連の実装詳細を持つ（本文法では道路網のみが対象）

d4descent 側では、これらの論文の「道路網がヒューリスティックに人口密度から生成される」プロセスを逆転し、**道路網（最適化対象）→ 人口密度の予測値を合成するforward model** を定義し、target の人口密度マップとの差を勾配降下＋離散書き換えで最小化する。

## 1. プリミティブ設計：グラフ表現

既存文法との違い:

| 文法 | トポロジー | 頂点共有の扱い |
|---|---|---|
| `Tree` | 木構造。ノード位置は親からの相対極座標（長さ+角度）で計算 | 親子関係は配列インデックスで暗黙的 |
| `Shape`（arclines） | 開放/閉ループ | プリミティブの `start`/`end` を **Tensorオブジェクトの同一性** で共有、`MergeClose` で貼り替え可能 |
| **Road（本計画）** | **一般グラフ（ループ可）** | ノード座標をテーブルとして持ち、エッジはノードindexのペアで参照 |

道路網はループ（交差点による環状構造）を持ちうるため `Tree` の木構造は採用しない。ノード座標配列 + エッジ（ノードindexペア + 幅）という明示的グラフ表現を採用する。

### 道路幅は離散固定値（最適化対象外）

道路幅 `width` は勾配最適化の対象にせず、あらかじめ決めた離散クラス（例: `STREET = 0.02`, `HIGHWAY = 0.05`）から選ぶ固定値とする。

- `RoadNetwork.widths` は `torch.Tensor`（勾配計算用の中間値としては使うが `requires_grad=False`）または単純な `tuple[float, ...]` として保持し、`parameters()` / `per_object_grads()` には含めない
- どの幅クラスを使うかは書き換え（`AddFree` / `AddFromNode`）の側で決定する。具体的には、新規道路を追加する候補ごとに **幅クラスの数だけ提案を複製**し（例: street版・highway版を両方提案）、損失が良い方を `combine_proposals` が選ぶ。これは `ArcLines` の `ToLine`/`ToArc` のように「離散的な選択肢を提案として列挙し、連続最適化ではなく書き換え選択に委ねる」既存パターンと同じ考え方
- 幅を変更する専用の書き換え（例: street→highway への格上げ）は今回のマイルストーンには含めない（必要になれば later で追加）

### データクラス（案）

```python
@dataclass
class RoadPayload:
    pass

@dataclass
class RoadNetwork:
    nodes: torch.Tensor              # (n_nodes, 2) ノード座標。勾配対象
    widths: torch.Tensor             # (n_edges,) 道路幅。離散固定値、勾配対象外
    edges: tuple[tuple[int, int], ...]  # (n_edges,) (start_node_idx, end_node_idx) 静的トポロジー
    id: int = field(default_factory=lambda: Context.get().gen_id())
    payload: RoadPayload = field(default_factory=RoadPayload)

@dataclass
class RoadNetworkCollection(ObjectCollection[RoadNetwork]):
    nodes: torch.Tensor              # (total_nodes, 2) 全ネットワーク分を結合。勾配対象
    widths: torch.Tensor             # (total_edges,) 勾配対象外
    edges: tuple[tuple[int, int], ...]   # 結合後のグローバルnode index
    edge_index_of: torch.Tensor      # (total_edges,) 各エッジがどのネットワーク(=object)に属するか
    node_indices: tuple[tuple[int, int], ...]  # 各ネットワークのノード範囲 (start, end)
    edge_indices: tuple[tuple[int, int], ...]  # 各ネットワークのエッジ範囲 (start, end)
    ids: tuple[int, ...]
    payloads: tuple[RoadPayload, ...]
```

`ObjectCollection` の1オブジェクト＝1つの道路網（=1つの都市の候補案）。離散書き換えの提案評価では、複数の道路網候補（例:「あるエッジを1本追加した版」を複数）を1つの `RoadNetworkCollection` にまとめてバッチ評価する（`Tri`/`Tree` と同じ運用）。

### SDF（`rasterize`）

各エッジをカプセル形状として距離計算し、`Tri`/`Shape` と同様に `scatter_reduce(..., reduce="amin")` でネットワーク単位に集約する。

```python
def rasterize(self, positions):
    # 1. 線分までの距離 - 幅/2 を全エッジについて計算 -> (total_edges, ...)
    # 2. scatter_reduce(amin) で edge_index_of によりネットワーク単位に集約 -> (n_networks, ...)
    ...
```

## 2. 書き換え操作

| 操作 | 内容 | 対応する既存文法の実装 |
|---|---|---|
| `AddFree` | 新規ノード2個 + エッジ1本を空間中に追加（起点となる孤立道路） | `Tree.AddAnywhere` に類似 |
| `AddFromNode(node_id)` | 既存ノードから新規ノード+エッジを伸ばす。分岐は既存ノードの次数が増えるだけなので専用操作は不要 | `Tree.AddBranch` |
| `RemoveEdge(edge_id)` | 末端（次数1）のエッジを削除し、孤立したノードも掃除 | `Tree.RemoveBranch` |
| `Snap(edge_id, target_node_id)` | エッジの一端（次数1の"ぶら下がった"端点）を、近傍の既存ノードに張り替えてループ/交差点を形成 | `Shape.MergeClose`（端点の張り替え） |

- `AddFree` / `AddFromNode` は幅クラスの数だけ提案を複製する（上記「道路幅は離散固定値」参照）
- `make_proposals` は `Tri`/`Tree` の `make_proposals_ex` に倣い、`torch.cat` をO(1)回に抑えたバッチ実装にする（`docs/perf_improvements.md` 参照）

### 2.1 Snap候補探索アルゴリズム

Snap は「次数1の“ぶら下がった”端点」ごとに、半径 `snap_radius` 以内にある既存ノードを探す処理。単純な全ペア総当たりは O(n²) になるため、以下のいずれかで高速化する。

**推奨: 一様グリッドによる空間ハッシュ（追加の依存ライブラリ不要）**

1. セルサイズを `snap_radius` に設定し、全ノードの座標を `cell = floor((pos - origin) / snap_radius)` で整数バケットに変換
2. バケットをキーにした辞書 `dict[(int,int), list[node_idx]]` を構築（O(n)）
3. 各ダングリング端点について、自分のバケットと周囲8近傍（計3×3セル）に属するノードだけを距離チェック（`snap_radius` 以内なら候補）
4. ノードが空間的に偏って密集しない限り平均 O(n) で完了する

これは `Shape.resolve_intersections` が行っているバウンディングボックスでの絞り込みと同じ発想で、既存コードのスタイルに合う。

**代替案: KD-tree（`scipy.spatial.cKDTree`）**

`cKDTree(all_node_positions).query_ball_point(dangling_positions, r=snap_radius)` で同様の近傍探索が O(n log n) で行える。ノード分布が極端に偏る場合や実装をより頑健にしたい場合の代替。ただし `scipy` は現状 `pyproject.toml` の依存に含まれていないため、追加が必要になる点に注意（一様グリッド法で十分なら不要）。

まずは一様グリッド法で実装し、大規模ネットワークで問題が出た場合にKD-treeへ切り替える方針とする。

## 3. 目的関数：人口密度の合成（カスタム `_compute_losses`）

`RasterLossMixin`（二値占有率のMSE）はそのまま使わず、道路網から人口密度場を合成するforward modelを自作する。target の入力経路自体は raster タスク（`_pngs`）や `_shc` と同じ `[0,1]` の画像（`target_img`）をそのまま流用するが、**値そのものは `target_inside_value`/`target_outside_value` で実際の密度レンジへアフィン変換してから使う**（3.0節）。

### 3.0 target値のレンジ変換（pngs・shc共通）

`target_img` は入力元（PNG画像 or `ShapeCollection.render01()`）によらず `[0,1]` に近い値で渡ってくるが、これを人口密度としてそのまま使うと「形状の中=1(最大密度)、外=0(密度ゼロ)」という極端な二値になりやすい（shcの場合は`render01`がほぼ厳密な二値、pngsでも黒背景/白背景の画像だと同様）。特に外側が厳密に0だと、そこに道路を敷く動機が一切生まれない（道路を敷くとtarget=0との誤差がむしろ増える）。

そこで `RoadDensityTask.__init__` で必ず次のアフィン変換を適用する（pngs・shcのどちらでも同じロジック）:

```
effective_target = target_outside_value + (target_inside_value - target_outside_value) * target_img
```

デフォルトは `target_inside_value=0.7`, `target_outside_value=0.15`——形状の外側にも非ゼロの目標密度を持たせることで、そこにも(幹線道路程度の疎な)道路網が伸びる動機を作る。この変換は常に適用される標準の仕組みであり、「二値のまま使う」という特別扱いは無い（`target_inside_value=1, target_outside_value=0` にすれば従来の二値相当に戻せるが、それを既定にはしない）。

### 3.1 エッジごとの被覆度

```
d_e(x)      = relu(sdf_e(x))                        # 道路外側の距離（内側は0）
sigma_e     = sigma0 + k_sigma * w_e^reach_exponent # 幅が太い道ほど到達範囲(reach)が広い
amplitude_e = min(amp_scale / sigma_e, 1)           # reachが広いほどピーク強度は下がる
c_e(x)      = amplitude_e * exp(-(d_e(x) / sigma_e)^2)  # 値域 (0, amplitude_e]
```

`amplitude_e = amp_scale / sigma_e` は「断面積(amplitude×sigma)がほぼ一定」になるような正規化で、幹線道路（幅広→sigma大→amplitude小）は「薄く広く」、街路（幅狭→sigma小→amplitude大）は「狭く大きく」効くようにする、というユーザー要望をそのまま数式化したもの。`reach_exponent > 1` にすると `w_e` の指数が効くため、幅の広い道路(幹線道路)の到達半径だけが不釣り合いに拡大し、幅の狭い道路(街路)は`sigma0`付近に留まる（幹線道路をさらに広く薄くしたいという追加要望に対応）。

### 3.2 複数エッジの合成：重ね合わせではなく最大値

当初は複数エッジを確率的OR（`1 - Π(1-c_e(x))`）で重ね合わせていたが、これだと近くに幹線道路と街路が両方あるとき、必ず両方の寄与が加算されてしまい、「街路の寄与の方が強ければ街路を採用する」という選択的な挙動にならない。ユーザー要望により**各点で最も寄与の大きいエッジを採用する（最大値）**方式に変更した：

```
density(x) = max_e c_e(x)
```

`scatter_reduce(..., reduce="amax")` で実装する（`amin`ベースの`rasterize`と対称的な形）。この方式なら、ある点が幹線道路の直上（距離0）にあっても、少し離れた街路の`c_e`の方が大きければ、街路の値が採用される。実際に検証済み: 幹線道路の内側(dist=0, c_e≈0.23)にいても、0.05離れた街路(c_e≈0.31)の方が値が大きければ街路が採用されることを数値確認した。

### 3.3 最低ライン（"保証" vs "罰則" の間で1往復した）

最低ラインの実装は以下の経緯で2転した:

1. **最初の案**: `density = baseline + (1 - baseline) * raw(x)` として forward model に無条件加算。しかし `density` が数式上 `baseline` を絶対に下回れず、「下回った場合のペナルティ」を追加しようにも常にゼロになる問題があった。
2. **2番目の案**: 無条件加算をやめ、`compute_density` は `raw(x)` をそのまま返し、代わりに損失側で `underflow_weight * relu(min_density_floor - density)^2` という非対称な罰則を課した。しかしこれは**保証にならない**——道路が届かない場所（背景など）で床を満たすにはそこまで道路を敷く必要がありコストが見合わないため、実際にはモデルは罰則を払うだけで済ませてしまい、密度はほぼ0のまま放置される。ユーザーからの指摘で発覚。
3. **最終案（現在）**: 「保証」を優先し、`compute_density` の出力に対して無条件の下限を再度課す。ただし複数エッジの合成が「最大値」方式（3.2節）になったことに合わせて、加算ではなく `max` で床を適用する:

```
raw(x)     = max_e c_e(x)
density(x) = max(raw(x), min_density_floor)
```

`min_density_floor` は `RoadCollectionArgs`（`compute_density` を計算する場所）に戻した。この形であれば `density` は数式上どんな状況でも `min_density_floor` を下回れないため、真の意味で「保証」になる。損失側の非対称罰則（`underflow_weight`）は不要になったため削除し、通常のMSEに戻した。

**教訓**: 「無条件で下回れない」＝保証、「下回ったら罰則」＝ソフトな目標、は両立しない概念。今回のように「道路が届かない場所ではコストが見合わず罰則を払うだけで済ませられる」ケースでは、ソフトな罰則では実質的に機能しない。保証したいなら無条件の下限（`clamp`/`max`）を使うべき。

**`min_density_floor` だけでは「街の外側にも道路を敷きたい」は実現できない点に注意**: `min_density_floor` は道路の有無に関わらず無条件で保証される値なので、これ単体では道路網に「外側にも道路を敷こう」という動機を一切与えない（道路を敷いてもこの保証値は変わらず、target=0の場所ではむしろ悪化する）。「外側にも道路を敷いてほしい」という要求には、3.0節の `target_outside_value` のように **target自体を底上げする**必要がある。`min_density_floor`（forward modelの無条件保証・道路非依存）と `target_outside_value`（targetの底上げ・道路を敷く動機を作る）は役割が異なる、独立した2つの仕組み。

### 3.4 損失

```python
def _compute_losses(self, collection, state):
    density = collection.compute_density(size, lim, center_pixel)  # (n_networks, size, size)。floor保証済み
    loss = (density - self.target_img).square().flatten(-2).mean(dim=-1)
    return loss, {}
```

`compute_density` は `rasterize`（`amin`集約、可視化・境界判定用）とは別のメソッドとして `RoadNetworkCollection` に実装する（集約方法が `amin` ではなく「街路/幹線道路のうち寄与最大のものを採用＋床でclamp」のため）。ただしエッジごとのカプセルSDF計算自体は両者で共通化できる。`self.target_img` は既存の raster タスクと同じ仕組み（`RasterLossArgs`/`target_img` 経由）でロードした `[0,1]` 画像をそのまま使う。**`img_mode` は `bow`（黒=前景を高密度として反転）を使うこと** — `wob` のままだと画像の余白（背景）が高密度、ロゴ/市街地形状が低密度に読み込まれ、意図と正反対になる。

## 4. 正則化（複雑さ＝建設コスト）

```
cost = sum_e length_e * width_e ^ cost_width_exponent * cost_weight
```

道路の長さ×幅（≒舗装面積）をコストとみなす。`cost_width_exponent > 1` にすると、幅が広い道路（幹線道路）への罰則が幅に対して超線形に強くなる（デフォルト `2.5`。単純な線形だと `width_classes` の比率分しか差がつかず、幹線道路が乱立しやすかったため導入）。`Tri` の `node_weight`/`size_weight` に相当する役割を `cost_weight`（と `cost_width_exponent`）が担う。`compute_simplicity` で各ネットワークについて集計する。

## 4.1 都市らしさの評価軸：ループ形成（meshedness）への報酬

密度一致とコストだけでは道路網の位相（トポロジー）に選好がなく、密度勾配を追って根本から放射状に伸びる木構造（クモの巣状）になりやすい。Parish & Müllerの論文にも "ほとんどの道路は他の道路と交差するかループになって終わる。行き止まりは例外" とあり、Petraschの実装でも道路網から街区ポリゴンを抽出する前提としてサイクル検出を行っている——つまり**都市であるためには閉じたループ(街区)の存在が本質的**。

これをグラフ理論の meshedness（alpha index）としてスコア化し、`compute_simplicity` に負のコスト（報酬）として加える:

```
V = 道路網を構成する生きたノード数(次数>0。孤立ノードは除く)
E = エッジ数
cycles     = max(E - V + 1, 0)     # 閉路数。連結成分が1つの場合は厳密値、複数ある場合は下限値（近似）
meshedness = cycles / max(2V - 5, 1)  # 平面グラフが取りうる最大閉路数に対する比。木構造で0

simplicity = cost * cost_weight - meshedness * mesh_weight
```

`RoadNetworkCollection.get_meshedness()` として実装（`E`, `V` の集計のみで済み、面(街区)の実検出は不要なので安価）。`cycles` の計算は連結成分数を1と仮定した近似（`AddFree`で複数の孤立した部分網ができている場合は真値よりわずかに小さく見積もられるが、`add_free_weight`は低いデフォルトなので実用上は問題にならない想定）。

数値確認済み: V=4のノード集合で木構造(E=3)は`simplicity`寄与≈+4.6e-7（ほぼ0）だが、そこに1本足してループを作る(E=4)と`simplicity`寄与≈-0.0167（エッジが1本増えてコストは上がっているのに、meshedness報酬がそれを大きく上回り正味で有利になる）。

## 4.2 都市らしさの評価軸：交差点の角度（鋭角の罰則）

密度一致・コスト・meshednessだけでは道路の**向き**に選好がなく、交差点で道路同士がほぼ同じ方向を向いて鋭角に交わる（針のような不自然な形状）ことがある。これを罰するため、各ノードで接続する道路の方向を角度順に並べ、隣接方向どうしの角度差(gap)が閾値を下回った分だけ罰則を与える。

```
各ノードの周りの道路の方向を角度順に並べる (次数dなら d個のgapがあり、合計は必ず2π)
gap_i が min_angle を下回ったら (min_angle - gap_i)^angle_penalty_exponent だけ罰する
```

`RoadNetworkCollection.get_angle_penalty(min_angle, exponent)` として実装。ソートは `(center_node, theta)` の複合キーで一括 `argsort` し、隣接差分から"内部gap"を求め、周回分の最後のgapは `2π - 内部gapの総和` として計算する（ノードごとの明示的なループを避け、`scatter_add`/`scatter_reduce` だけで完結させている）。

他の正則化（コスト・meshedness）は `compute_simplicity`（離散書き換えの採否にのみ影響）に入れているが、**角度罰則は `_compute_losses` に直接加算する**——連続最適化（勾配降下）でノード位置そのものを動かして角度を改善してほしいため。次数1以下のノード（角度が定義できない）は対象外。

数値検証済み: 中心ノードから10°で開く2本の道 vs 90°で開く2本の道を比較（`min_angle=45°`）——鋭角側は罰則≈0.373、広角側は罰則=0（45°を超えているため）。勾配も正しく流れ、鋭角の枝を開く方向に力がかかることを確認した。

## 5. `TaskArgs`（想定フィールド）

`RoadCollectionArgs`（密度・SDFの計算に使う。`RoadTask` 経由で `patch_args` される）:

| パラメータ | 意味 | 備考 |
|---|---|---|
| `sigma0` | 最小到達半径（street相当） | 密度合成カーネルの基準スケール |
| `k_sigma` | 幅→到達範囲の係数 | highway ほど広域に効くようにする |
| `reach_exponent` | 到達半径の幅に対する指数 | 1より大きいほど幹線道路の到達範囲だけ不釣り合いに拡大する |
| `amp_scale` | ピーク強度の基準スケール（`amplitude_e = amp_scale/sigma_e`） | 到達範囲が広いほどピークが下がる |
| `min_density_floor` | 人口密度の絶対的な最低ライン | `compute_density` で `max` によって無条件保証（3.3節）。道路が無くても保証される値なので、通常は `target_outside_value` より低く設定する |

`RoadArgs`（Task側）:

| パラメータ | 意味 | 備考 |
|---|---|---|
| `width_classes` | 選択可能な道路幅の離散値一覧 | 例: `(0.02, 0.05)`。Add系書き換えで各値を提案 |
| `target_inside_value` / `target_outside_value` | `target_img` を実際の密度値へアフィン変換する際の範囲 | 3.0節。デフォルト `0.7`/`0.15`（pngs・shc共通） |
| `cost_weight` | 建設コスト正則化の重み | `Tri.node_weight` に相当 |
| `cost_width_exponent` | 建設コストの幅に対する指数 | 1より大きいほど幹線道路への罰則が強くなる |
| `mesh_weight` | ループ形成(meshedness)への報酬の重み | 4.1節。大きいほどSnapによるループ化を優先する |
| `min_angle` / `angle_penalty_exponent` / `angle_weight` | 交差点の角度が小さすぎることへの罰則 | 4.2節。`min_angle` はラジアン（デフォルト `math.radians(45)`）。`_compute_losses` に直接加算 |
| `snap_radius` | スナップ候補とみなす最大距離（グリッドのセルサイズにも使う） | |
| `default_length` / `length_range` | Add系書き換えの新規エッジ長 | |
| `add_weight` / `add_free_weight` | `AddFromNode`/`AddFree` の候補数に比例した重み | `n_candidates * weight` 個の候補を生成する。値を下げると相対的にその書き換えが選ばれにくくなる |

## 6. 可視化

`Tri`/`UR` の `visualize` 実装（`tasks/tri.py`）と同じ3層構成に倣う。1枚の `MPLVisualizerAxes` 上に:

1. **背景**: `self.target_img`（target 人口密度）を `imshow(..., cmap="plasma", alpha=0.2)` で薄く表示
2. **重ね書き**: `collection.compute_density(...)` で計算した予測密度を `imshow(..., cmap="magma", alpha=0.5〜0.6)` で表示（roadsから合成された密度そのものが主役なのでtargetより濃いめにする）
3. **前景**: 道路網を線分として描画。幅クラスごとに線の太さを変える。細い道路(street)が背景ヒートマップに埋もれて見えなくなる問題があったため、白の本体の下に一回り太い黒の「ケーシング」を先に描く（地図表現でよく使われる技法。`RoadNetwork.visualize` に実装済み）

```python
def visualize(self, collection: ObjectCollection[RoadNetwork], step: int, loss: float, state: None) -> np.ndarray:
    assert isinstance(collection, RoadNetworkCollection)
    net = collection[0]
    fig = MPLVisualizer(1, 1, 10.8, 10.8, xlim=self.render_args.lim, ylim=self.render_args.lim, notebook=False)
    ax = fig[0]
    extent = (self.render_args.lim[0], self.render_args.lim[1], self.render_args.lim[1], self.render_args.lim[0])
    ax.ax.imshow(self.target_img.detach().cpu().numpy(), extent=extent, cmap="plasma", vmin=0, vmax=1, alpha=0.2)
    density = collection.compute_density(grid_positions, ...)[0]  # (size, size)
    ax.ax.imshow(density.detach().cpu().numpy(), extent=extent, cmap="magma", vmin=0, vmax=1, alpha=0.5)
    for (i, j), w in zip(net.edges, net.widths.tolist()):
        (x1, y1), (x2, y2) = net.nodes[i].tolist(), net.nodes[j].tolist()
        ax.ax.plot([x1, x2], [y1, y2], color="white", linewidth=width_to_pt(w), solid_capstyle="round")
    ax.ax.set_title(f"{self.get_elapsed_time():.0f}s: {net.id}: {loss:.2e}: E{len(net.edges)}")
    return fig.get_image()
```

## 7. 実装マイルストーン

1. `RoadNetwork` / `RoadNetworkCollection`：SDF (`rasterize`) と `visualize`（密度なしでまず道路のみ）を実装し、手動で作ったネットワークが正しく描画されるか確認
2. 書き換え：`AddFromNode` / `RemoveEdge` のみ（ループなし木構造、幅クラスは固定1種類）で `make_proposals` / `combine_proposals` の動作確認
3. カスタム損失（`compute_density` + baseline + MSE）と `visualize` への密度描画追加を実装し、簡単な合成 target マップに対して収束するか確認
4. `Snap` 書き換え（一様グリッド探索）を追加してループ形成を確認
5. 幅クラスを複数に増やし、Add提案の複製ロジックを確認
6. `cost_weight` 等のハイパーパラメータ調整、実データ（人口密度マップ）でのテスト

## 9. 文法の根本再設計：4つの性質（Reversibility / Jump Continuity / Local Geometric Control / Repairability）に基づく（2026-07-04）

### 9.0 動機

実データでの学習結果を見ると、(a) 道路網が複数の非連結な部分に分かれる、(b) 幹線道路が街路に完全に囲まれて孤立している（幹線道路網としての連続性がない）、という2つの「都市として不自然」な問題が見えた。これらは損失関数の重み付け（meshedness報酬・角度罰則など）をいくら足しても**確率的にしか抑制できず、構造として禁止できない**。論文 (`design-for-descent.pdf`) が提示する4つの性質（Table 1）に立ち返り、**連結性・幹線道路の階層性を"損失"ではなく"文法の構造的な不変条件"として保証する**方向に文法自体を再設計する。

論文の4性質（Table 1 の定義そのまま）:

| 性質 | 定義 | 論文中の例 |
|---|---|---|
| **Reversibility** | 書き換え `A→B` があるなら、逆方向 `B→A` も存在すること | Split/Merge、Add/Remove-Loop |
| **Jump Continuity** | 書き換えの適用による形状の瞬間的な変化が無視できるほど小さいこと | 局所的なセグメント分割 |
| **Local Geometric Control** | 形状のどこにでも、遠方に影響を与えずに局所的な変更を加える書き換えが存在すること | Add-Anywhere |
| **Repairability** | 制約が存在するなら、それに違反した形状を修復する書き換えが存在すること | Resolve-Intersections |

Tree文法（`objects/tree.py`）は上記をすでに満たす実例になっている: `SplitBranch`↔（分割自体は`Merge`相当の操作は無いが可逆な`RemoveBranch`と対）、`AddBranch`（epsilon長で追加=Jump Continuity）↔`RemoveBranch`、`AddAnywhere`（**遠方の任意の点へ、既存ノードから新規ノードの鎖で"必ず接続したまま"到達する** = Local Geometric Control）、`cleanup()`（交差する枝を検出し`SplitBranch`で再接続する = Repairability、制約は「枝同士が交差しない」）。今回の道路網再設計は、この Tree の設計パターンをグラフ表現に翻訳する。

### 9.1 構造的な不変条件（損失ではなく書き換えの前提条件として保証する）

現行文法の `AddFree`（空間中に孤立した新規道路を置く）が非連結性の直接の原因。これを**廃止**し、代わりに次の2つの不変条件を、全ての書き換えの生成条件（`gen_rewrite_specs` の候補列挙時）でチェックすることによって常に保証する:

- **不変条件A（連結性）**: ネットワークは常に単一の連結成分である。初期化時は単一エッジのみなので自明に満たされる。以降、ノード・エッジを追加する書き換えは必ず既存ノードから辿れる形でのみ行う（`AddFree` 相当の操作を削除し、後述 `AddAnywhere` に置き換える）。ノード・エッジを削除する書き換えは、削除後も連結性が保たれる場合のみ候補に含める。
- **不変条件B（幹線道路の階層性）**: `width_classes` の中で最も太いクラス（highway）が張るサブグラフ（highwayエッジのみを辺とする部分グラフ）は常に単一連結成分であり、かつ初期化時のルートノードを含む。つまり **highwayエッジは、既にhighwayエッジに触れているノード（またはルート）からしか新規に生えない**。street は highway ノード・street ノードのどちらからでも自由に生える（現実の都市同様、幹線道路の途中から街路が分岐するのは自然）。この条件により「幹線道路が街路network経由でしか本体に繋がっていない」という孤立が構造的に発生しなくなる。

これらは `compute_simplicity`/`_compute_losses` の罰則ではなく、**そもそもそのような書き換え提案を生成しない**という形で保証する（生成されなければ`combine_proposals`が選びようがない）。

### 9.2 書き換えセットの再設計

| 書き換え | 内容 | 逆操作 | 満たす性質 |
|---|---|---|---|
| `Add(node_id, width_class)` | 既存ノードから epsilon 長の新規エッジを伸ばす（末端ノード新設）。`width_class=highway` の場合は `node_id` が不変条件Bの意味でhighway適格（隣接エッジに1本以上highwayがある、またはルート）であることが必須 | `Remove` | Jump Continuity（epsilon長）／Reversibility |
| `Remove(edge_id)` | 次数1（末端）のエッジを削除。末端なので常に安全（連結性を壊さない） | `Add` | Reversibility |
| `AddAnywhere(target_point, width_class)` | 空間中の任意の点へ向け、**最も近い適格な既存ノードから新規ノードの鎖（1本以上）で接続したまま**到達する（`Tree.AddAnywhere`と同型のアルゴリズム）。`width_class=highway`の場合は起点ノードが不変条件Bのhighway適格ノードに制限される | 鎖の末端から`Remove`を繰り返す | **Local Geometric Control**（遠方への局所変更）／Reversibility |
| `Split(edge_id, t)` | エッジをパラメータ `t∈(0,1)` の位置で分割し、同じ`width_class`の2本の新エッジ＋次数2の新規ノードにする。位置は分割前と完全に一致するため形状は変化しない | `Merge` | **Jump Continuity**（論文の例そのもの：局所的なセグメント分割）／Reversibility |
| `Merge(node_id)` | 次数2のノードで、両側のエッジの `width_class` が同じかつほぼ共線（角度ずれ`< eps`、`ArcLines`の`ToLine`/`Merge`と同じ厳密性のゲート）の場合にノードを消して1本のエッジに統合 | `Split` | Reversibility（`eps`ゲートによりJump Continuityも保つ） |
| `Snap(edge_id, target_node_id)` | ぶら下がった端点を既存ノードに張り替えてループ/交差点を作る。`width_class=highway`のエッジは、張り替え先もhighway適格ノードに制限（不変条件B維持） | `Unsnap` | Reversibility（論文の Add/Remove-Loop 例） |
| `Unsnap(edge_id)` | サイクル上のエッジ（=削除しても連結性を壊さないエッジ）を選び、片方の端点を**同じ座標に新規複製したノード**に付け替え、ぶら下がった端点に戻す（形状は瞬間的に不変）。highwayエッジの場合は「highwayサブグラフだけを見ても連結性が壊れない」ことも追加で確認（不変条件B維持） | `Snap` | Reversibility／Jump Continuity（同座標に複製するため見た目は変わらない） |
| `Widen(edge_id)` | street→highway への格上げ。両端点それぞれについて「このエッジを除いた残りの隣接エッジが全てhighway、または次数1（=このエッジのみ）」を満たす場合のみ許可（格上げが不変条件Bを壊さない場合のみ） | `Narrow` | Reversibility |
| `Narrow(edge_id)` | highway→street への格下げ。street は制約が無いので常に許可 | `Widen` | Reversibility |

**削除する操作**: `AddFree`（不変条件Aに反するため廃止。役割は `AddAnywhere` が代替）。

### 9.3 Repairability：交差の解消

不変条件A・Bは書き換えの生成条件で保証されるため、通常はこれらに関する「修復」は不要になる（違反する形状がそもそも生成されない）。一方、**勾配降下によるノード位置の連続的な移動**は書き換えとは独立に起こるため、位置更新の結果、共有ノードを持たない2本のエッジが幾何的に交差してしまう（本来は交差点＝共有ノードであるべき）ケースが起こりうる。これは道路網特有の制約（「交差する道路は必ずノードを共有する」）であり、論文の Repairability の直接的な適用対象になる。

`Tree.cleanup()` が行っている「交差するエッジを検出し `SplitBranch` で分割・再接続する」アルゴリズムをグラフ表現に翻訳し、`RoadNetwork` にも `cleanup()`（または `resolve_crossings()`）として実装する。具体的には、交差する2エッジそれぞれを交点で `Split` し、生成された2つの新規ノードを1つに統合する（`Merge`とは異なり幅クラスが異なっていても統合可能な特別処理、または単純に一方のノードへ他方のエッジを張り替える）。`optimizer.py` が定期的に呼ぶ `Task.cleanup()` から呼び出す（既存の `prune_orphan_nodes()` 呼び出しと同じ場所）。

### 9.4 現行実装からの変更点まとめ

- `RoadRewriteType` から `AddFree` を削除し、`AddAnywhere`（Tree型の鎖接続）に置き換える
- `RoadRewriteType` に `Split` / `Merge` / `Unsnap` / `Widen` / `Narrow` を追加
- 各書き換えの候補生成（`gen_rewrite_specs`）に、不変条件A（連結性）・不変条件B（highway階層性）のチェックを追加する。特に「ノードがhighway適格か」の判定（隣接エッジ集合から計算）はほぼ全ての書き換えで共通して必要になるため、共通ヘルパー（例: `RoadNetwork.is_highway_eligible(node_id)`）として実装する
- `Unsnap` の「削除しても（highwayサブグラフに限定しても）連結性を壊さないエッジか」の判定は、ネットワーク全体では稀にしか呼ばれない想定なので、候補ごとに小規模なBFS（対象ノードから、削除対象エッジを除いたグラフで到達可能か）で十分。全体を都度再計算するのではなく、Snap候補と同様に「ローカルな判定」として実装する
- `RoadNetwork.cleanup()`（交差解消、9.3節）を新設し、`RoadTask.cleanup()` から `prune_orphan_nodes()` と並べて呼び出す
- meshedness報酬（4.1節）・角度罰則（4.2節）・建設コスト（4節）は引き続き損失/simplicityとして残す——これらは「あった方が良い」性質（都市らしい見た目のスコア）であり、連結性・階層性のような「あってはならない違反」とは性質が異なるため、両者は併用する

### 9.5 未確定・実装時に判断が必要な点

- `AddAnywhere` のデフォルト `width_class` は street とする想定（Parish & Müller の「streetがhighway間を埋める」という役割分担に合わせる）。highway の `AddAnywhere` も許可するかは、実験して都市らしさを見ながら判断する
- `Unsnap` のBFSコストが大きい場合は、`get_meshedness()` と同様に「近似で十分」という割り切り（例えば次数3以上のノードに接続するエッジのみを候補にする、など）も検討する
- `Widen`/`Narrow` は今回のユーザー要望（連結性・階層性）に必須ではないが、Reversibilityの観点で「幅クラスを変える書き換えが存在するなら逆方向も必要」という論文の原則に従うために追加を提案している。優先度を下げて後回しにする選択肢もある

### 9.6 実装完了（2026-07-04）

9節の設計に基づき `objects/roads.py`/`tasks/roads.py` を全面的に書き直した（既存ファイルの新規置き換え。他の既存コードは変更なし）。目的関数（3〜4節）は変更せず、書き換え（rewrite）の生成条件・適用ロジックのみを再設計した。

**実装した書き換え**: `Add`/`Remove`/`AddAnywhere`/`Split`/`Merge`/`Snap`/`Unsnap`/`Widen`/`Narrow`（9.2節の表のとおり）。`AddFree` は完全に削除。

**テストで発見・修正した設計上のバグ**（いずれも「単体では安全な書き換えが、同じ最適化ステップ内で複数同時に採択されると組み合わせで不変条件を破る」というクラスの問題。`RoadTask.combine_proposals` は複数の改善提案を1ステップでまとめて採択する（`accept_parallel`）ため、この検証が必須だった）：

1. **Narrow がhighwayサブグラフのブリッジ辺を切ってしまう**: 最後の1本かどうかだけをチェックしていたが、highwayが木構造（枝分かれ）になっている場合、中間の辺を格下げするとhighway網が2つに分断される。`_is_reachable_without_edge`（highway限定BFS）によるブリッジ判定を追加して修正
2. **`apply_all_rewrites` 内での「孤立化 vs 接続」の競合**: `Merge`/`Remove`/`Snap`はいずれも対象ノードの一方を孤立させる（または孤立していたノードを消費する）操作だが、同じバッチ内で別の`Add`/`AddAnywhere`/`Snap`がその"孤立する側"に新しい枝を接続していると、その枝ごと本体から切り離されてしまう。`node_status`（ノードごとの"orphaned"/"attached"状態）辞書を追加し、孤立操作と接続操作が同じノードで競合したら後勝ちを拒否するよう修正
3. **同一バッチ内の複数`Unsnap`が同じループを共有する場合**: 1本目のUnsnapでループが開いた後は残りの辺がブリッジになるため、2本目以降のUnsnapは無効化する必要がある。`apply_all_rewrites`内で`live_edges`という軽量な辞書ベースのグラフ状態を保持し、Unsnap/Narrowの安全性チェックを「バッチ開始時点」ではなく「その時点までに確定した変更を反映した最新状態」に対して行うことで解決

**検証方法**: 通常の単体テスト（不変条件の生成条件チェック、バッチ版`make_proposals_ex`と単体版`apply_rewrite`のクロスチェック）に加えて、`apply_all_rewrites`にランダムな提案の部分集合（最大25件）を繰り返し適用し、毎ステップ後に「全体の連結性」「highwayサブグラフの連結性・非空性」を検査するfuzzテストを実装（スクラッチパッドのみ、リポジトリには含めていない）。30種のランダムシードで400ステップ、7種で1200ステップ（ネットワークが350〜420ノードまで成長）、いずれも違反なしを確認。実際の`scripts/optimize_pngs.py`経由のCLI実行（`_rungen_roads_pngs.py`が生成するコマンド）でも動作確認済み。

## 9.7 spreading（道路が広がらず中央に密集する）バグの修正（2026-07-05）

実データ（`Road_Donut`）で「道路が中央のみに密集し、ターゲット領域の外周に広がらない」現象が発生。Donutターゲットで最適化を走らせ損失を分解して切り分けた結果：

- **`compute_density` にバグは無い**（街路1本のプロファイルを数値確認：ピーク0.957、sigma=0.047で0.329、それ以遠はfloor 0.05。設計通り）
- 176エッジ追加しても密度MSEが0.084→0.063としか下がらず、ノードはstd≈0.16で中央集中。原因は **AddAnywhere（連結を保つ唯一のspreading手段）が機能不全** だったこと。3つの要因が重なっていた：
  1. **提案予算の枯渇**: `gen_rewrite_specs`が生成する候補は`Add`（局所成長）が支配的（`add_weight=3.0`で約190件）で、`make_proposals_ex`が`random.sample`で一様に64件へ絞ると、少数の`AddAnywhere`（weight 0.15で約2件）はほぼ毎回サンプルから漏れていた
  2. **狙う点の8割が背景**: AddAnywhereの目標点を`lim`全体から一様サンプルしていたため、Donutでは約81%がターゲット領域外（背景）を狙い、そこへ橋を架けても密度MSEを悪化させるだけで却下されていた
  3. **長大な橋**: 遠方点へは最大14本もの鎖エッジを1提案で架け、その大半が背景を貫くためやはり却下されやすかった

**修正（3点、いずれも新規コード内で完結、損失関数は不変）**：
1. **層化サンプリング** (`RoadTask._stratified_sample`): 提案をタイプ別にグループ化し、`num_proposals`の予算をラウンドロビンで公平に配分。候補数の多いタイプが予算を独占せず、AddAnywhere等の少数タイプも必ず評価される
2. **ターゲット偏重の目標点** (`RoadDensityTask._precompute_add_anywhere_targets` → `gen_rewrite_specs(add_anywhere_targets=...)`): AddAnywhereの目標点を、元のtarget_imgに比例した確率で事前サンプルした点プールから選ぶ。Donutでターゲット領域内の点が19%→99%に
3. **鎖の長さ上限** (`RoadRewriteArgs.max_add_anywhere_hops=6`): 1提案での長大な橋を防ぎ、連結を保ったまま徐々に伸ばす

**結果（Donut, 200ステップ, size128での比較）**: 密度MSE 0.069→0.035（約半減）、ノードのy方向到達 0.835→0.991（ターゲット上端y=0.98に到達）、ターゲット被覆率 31%→55%。`add_anywhere_weight`のデフォルトと実行スクリプトを0.15→1.0に、`n_add_anywhere_candidates`を8→32に引き上げた。

**教訓**: 「複数の書き換えタイプを一様サンプリングで評価対象に選ぶ」設計は、候補生成数が偏ると少数タイプが構造的に飢餓状態になる。特にspreadingのような「稀だが決定的に重要」な操作は、層化サンプリングで代表を保証すべき。また探索的な操作（どこへ道路を伸ばすか）は、探索先を目的関数の有効領域（人口密度の高い場所）へ偏らせると劇的に効率が上がる。

**未対応の関連課題（今回のバグとは別、性能面）**: `compute_density`はO(エッジ数×size²)で、size=256・エッジ数が数百〜千規模になると連続最適化・提案評価が重い（CPUで1バッチ数十秒）。`RoadNetwork.cleanup`（交差解消）もO(E²)の純Pythonループ。spreadingが効いてエッジ数が増えるほど顕在化するため、必要なら空間ハッシュ化等の高速化を別途検討する。

## 9.8 「最小限の道路で被覆する」ための連続建設コスト（size_weight）（2026-07-05）

spreading修正後も「中央に幹線道路が団子状に溜まる」現象が残った。原因は**建設コストが`compute_simplicity`（離散書き換えの採否のみ）にしか入っておらず、`_compute_losses`（連続勾配最適化）には全く入っていなかった**こと。そのため連続最適化フェーズで道路を短く・少なくする力が皆無で、一度できた冗長な道路には縮む動機がない。特に：

- 幹線道路(highway)は amplitude≈0.23 で、ターゲット内側(0.7)を満たすのは街路(amplitude≈0.96)。よって**街路が敷かれた領域内の幹線道路は amax で常に街路に負け、密度的に完全に冗長**。にもかかわらず縮む力も削除経路(内部エッジはRemove不可)も無いため、角度罰則・密度勾配で動き回って団子として堆積する。

**修正**: `RoadArgs.size_weight` を追加し、`_compute_losses` に `size_weight * Σ(length_e * width_e^cost_width_exponent)` を加算（Triの`size_weight`と同型。`compute_simplicity`の離散`cost_weight`とは別物で、**連続の勾配としてノード位置に「道路を短く保つ」力を与える**）。`width^cost_width_exponent`(=2.5)重みなので幹線道路(幅広)ほど強く縮み、追加提案も却下される。冗長な道路は密度カバレッジのアンカーが無いのでこの力で縮み、実際に有用な道路（＝縮めると被覆が悪化するもの）だけが残る＝「最小限の道路で被覆」。

**検証（Donut, size128, 200step, size_weightを掃引）**:
| size_weight | edges | highways | coverage | 中央の幹線道路 |
|---|---|---|---|---|
| 0（旧） | 252 | 161 | 49% | 33 |
| **20（採用）** | **122** | **13** | **56%** | **5** |
| 60 | 1 | 1 | 0% | 崩壊 |

`size_weight=20`で幹線道路 161→13本・総エッジ 252→122・中央の幹線道路 33→5 と大幅に疎になりつつ被覆率は 49→56% に向上。狙い通り「amaxで冗長な中央の幹線道路が消え、連結バックボーンの幹線道路だけが残る」創発が起きた。大きすぎる(60)と道路網ごと縮んで消えるので20前後が最適。デフォルトと実行スクリプトに`size_weight=20.0`、`cost_width_exponent`デフォルトも2.5に統一（コスト式のスケールを揃えるため）。

**教訓**: 正則化を`compute_simplicity`（離散採否）だけに置くと、連続最適化フェーズでは全く効かず「一度できた冗長な構造」を縮められない。位置・形状を動かして構造を簡素化させたい正則化は`_compute_losses`（微分可能な連続損失）に入れる必要がある（[[project-city-grammar]] 角度罰則と同じ判断、Triのnode_weight(離散) vs size_weight(連続)の役割分担そのもの）。

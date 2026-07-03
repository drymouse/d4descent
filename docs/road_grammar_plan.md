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

`RasterLossMixin`（二値占有率のMSE）はそのまま使わず、道路網から人口密度場を合成するforward modelを自作する。ただし **target の入力形式は raster タスク（`_pngs`）と同じ `[0,1]` の画像（`target_img`）をそのまま人口密度マップとして扱う**（専用の正規化パイプラインは用意しない）。

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

### 3.3 最低ライン（当初案から変更）

当初は `density_pred = baseline + (1 - baseline) * raw(x)` として forward model に無条件で加算していたが、これだと `density_pred` が数式上 `baseline` を下回ることが絶対にできず、「最低ラインを下回った場合のペナルティ」を追加しようにも常にゼロになってしまう（実際に運用してみて気づいた設計ミス）。

そこで **`compute_density` は `raw(x)` をそのまま返す**（無条件の下駄を廃止）。最低ラインは損失側の非対称な罰則として実装する:

```
mse(x)       = (density(x) - target(x))^2
underflow(x) = relu(min_density_floor - density(x))^2   # 床を下回った分だけ二乗で罰する。targetの値によらない
loss(x)      = mse(x) + underflow_weight * underflow(x)
```

- `min_density_floor` / `underflow_weight` は `RoadArgs`（Task側）のフィールド。`RoadCollectionArgs` からは `baseline_density` を削除した
- こうすることで「最低ラインを上げる」(`min_density_floor` を上げる)と「下回った時のペナルティを強める」(`underflow_weight` を上げる)を独立に制御できる

### 3.4 損失

```python
def _compute_losses(self, collection, state):
    density = collection.compute_density(size, lim, center_pixel)  # (n_networks, size, size)
    mse = (density - self.target_img).square()
    underflow = (self.args.min_density_floor - density).clamp(min=0.0).square()
    loss = (mse + self.args.underflow_weight * underflow).flatten(-2).mean(dim=-1)
    return loss, {}
```

`compute_density` は `rasterize`（`amin`集約、可視化・境界判定用）とは別のメソッドとして `RoadNetworkCollection` に実装する（集約方法が `amin` ではなく「和→飽和」のため）。ただしエッジごとのカプセルSDF計算自体は両者で共通化できる。`self.target_img` は既存の raster タスクと同じ仕組み（`RasterLossArgs`/`target_img` 経由）でロードした `[0,1]` 画像をそのまま使う。**`img_mode` は `bow`（黒=前景を高密度として反転）を使うこと** — `wob` のままだと画像の余白（背景）が高密度、ロゴ/市街地形状が低密度に読み込まれ、意図と正反対になる。

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

## 5. `TaskArgs`（想定フィールド）

| パラメータ | 意味 | 備考 |
|---|---|---|
| `width_classes` | 選択可能な道路幅の離散値一覧 | 例: `(0.02, 0.05)`。Add系書き換えで各値を提案 |
| `sigma0` | 最小到達半径（street相当） | 密度合成カーネルの基準スケール |
| `k_sigma` | 幅→到達範囲の係数 | highway ほど広域に効くようにする |
| `reach_exponent` | 到達半径の幅に対する指数 | 1より大きいほど幹線道路の到達範囲だけ不釣り合いに拡大する |
| `amp_scale` | ピーク強度の基準スケール（`amplitude_e = amp_scale/sigma_e`） | 到達範囲が広いほどピークが下がる |
| `min_density_floor` | 人口密度の最低ライン | 損失側の非対称罰則（3.3節）で使う。forward modelには焼き込まない |
| `underflow_weight` | 最低ラインを下回った分への追加罰則の重み | 大きいほど床割れを強く嫌う |
| `cost_weight` | 建設コスト正則化の重み | `Tri.node_weight` に相当 |
| `cost_width_exponent` | 建設コストの幅に対する指数 | 1より大きいほど幹線道路への罰則が強くなる |
| `mesh_weight` | ループ形成(meshedness)への報酬の重み | 4.1節。大きいほどSnapによるループ化を優先する |
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

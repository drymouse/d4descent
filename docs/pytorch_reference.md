# このプロジェクトで使われる PyTorch リファレンス

## 1. Tensor の生成

```python
torch.tensor([1.0, 2.0, 3.0])      # Pythonリストから作る
torch.zeros((3, 2))                 # ゼロで埋めた (3,2) のTensor
torch.full((n,), torch.inf)         # 定数で埋める
torch.rand((n, 2))                  # [0,1) の乱数
torch.arange(n)                     # 0,1,...,n-1 の整数列
torch.as_tensor(x, dtype=...)       # NumPy配列など他の型から変換
torch.from_numpy(arr)               # NumPy配列から変換
```

---

## 2. 形状の変換

```python
x.shape          # (3, 2) のようなサイズ情報（属性）
x.ndim           # 次元数

x.unsqueeze(0)   # 次元を追加: (3,) → (1, 3)
x.squeeze(-1)    # サイズ1の次元を削除: (3, 1) → (3,)
x.reshape(n, 2)  # 形を変える（要素数は同じ）
x.flatten(-2)    # 末尾2次元を1つにまとめる
x.permute(2, 0, 1)  # 次元の順番を入れ替える
x.expand(n, -1)  # メモリコピーなしでサイズを広げる（-1は変更なし）
x.unbind(dim=-1) # 指定次元で分解してtupleに
```

---

## 3. Tensor の結合

```python
torch.cat([a, b], dim=0)    # 既存の次元方向に連結
#   (3,) と (2,) → (5,)

torch.stack([a, b], dim=-1) # 新しい次元を作って積み重ねる
#   (3,) と (3,) → (3, 2)
```

---

## 4. 数学演算

すべて**要素ごと**に計算され、微分可能です。

```python
# 四則演算（通常の演算子がそのまま使える）
a + b,  a - b,  a * b,  a / b
a ** 2                             # べき乗

# 三角関数
torch.sin(x),  torch.cos(x)
x.sin(),       x.cos()             # メソッド形式でも同じ

# 逆三角関数
torch.atan2(y, x)                  # arctan(y/x)、四象限対応

# べき乗・平方根
x.square()                         # x²
x.sqrt()                           # √x

# 絶対値
x.abs()

# ベクトルのノルム（長さ）
x.norm(dim=-1)                     # 最後の次元方向のL2ノルム
#   (n, 2) → (n,)  ← 2D点群の各点の原点からの距離

# 内積
torch.dot(a, b)                    # 1Dベクトル同士
(a * b).sum(dim=-1)                # バッチ内積（高次元対応）
```

---

## 5. 集約演算

```python
x.sum(dim=-1)      # 指定次元で合計: (n, 3) → (n,)
x.mean(dim=-1)     # 平均
x.min(dim=-1)      # 最小値。.values と .indices が返る
x.max(dim=-1)      # 最大値
x.prod(dim=-1)     # 積
x.any(dim=-1)      # どれかTrueか (bool Tensor用)
x.all(dim=-1)      # すべてTrueか

torch.maximum(a, b)  # 要素ごとの最大値（2Tensorの比較）
torch.minimum(a, b)
```

---

## 6. クランプ（値の範囲制限）

```python
x.clamp(min=0.0)           # 0未満を0にする
x.clamp(min=0.0, max=1.0)  # [0, 1] に収める
x.clamp_(min=0.0)          # 末尾 _ はin-place（xを直接書き換える）
```

SDF の計算でよく使います：

```python
d.clamp(min=0).norm(dim=-1) + d.max(dim=-1).values.clamp(max=0)
# ↑ 矩形・三角形のSDFの核心部分
```

---

## 7. 条件分岐

```python
torch.where(condition, a, b)
# conditionがTrueの要素はaを、FalseはbをとるTensorを返す
# if文の代わり（Tensor全体に一括適用できる）
```

---

## 8. 自動微分

```python
x = x.detach()              # 計算グラフから切り離す
x = x.clone()               # メモリごとコピー（グラフは引き継ぐ）
x = x.detach().clone()      # 切り離してコピー（よく一緒に使う）

x.requires_grad_(True)      # 「この変数を微分する」と宣言

loss.backward()             # 勾配を逆伝播
x.grad                      # backward() 後に勾配が入る

# 勾配計算が不要なブロック（可視化・推論時に使う）
with torch.no_grad():
    ...
```

---

## 9. デバイス・型

```python
x.device    # 'cpu' か 'cuda:0' など（属性）
x.dtype     # torch.float32 など（属性）

x.to(device="cuda")          # GPU に転送
x.to(dtype=torch.float64)    # 型を変換

torch.long    # 64bit整数型（インデックス用）
torch.bool    # bool型
torch.float32 # 32bit浮動小数点（デフォルト）
```

---

## 10. Python との変換

```python
x.item()     # 要素が1個のTensorをPythonのfloat/intに変換
x.tolist()   # TensorをPythonのリストに変換
```

---

## TriangleUnion の SDF で使うもの

三角形のSDF実装では主にこれだけ使えば書けます。

```python
# 各辺への射影と距離
ba = b - a
t = (pa * ba).sum(dim=-1) / ba.square().sum(dim=-1)
t = t.clamp(0, 1)
dist = (pa - ba * t.unsqueeze(-1)).norm(dim=-1)

# 内外判定（外積の符号）
cross = ba[..., 0] * pa[..., 1] - ba[..., 1] * pa[..., 0]
sdf = torch.where(cross >= 0, dist, -dist)
```

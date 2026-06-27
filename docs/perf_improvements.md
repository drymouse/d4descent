# Tri 文法 パフォーマンス改善ログ

## 背景

Rewrite ステップが遅い問題を解消するため、3つのボトルネックを修正した。

---

## 修正1: `make_proposals_ex` のバッチ処理（最大の改善）

**ファイル**: `src/d4descent/tasks/tri.py`

**問題**: `proposal_size=64` 個の提案を生成するとき、`apply_rewrite` を 64 回ループ呼び出しし、そのたびに `torch.cat` で三角形テンソルをコピーしていた。

```python
# 旧: 64回のループ → 64回の torch.cat
for spec in specs:
    rewritten.append(obj.apply_rewrite(spec, self.args.tri_args))
return self._Collection.from_objects(rewritten), specs
```

**修正**: 型ごとにまとめて1回のテンソル演算で全提案を生成するよう書き換え。

- **AddTri**: `torch.tensor` で全頂点を一括生成 → `expand` + `cat` で `(n_add, n_base+1, 3, 2)` を作り `reshape` で平坦化。CPU→GPU 転送が N 回 → 1 回に削減。
- **RemoveTri**: `torch.gather` で各提案の「削除後三角形リスト」を一括取得。`(n_remove, n_base-1, 3, 2)` を1回の gather で生成。

どちらも `TriCollection` を直接構築し、`from_objects` / `apply_rewrite` のループを完全に除去した。

**効果**: `torch.cat` の呼び出し回数が `proposal_size` 回 → 数回（型数分）に削減。

---

## 修正2: `apply_all_rewrites` の削除処理をマスク化

**ファイル**: `src/d4descent/objects/tri.py`

**問題**: `deleted` セット内のインデックスに対してループで `torch.cat` を呼び出し、削除1件ごとに新しいテンソルを作っていた。

```python
# 旧: n_deleted 回の torch.cat
for i in sorted(deleted, reverse=True):
    new_xs = torch.cat([new_xs[:i], new_xs[i + 1:]])
```

**修正**: ブールマスクで一括削除。

```python
# 新: テンソルの boolean indexing で1回で完了
if deleted:
    keep = torch.ones(len(new_xs), dtype=torch.bool, device=new_xs.device)
    for i in deleted:
        keep[i] = False
    new_xs = new_xs[keep]
```

**効果**: `n_deleted` 回の `torch.cat` → 1 回の boolean indexing に削減。

---

## 改善されなかった点（意図的に残した箇所）

- `gen_rewrite_specs` での `.to("cpu").detach().numpy()`: 修正1により GPU→CPU 転送が 1 回になったため、ここを変えてもほぼ効果なし。
- `apply_rewrite`: 修正1により `make_proposals_ex` から直接呼ばれなくなった（`combine_proposals` からは引き続き使用、こちらは呼び出し回数が少ないため問題ない）。

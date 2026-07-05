"""
City文法のアブレーション実験。design-for-descent.pdf の Table 2 と同じ形式で、
書き換え規則(rewrite rules)を段階的に有効化しながら最適化品質(PSNR)と単純さ(#プリミティブ=辺数)
がどう変化するかを比較する。

各バリアントは「前段のバリアント + 1つの機能」という累積的な構成にしてあり(論文のTr-1..Tr-Fと同じ
比較の仕方)、rewrite_args 以外(cost_weight, decimate, cleanup等)は全バリアントで固定する。
データセットは data/arclines/ 以下の bench128(One) / donut25(Dnt) / twocomp23(Two) を使う。
これは論文のOneComp/Donut/TwoCompデータセットに相当するものとして本リポジトリに既に存在する。

使い方:
    uv run python scripts/ablation_city.py
    uv run python scripts/ablation_city.py --n_steps 2000 --n_shapes 3 --device cuda
    uv run python scripts/ablation_city.py --variants C-1,C-F --datasets One
"""

import argparse
import csv
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import torch
from rich.console import Console
from rich.table import Table

from d4descent.objects.arclines import ShapeCollection
from d4descent.objects.city import CityCollectionArgs, CityNetworkCollection, CityRewriteArgs
from d4descent.losses.raster import RasterLossArgs
from d4descent.optimizer import OptimizeArgs, optimize
from d4descent.tasks._base import RenderArgs
from d4descent.tasks.city import CityArgs
from d4descent.util import save_rgb8

PROJ_DIR = Path(__file__).parent.parent

DATASETS: dict[str, str] = {
    "One": "data/arclines/bench128.shc",
    "Dnt": "data/arclines/donut25.shc",
    "Two": "data/arclines/twocomp23.shc",
}


@dataclass
class GrammarVariant:
    name: str
    description: str
    rewrite_args: CityRewriteArgs


def _rw(**enabled: float) -> CityRewriteArgs:
    """全書き換えタイプの重みを0にした上で、enabledで指定したものだけ有効にする。"""
    weights = dict(
        add_weight=0.0,
        add_anywhere_weight=0.0,
        remove_weight=0.0,
        split_weight=0.0,
        merge_weight=0.0,
        snap_weight=0.0,
        unsnap_weight=0.0,
    )
    weights.update(enabled)
    return CityRewriteArgs(
        length_range=(0.05, 0.15),
        snap_radius=0.05,
        n_add_candidates=32,
        n_add_anywhere_candidates=32,
        n_split_candidates=16,
        **weights,
    )


def make_variants() -> list[GrammarVariant]:
    """
    累積的な5段階のバリアント(Tr-1..Tr-Fに相当)。
    - C-1: Add のみ。Reversibility無し(伸ばすだけで元に戻せない)。
    - C-2: + Remove。Add<->RemoveでReversibilityを満たす(論文Table1のAdd/Remove例)。
    - C-3: + Split/Merge。形状を変えずに再分割・再結合できる(Reversibility兼Jump Continuityの例)。
    - C-4: + Snap/Unsnap。端点同士を繋いでループを作れる(Snap<->UnsnapでReversibility、
      ループ形成による街区構造)。
    - C-F: + AddAnywhere。遠方の未到達領域へ連結を保ったまま到達できる(Local Geometric Control、
      論文がこの性質の代表例として名指ししている操作そのもの)。
    """
    return [
        GrammarVariant("C-1", "Add only (no Reversibility)", _rw(add_weight=1.0)),
        GrammarVariant("C-2", "C-1 + Remove (Reversibility)", _rw(add_weight=1.0, remove_weight=1.0)),
        GrammarVariant(
            "C-3", "C-2 + Split/Merge", _rw(add_weight=1.0, remove_weight=1.0, split_weight=1.0, merge_weight=1.0)
        ),
        GrammarVariant(
            "C-4",
            "C-3 + Snap/Unsnap",
            _rw(
                add_weight=1.0, remove_weight=1.0, split_weight=1.0, merge_weight=1.0, snap_weight=1.0,
                unsnap_weight=1.0,
            ),
        ),
        GrammarVariant(
            "C-F",
            "Full (+ AddAnywhere, Local Geom. Control)",
            _rw(
                add_weight=3.0, remove_weight=1.0, split_weight=1.0, merge_weight=1.0, snap_weight=1.0,
                unsnap_weight=1.0, add_anywhere_weight=1.0,
            ),
        ),
    ]


@dataclass
class RunResult:
    variant: str
    dataset: str
    shape_idx: int
    psnr: float
    n_primitives: int
    total_length: float
    final_loss: float
    elapsed_s: float


def make_city_args(rewrite_args: CityRewriteArgs) -> CityArgs:
    """バリアント間で固定するCityArgs(rewrite_args以外)。実行スクリプトで検証済みの既定値を使う。"""
    return CityArgs(
        cost_weight=1e-4,
        size_weight=0.0,
        cleanup_resolve_crossings=True,
        cleanup_max_iter=4,
        cleanup_min_seg=0.04,
        cleanup_min_angle=math.radians(20),
        decimate_dense=True,
        decimate_cell_size=0.1,
        decimate_max_per_cell=2,
        rewrite_args=rewrite_args,
        city_collection_args=CityCollectionArgs(width=0.05),
    )


def run_one(
    variant: GrammarVariant,
    dataset_name: str,
    target_img: torch.Tensor,
    shape_idx: int,
    render_args: RenderArgs,
    optim_args: OptimizeArgs,
    device: str,
    image_out: Optional[Path],
) -> RunResult:
    args = make_city_args(variant.rewrite_args)
    task = args.create(render_args, RasterLossArgs(), device, target_img)

    t0 = time.time()
    top_shape, loss, _all_objects, _all_metrics = optimize(
        task, optim_args, None, disable=True,
        desc=f"{variant.name}/{dataset_name}#{shape_idx}",
    )
    elapsed = time.time() - t0

    Collection = task.get_collection_constructor()
    coll = Collection.from_object(top_shape)
    assert isinstance(coll, CityNetworkCollection)
    density = coll.render01(render_args.size, render_args.lim, center_pixel=render_args.center_pixel, blur=render_args.blur)[0]
    mse = (density - target_img).square().mean().item()
    psnr = 10 * math.log10(1.0 / max(mse, 1e-12))
    p0 = top_shape.nodes[top_shape.edges[:, 0]]
    p1 = top_shape.nodes[top_shape.edges[:, 1]]
    total_length = (p1 - p0).norm(dim=-1).sum().item() if len(top_shape.edges) > 0 else 0.0

    if image_out is not None:
        img = task.visualize(coll, step=optim_args.n_steps, loss=loss, state=None)
        save_rgb8(image_out, img)

    return RunResult(
        variant=variant.name,
        dataset=dataset_name,
        shape_idx=shape_idx,
        psnr=psnr,
        n_primitives=len(top_shape.edges),
        total_length=total_length,
        final_loss=loss,
        elapsed_s=elapsed,
    )


def print_table2_style(results: list[RunResult], variants: list[GrammarVariant], dataset_names: list[str]) -> None:
    console = Console()
    table = Table(title="City文法アブレーション(design-for-descent.pdf Table 2形式)")
    table.add_column("Gr.")
    table.add_column("Description")
    for d in dataset_names:
        table.add_column(f"PSNR↑ {d}", justify="right")
    for d in dataset_names:
        table.add_column(f"#Prim↓ {d}", justify="right")

    by_key: dict[tuple[str, str], list[RunResult]] = {}
    for r in results:
        by_key.setdefault((r.variant, r.dataset), []).append(r)

    for v in variants:
        row = [v.name, v.description]
        for d in dataset_names:
            rs = by_key.get((v.name, d), [])
            row.append(f"{sum(r.psnr for r in rs) / len(rs):.1f}" if rs else "-")
        for d in dataset_names:
            rs = by_key.get((v.name, d), [])
            row.append(f"{sum(r.n_primitives for r in rs) / len(rs):.0f}" if rs else "-")
        table.add_row(*row)

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_steps", type=int, default=800)
    parser.add_argument("--n_shapes", type=int, default=1, help="各データセットから使う先頭n件の形状の数")
    parser.add_argument("--propose_every", type=int, default=25)
    parser.add_argument("--proposal_size", type=int, default=32)
    parser.add_argument("--render_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, default=PROJ_DIR / "output" / "ablation_city")
    parser.add_argument(
        "--variants", type=str, default=None, help="カンマ区切りでバリアント名を指定(既定: 全部 C-1,C-2,C-3,C-4,C-F)"
    )
    parser.add_argument(
        "--datasets", type=str, default=None, help="カンマ区切りでデータセット名を指定(既定: 全部 One,Dnt,Two)"
    )
    parser.add_argument("--no_save_images", action="store_true", help="各実行の最終画像を保存しない")
    parser.add_argument(
        "--stopping_patience",
        type=int,
        default=None,
        help="指定すると、rewriteラウンドでこの回数だけ相対改善が無ければ早期終了する(既定None=無効、"
        "常にn_stepsいっぱいまで実行)。バリアント間の比較の公平性より速度を優先したい場合に使う。",
    )
    args = parser.parse_args()

    all_variants = make_variants()
    if args.variants is not None:
        wanted = set(args.variants.split(","))
        all_variants = [v for v in all_variants if v.name in wanted]
        assert all_variants, f"No variants matched {args.variants}"

    dataset_names = list(DATASETS.keys())
    if args.datasets is not None:
        wanted_d = set(args.datasets.split(","))
        dataset_names = [d for d in dataset_names if d in wanted_d]
        assert dataset_names, f"No datasets matched {args.datasets}"

    render_args = RenderArgs(size=args.render_size, lim=(-1.5, 1.5), center_pixel=True, blur=1 / math.sqrt(2))
    optim_args = OptimizeArgs(
        n_steps=args.n_steps,
        scheduler="AdaptiveLR",
        lr=0.2,
        reduce_lr_min_lr=0.005,
        clip_grad=2.0,
        proposal_trigger="step",
        propose_every=args.propose_every,
        cleanup_every=args.propose_every + 1,
        proposal_size=args.proposal_size,
        proposal_criterion="loss",
        proposal_steps=1,
        batch_param_count=8192,
        # 既定はNone(早期終了なし、必ずn_stepsいっぱいまで実行)。バリアント間で同じ予算を
        # 与えて比較するため。--stopping_patience を指定すると速度優先で早期終了できる。
        stopping_patience=args.stopping_patience,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[RunResult] = []
    total_runs = len(all_variants) * len(dataset_names) * args.n_shapes
    run_i = 0
    t_start = time.time()

    for dataset_name in dataset_names:
        shc = torch.load(PROJ_DIR / DATASETS[dataset_name], weights_only=False)
        n_shapes = min(args.n_shapes, len(shc))
        target_imgs = [
            ShapeCollection.from_shape(shc[i]).render01(
                render_args.size, render_args.lim, center_pixel=render_args.center_pixel, blur=render_args.blur
            )[0].to(args.device)
            for i in range(n_shapes)
        ]

        for variant in all_variants:
            for shape_idx, target_img in enumerate(target_imgs):
                run_i += 1
                torch.manual_seed(args.seed + shape_idx)
                image_out = None
                if not args.no_save_images:
                    image_out = args.output_dir / f"{variant.name}_{dataset_name}_{shape_idx}.png"
                print(f"[{run_i}/{total_runs}] {variant.name} / {dataset_name} shape#{shape_idx} ...", flush=True)
                result = run_one(
                    variant, dataset_name, target_img, shape_idx, render_args, optim_args, args.device, image_out
                )
                results.append(result)
                print(
                    f"    -> PSNR={result.psnr:.2f} #edges={result.n_primitives} "
                    f"length={result.total_length:.2f} loss={result.final_loss:.3e} "
                    f"({result.elapsed_s:.1f}s)",
                    flush=True,
                )

    total_elapsed = time.time() - t_start
    print(f"\nAll {total_runs} runs finished in {total_elapsed:.1f}s")

    csv_path = args.output_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "dataset", "shape_idx", "psnr", "n_primitives", "total_length", "final_loss", "elapsed_s"])
        for r in results:
            writer.writerow([r.variant, r.dataset, r.shape_idx, r.psnr, r.n_primitives, r.total_length, r.final_loss, r.elapsed_s])
    print(f"Raw results saved to {csv_path}")

    print_table2_style(results, all_variants, dataset_names)


if __name__ == "__main__":
    main()

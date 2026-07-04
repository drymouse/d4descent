from pathlib import Path
from confify.builder import CLIBuilder
import shutil
import math

PROJ_DIR = Path(__file__).parent.parent
GEN_DIR = folder = Path(__file__).parent / "_generated"


def main():
    b = CLIBuilder("Road-F")
    b.add("--save_path", "output/###JOB_NAME###")
    b.add("--render.blur", 1 / math.sqrt(2))
    b.add("--img_mode", "bow")  # 画像の黒(ロゴ/市街地)=高密度になるよう反転する
    # loss (target_img のロード経路のみ流用。損失自体は RoadDensityTask のカスタム実装)
    b.add("---loss", "configs/losses/raster.yaml")
    # task
    b.add("---task", "configs/tasks/roads.yaml")
    b.add("--task.cost_weight", 1e-3)
    b.add("--task.cost_width_exponent", 2.5)  # 離散コスト: 幹線道路(幅広)の"追加"罰則を超線形に強くする
    # 建設コストを連続最適化にも効かせ「最小限の道路で被覆」させる。冗長な道路を縮め追加も抑える。
    b.add("--task.size_weight", 20.0)
    # 連続コストの幅指数。街路にコストを負担させようと下げると街路が崩壊するため離散と同じ2.5に保つ
    # (街路の散らかり・疎密の作り分けは街路コストではなく下の90度罰則が担う)。
    b.add("--task.size_width_exponent", 2.5)
    b.add("--task.mesh_weight", 0.05)  # ループ形成(meshedness)への報酬。都市らしい街区構造を促す
    # cleanup で密集地帯の街路交差点を間引く。1セル(cell_size)にmax_per_cellを超える街路ノードが
    # 集まったら1つに統合し、高密度域のスクリブル過密化を抑える(幹線・連結性は保持)。
    b.add("--task.decimate_dense", True)
    b.add("--task.decimate_cell_size", 0.06)
    b.add("--task.decimate_max_per_cell", 2)
    # 90度交差の選好(原則3)。gapが90度格子{90,180,270}°からずれることを罰する。街路を直交・直進させ、
    # 低人口域への無秩序な蛇行も抑える(実験で街路の低人口域漏れ64%→11%、90度からのずれ37°→10°)。
    b.add("--task.angle_deadzone", math.radians(10))  # 格子まわりの許容幅(ラジアン)
    b.add("--task.angle_penalty_exponent", 2.0)
    b.add("--task.angle_weight", 0.03)
    b.add("--task.road_collection_args.sigma0", 0.03)
    b.add("--task.road_collection_args.k_sigma", 300.0)
    b.add("--task.road_collection_args.reach_exponent", 2.5)  # 幹線道路の到達半径を不釣り合いに拡大
    b.add("--task.road_collection_args.amp_scale", 0.045)  # 幹線道路=薄く広く、街路=狭く大きく
    b.add("--task.road_collection_args.min_density_floor", 0.05)  # 道路が無くても保証される絶対的な最低ライン
    b.add("--task.target_inside_value", 0.7)  # 画像の高輝度側(市街地)の目標密度
    b.add("--task.target_outside_value", 0.15)  # 画像の低輝度側にも目標密度を持たせ、幹線道路が伸びる動機にする
    b.add("--task.rewrite_args.width_classes", [0.02, 0.05])
    b.add("--task.rewrite_args.length_range", [0.05, 0.15])
    b.add("--task.rewrite_args.snap_radius", 0.05)
    b.add("--task.rewrite_args.add_weight", 3.0)  # 先端から伸ばす候補を多めに生成する
    # AddAnywhere は連結を保ったまま遠方(ターゲット密度の高い領域)へ道路を広げる唯一の手段。
    # 層化サンプリングにより提案予算は各書き換えタイプへ公平に配分されるので、ここでは候補の
    # 多様性(狙う点の数)を確保するために weight を1.0にする（0.15だと候補が枯れて spreading が停滞した）。
    b.add("--task.rewrite_args.add_anywhere_weight", 1.0)
    b.add("--task.rewrite_args.n_add_anywhere_candidates", 32)
    # optim
    b.add("--optim.proposal_trigger", "step")
    b.add("--optim.propose_every", 25)
    b.add("--optim.proposal_size", 64)
    b.add("--optim.clip_grad", 2.0)
    b.add("--optim.n_steps", 5000)
    b.add("--optim.stopping_patience", 25)
    b.add("--optim.batch_param_count", 8192)
    b.add("--optim.proposal_criterion", "loss")
    b.add("--optim.proposal_steps", 1)
    b.add("--optim.scheduler", "AdaptiveLR")
    b.add("--optim.lr", 0.2)
    b.add("--optim.reduce_lr_min_lr", 0.005)
    # script
    b.add("--restart", False)

    b.add_sweep_set(
        {
            # TODO: 実際の人口密度マップに差し替える（白=高密度のグレースケール画像フォルダ）
            "_SIGLOGO": {
                "--png_path": "data/pngs/siglogo",
            },
        }
    )

    gen_dir = GEN_DIR / "roads_pngs"
    if gen_dir.exists():
        shutil.rmtree(gen_dir)
    gen_dir.mkdir(exist_ok=True, parents=True)

    for name, args, _ in b.build():
        file = gen_dir / f"{name}.sh"
        script = (
            f"#!/bin/bash\n\ncd {PROJ_DIR}\nuv run python scripts/optimize_pngs.py \\\n\t" + " \\\n\t".join(args) + "\n"
        )
        script = script.replace("###JOB_NAME###", name)
        file.write_text(script)
        file.chmod(0o755)
        print(str(file))


if __name__ == "__main__":
    main()

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
    b.add("--task.cost_width_exponent", 2.5)  # 幹線道路(幅広)への罰則を幅に対して超線形に強くする
    b.add("--task.mesh_weight", 0.05)  # ループ形成(meshedness)への報酬。都市らしい街区構造を促す
    b.add("--task.min_density_floor", 0.2)  # 人口密度の最低ライン
    b.add("--task.underflow_weight", 5.0)  # 最低ラインを下回った分への追加罰則
    b.add("--task.road_collection_args.sigma0", 0.03)
    b.add("--task.road_collection_args.k_sigma", 300.0)
    b.add("--task.road_collection_args.reach_exponent", 2.5)  # 幹線道路の到達半径を不釣り合いに拡大
    b.add("--task.road_collection_args.amp_scale", 0.045)  # 幹線道路=薄く広く、街路=狭く大きく
    b.add("--task.rewrite_args.width_classes", [0.02, 0.05])
    b.add("--task.rewrite_args.length_range", [0.05, 0.15])
    b.add("--task.rewrite_args.snap_radius", 0.05)
    b.add("--task.rewrite_args.add_weight", 3.0)  # 先端から伸ばす確率を上げる
    b.add("--task.rewrite_args.add_free_weight", 0.15)  # 任意の場所への追加確率を下げる
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

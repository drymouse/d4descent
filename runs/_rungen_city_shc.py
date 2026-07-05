from pathlib import Path
from confify.builder import CLIBuilder
import shutil
import math


PROJ_DIR = Path(__file__).parent.parent
GEN_DIR = folder = Path(__file__).parent / "_generated"


def main():
    b = CLIBuilder("City")
    b.add("--save_path", "output/###JOB_NAME###")
    b.add("--render.blur", 1 / math.sqrt(2))
    # loss: render01(道路網のSDFをぼかして[0,1]化)と target_img の素直なMSE
    b.add("---loss", "configs/losses/raster.yaml")
    # task
    b.add("---task", "configs/tasks/city.yaml")
    b.add("--task.cost_weight", 1e-4)  # 建設コストの離散正則化(冗長な道路の追加を少しだけ抑える)
    b.add("--task.size_weight", 0.0)  # 連続コスト。render01が既に1の場所は追加道路の損失が下がらないため既定は0
    # 90度交差の選好(原則3)。gapが90度格子{90,180,270}°からずれることを罰する。render01損失だけでは
    # 街路が任意角度で交差するスクリブルになりがちなので、これを加えて格子状の街路を促す。
    b.add("--task.angle_deadzone", math.radians(10))  # 格子まわりの許容幅(ラジアン)
    b.add("--task.angle_penalty_exponent", 2.0)
    b.add("--task.angle_weight", 0.05)
    # cleanup で密集地帯のノードと接続道路を間引く(スクリブル過密化を抑える)
    b.add("--task.decimate_dense", True)
    b.add("--task.decimate_cell_size", 0.06)
    b.add("--task.decimate_max_per_cell", 2)
    b.add("--task.city_collection_args.width", 0.02)  # 単一種類の道路(街路)の幅
    b.add("--task.rewrite_args.length_range", [0.05, 0.15])
    b.add("--task.rewrite_args.snap_radius", 0.05)
    b.add("--task.rewrite_args.add_weight", 3.0)  # 先端から伸ばす候補を多めに生成する
    # AddAnywhere は連結を保ったまま遠方(図形内部)へ道路を広げる唯一の手段。層化サンプリングにより
    # 提案予算は各書き換えタイプへ公平に配分されるので、ここでは候補の多様性(狙う点の数)を
    # 確保するために weight を1.0にする。
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

    # shc は複数のターゲット形状を含む（ArcLinesのベンチマーク用データを図形シルエットとして流用）。
    # 手早く試すだけなら生成後のシェルスクリプトに "--until 1" 等を足して1件だけ処理させるとよい。
    b.add_sweep_set(
        {
            "_OneComp": {
                "--target_points_path": "data/arclines/bench128.shc",
            },
            "_Donut": {
                "--target_points_path": "data/arclines/donut25.shc",
            },
            "_TwoComp": {
                "--target_points_path": "data/arclines/twocomp23.shc",
            },
        }
    )

    gen_dir = GEN_DIR / "city_shc"
    if gen_dir.exists():
        shutil.rmtree(gen_dir)
    gen_dir.mkdir(exist_ok=True, parents=True)

    for name, args, _ in b.build():
        file = gen_dir / f"{name}.sh"
        script = (
            f"#!/bin/bash\n\ncd {PROJ_DIR}\nuv run python scripts/optimize_shc.py \\\n\t" + " \\\n\t".join(args) + "\n"
        )
        script = script.replace("###JOB_NAME###", name)
        file.write_text(script)
        file.chmod(0o755)
        print(str(file))


if __name__ == "__main__":
    main()

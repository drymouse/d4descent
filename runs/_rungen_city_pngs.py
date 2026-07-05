from pathlib import Path
from confify.builder import CLIBuilder
import shutil
import math


PROJ_DIR = Path(__file__).parent.parent
GEN_DIR = folder = Path(__file__).parent / "_generated"


def main():
    b = CLIBuilder("City-F")
    b.add("--save_path", "output/###JOB_NAME###")
    b.add("--render.blur", 1 / math.sqrt(2))
    b.add("--img_mode", "bow")  # 画像の黒(ロゴ/市街地)=図形の内部(target=1)になるよう反転する
    # loss: render01(道路網のSDFをぼかして[0,1]化)と target_img の素直なMSE
    b.add("---loss", "configs/losses/raster.yaml")
    # task
    b.add("---task", "configs/tasks/city.yaml")
    b.add("--task.cost_weight", 1e-4)  # 建設コストの離散正則化(冗長な道路の追加を少しだけ抑える)
    b.add("--task.size_weight", 0.0)  # 連続コスト。render01が既に1の場所は追加道路の損失が下がらないため既定は0
    # cleanup で、ノードを共有せず幾何的に交差した2辺を検出し交点にノードを挿入して分割する
    # (Repairability: 交差する道路は必ずノードを共有する、という制約への修復)。
    b.add("--task.cleanup_resolve_crossings", True)
    b.add("--task.cleanup_max_iter", 4)
    b.add("--task.cleanup_min_seg", 0.04)
    b.add("--task.cleanup_min_angle", math.radians(20))
    # cleanup で密集地帯のノードと接続道路を間引く(交差解消の断片化暴走に対する安全弁も兼ねる)
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

    b.add_sweep_set(
        {
            # TODO: 実際の対象形状に差し替える（白=市街地のグレースケール画像フォルダ）
            "_SIGLOGO": {
                "--png_path": "data/pngs/siglogo",
            },
        }
    )

    gen_dir = GEN_DIR / "city_pngs"
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

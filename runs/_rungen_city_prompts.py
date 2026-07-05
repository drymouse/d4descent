from pathlib import Path
from confify.builder import CLIBuilder
import shutil
import math


PROJ_DIR = Path(__file__).parent.parent
GEN_DIR = folder = Path(__file__).parent / "_generated"


def main():
    b = CLIBuilder("CT-F_SDS")
    b.add("---", "configs/sds_600.yaml")
    b.add("--save_path", "output/###JOB_NAME###")
    b.add("--render.blur", 1 / math.sqrt(2))
    # stabilityai/stable-diffusion-2-1-base は2025年11月頃Stability AIにより一般提供が終了しており
    # (EU AI Act対応目的の非公開化。ライセンス同意では解決しない)、現在アクセスできない。
    # 動作確認済みの stable-diffusion-v1-5/stable-diffusion-v1-5 に明示的に差し替える。
    b.add("--loss.sd.pretrained_model_name_or_path", "stable-diffusion-v1-5/stable-diffusion-v1-5")
    # task
    b.add("---task", "configs/tasks/city.yaml")
    # cost_weight: 既定1e-4はRasterLossのMSEスケール(~1e-2)向けの値。SDSの損失はスケールが全く異なる
    # (数十〜数百になりうる)ため、そのままだと単純さ罰則が実質ゼロになりエッジ数が爆発する恐れがある。
    # まず1e-4で1本回してログのlossとエッジ数Eを見て、Eが数百を超えて増え続けるなら1e-3->1e-2と上げる。
    b.add("--task.cost_weight", 1e-4)
    b.add("--task.size_weight", 0.0)
    b.add("--task.cleanup_resolve_crossings", True)
    b.add("--task.cleanup_max_iter", 4)
    b.add("--task.cleanup_min_seg", 0.04)
    b.add("--task.cleanup_min_angle", math.radians(20))
    b.add("--task.decimate_dense", True)
    b.add("--task.decimate_cell_size", 0.1)
    b.add("--task.decimate_max_per_cell", 2)
    # SDのVAEが認識するには十分な太さ。細くする場合も0.03未満にはしない(VAEで潰れて勾配が消える)
    b.add("--task.city_collection_args.width", 0.05)
    b.add("--task.rewrite_args.length_range", [0.05, 0.15])
    b.add("--task.rewrite_args.snap_radius", 0.05)
    b.add("--task.rewrite_args.add_weight", 3.0)
    b.add("--task.rewrite_args.add_anywhere_weight", 1.0)
    b.add("--task.rewrite_args.n_add_anywhere_candidates", 32)

    # プロンプト(まずこの4本。1本目が本命)。SDSLossMixinは1-render01(=白地に黒の道路)をSDに渡すので、
    # "black lines on white background" というsuffixはレンダリングの見た目と一致する。
    b.add(
        "--prompts",
        [
            "street map of a city",
            "road network of manhattan, grid streets",
            "radial street map of paris",
            "street map of a medieval european city",
        ],
    )
    b.add("--prompt_suffix", "top-down map, black lines on white background")

    # optim
    b.add("--optim.proposal_trigger", "step")
    b.add("--optim.propose_every", 75)
    b.add("--optim.proposal_size", 32)  # VRAMが苦しければ16へ
    b.add("--optim.scheduler", "none")
    b.add("--optim.lr", 0.01)
    b.add("--optim.batch_size", 4)
    b.add("--optim.n_steps", 800)  # まず800で回して映像を確保する(UR-Fは1500)
    b.add("--optim.proposal_criterion", "loss")
    b.add("--optim.proposal_steps", 1)
    b.add("--restart", False)

    gen_dir = GEN_DIR / "city_prompts"
    if gen_dir.exists():
        shutil.rmtree(gen_dir)
    gen_dir.mkdir(exist_ok=True, parents=True)

    for name, args, _ in b.build():
        file = gen_dir / f"{name}.sh"
        script = (
            f"#!/bin/bash\n\ncd {PROJ_DIR}\nuv run python scripts/optimize_prompts.py \\\n\t"
            + " \\\n\t".join(args)
            + "\n"
        )
        script = script.replace("###JOB_NAME###", name)
        file.write_text(script)
        file.chmod(0o755)
        print(str(file))


if __name__ == "__main__":
    main()

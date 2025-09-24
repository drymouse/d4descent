from pathlib import Path
from confify.builder import CLIBuilder
import shutil
import math

PROJ_DIR = Path(__file__).parent.parent
GEN_DIR = folder = Path(__file__).parent / "_generated"


def main():
    b = CLIBuilder("UR-F")
    b.add("--save_path", "output/###JOB_NAME###")
    b.add("--render.blur", 1 / math.sqrt(2))
    # loss
    b.add("---loss", "configs/losses/raster.yaml")
    # task
    b.add("---task", "configs/tasks/csgb.yaml")
    b.add("--task.rewrite_args.merge_threshold", 100)
    b.add("--task.rewrite_args.remove_threshold", 0.005)
    b.add("--task.rewrite_args.add_hole_weight", 1.0)
    b.add("--task.csgb_args.offset_scale", 0.0)
    b.add("--task.cleanup_strategy", "smarter")

    b.add_sweep_set(
        {
            "": {
                "--optim.proposal_criterion": "loss",
                "--optim.proposal_steps": 1,
            },
        }
    )

    b.add_sweep_set(
        {
            "": {
                "--task.csgb_args.theta_scale": math.pi,
            },
        }
    )

    b.add_sweep_set(
        {
            "": {
                "--optim.scheduler": "AdaptiveLR",
                "--optim.lr": 0.2,
                "--optim.reduce_lr_min_lr": 0.005,
            },
        }
    )

    # optim
    b.add("--optim.proposal_trigger", "step")
    b.add("--optim.propose_every", 25)
    b.add("--optim.proposal_size", 64)
    b.add("--optim.clip_grad", 2.0)
    b.add("--optim.n_steps", 5000)
    b.add("--optim.stopping_patience", 25)
    b.add("--optim.batch_param_count", 8192)
    b.add("--restart", False)

    b.add_sweep_set(
        {
            "_SIGLOGO": {
                "--task.node_weight": 1e-6,
                "--png_path": "data/pngs/siggraph",
            },
            "_SIGGRAPH": {
                "--task.node_weight": 1e-4,
                "--png_path": "data/pngs/letters",
            },
        }
    )

    gen_dir = GEN_DIR / "ur_pngs"
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

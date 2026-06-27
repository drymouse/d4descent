from pathlib import Path
from confify.builder import CLIBuilder
import shutil
import math

PROJ_DIR = Path(__file__).parent.parent
GEN_DIR = folder = Path(__file__).parent / "_generated"


def main():
    b = CLIBuilder("")
    b.add("--save_path", "output/###JOB_NAME###")
    b.add("--render.blur", 1 / math.sqrt(2))
    # loss
    b.add("---loss", "configs/losses/raster.yaml")
    # task
    b.add("---task", "configs/tasks/tri.yaml")
    b.add("--task.rewrite_args.remove_threshold", 0.005)
    # b.add("--task.rewrite_args.add_hole_weight", 1.0)
    # b.add("--task.ur_args.offset_scale", 0.0)
    # b.add("--task.cleanup_strategy", "smarter")
    # b.add("--task.node_weight", 1e-4)
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
    # script
    b.add("--restart", False)

    b.add_sweep_set(
        {
            "Tri-F": {},
            "Tri-1": {
                "--task.rewrite_args.add_tri_weight": 0,
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
            # "_NoStep": {
            #     "--optim.proposal_steps": 0,
            # },
            # "_OneRewrite": {
            #     "--optim.proposal_accept_parallel": False,
            # },
            # "_Fixed": {
            #     "--optim.scheduler": "none",
            #     "--optim.lr": 0.02,
            # },
            # "_OneRewrite_Fixed": {
            #     "--optim.proposal_accept_parallel": False,
            #     "--optim.scheduler": "none",
            #     "--optim.lr": 0.02,
            # },
        }
    )

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

    gen_dir = GEN_DIR / "tri"
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

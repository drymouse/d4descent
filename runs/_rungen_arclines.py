from pathlib import Path
from confify.builder import CLIBuilder
import shutil
import math

PROJ_DIR = Path(__file__).parent.parent
GEN_DIR = Path(__file__).parent / "_generated"


def main():
    b = CLIBuilder("")
    b.add("--save_path", "output/###JOB_NAME###")
    b.add("--render.blur", 1 / math.sqrt(2))
    # loss
    b.add("--loss.\\$type", "d4descent.losses.raster.RasterLossArgs")
    # task
    b.add("--task.\\$type", "d4descent.tasks.arclines.ArclinesArgs")
    b.add("--task.rewrite_args.add_hole_random", True)
    b.add("--task.arclines_args.ks_scale", 1 / math.sqrt(2))
    # optim
    b.add("--optim.proposal_trigger", "step")
    b.add("--optim.propose_every", 25)
    b.add("--optim.proposal_size", 64)
    b.add("--optim.n_steps", 5000)
    b.add("--optim.stopping_patience", 25)
    b.add("--optim.batch_param_count", 8192)
    b.add("--optim.proposal_criterion", "loss")
    b.add("--optim.proposal_steps", 1)
    b.add("--restart", False)

    b.add_sweep_set(
        {
            "AL-F": {
                # "--task.rewrite_args.add_holes_weight": 0.25, default
            },
            # "AL-1": {
            #     "--task.rewrite_args.add_holes_weight": 0.0,
            # },
        }
    )


    b.add_sweep_set(
        {
            "": {
                "--optim.scheduler": "AdaptiveLR",
                "--optim.lr": 0.5,
            },
            # "_NoStep": {
            #     "--optim.proposal_steps": 0,
            # },
            # "_OneRewrite": {
            #     "--optim.proposal_accept_parallel": False,
            # },
            # "_Fixed": {
            #     "--optim.scheduler": "none",
            #     "--optim.lr": 0.05,
            # },
            # "_OneRewrite_Fixed": {
            #     "--optim.proposal_accept_parallel": False,
            #     "--optim.scheduler": "none",
            #     "--optim.lr": 0.05,
            # }
        }
    )

    b.add_sweep_set(
        {
            "": {
                "--task.line_weight": 1e-5,
                "--task.arc_weight": 1e-5,
            },
            # "_w14": {
            #     "--task.line_weight": 1e-4,
            #     "--task.arc_weight": 1e-4,
            # },
            # "_w16": {
            #     "--task.line_weight": 1e-6,
            #     "--task.arc_weight": 1e-6,
            # },
            # "_w17": {
            #     "--task.line_weight": 1e-7,
            #     "--task.arc_weight": 1e-7,
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


    gen_dir = GEN_DIR / "arclines"
    if gen_dir.exists():
        shutil.rmtree(gen_dir)
    gen_dir.mkdir(exist_ok=True, parents=True)

    for name, args, _ in b.build():
        file = gen_dir / f"{name}.sh"
        script = f"#!/bin/bash\n\ncd {PROJ_DIR}\nuv run python scripts/optimize_shc.py \\\n\t" + " \\\n\t".join(args) + "\n"
        script.replace("###JOB_NAME###", name)
        file.write_text(script)
        file.chmod(0o755)
        print(str(file))


if __name__ == "__main__":
    main()

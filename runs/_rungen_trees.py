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
    b.add("---task", "configs/tasks/trees.yaml")
    b.add("--task.tree_args.optimize_roots", False)
    b.add("--task.tree_args.ls_max", 0.2)
    b.add("--task.tree_args.scale_strategy", "linear")
    b.add("--task.tree_args.leaf_shape", "leaf1")
    b.add("--task.node_weight", 1e-4)

    # optim
    b.add("--optim.proposal_trigger", "step")
    b.add("--optim.propose_every", 25)
    b.add("--optim.proposal_size", 64)
    b.add("--optim.scheduler", "none")
    b.add("--optim.lr", 0.5)
    b.add("--optim.reduce_lr_min_lr", 0.1)
    b.add("--optim.clip_grad", 2.0)
    b.add("--optim.n_steps", 5000)
    b.add("--optim.stopping_patience", 10)
    b.add("--optim.batch_param_count", 4096)
    b.add("--optim.proposal_criterion", "loss")
    b.add("--optim.proposal_steps", 1)

    b.add("--restart", False)

    b.add_sweep_set(
        {
            # "Tr-1": {
            #     "--task.rewrite_args.add_branch": True,
            #     "--task.rewrite_args.add_branch_epsilon": False,
            #     "--task.rewrite_args.split_branch": False,
            #     "--task.rewrite_args.remove_branch": False,
            # },
            # "Tr-2": {
            #     "--task.rewrite_args.add_branch": True,
            #     "--task.rewrite_args.add_branch_epsilon": False,
            #     "--task.rewrite_args.split_branch": False,
            #     "--task.rewrite_args.remove_branch": True,
            #     "--task.rewrite_args.remove_branch_epsilon": False,
            # },
            # "Tr-3": {
            #     "--task.rewrite_args.add_branch": True,
            #     "--task.rewrite_args.add_branch_epsilon": True,
            #     "--task.rewrite_args.split_branch": False,
            #     "--task.rewrite_args.remove_branch": True,
            #     "--task.rewrite_args.remove_branch_epsilon": True,
            # },
            # "Tr-4": {
            #     "--task.rewrite_args.add_branch": True,
            #     "--task.rewrite_args.add_branch_epsilon": True,
            #     "--task.rewrite_args.split_branch": True,
            #     "--task.rewrite_args.remove_branch": True,
            #     "--task.rewrite_args.remove_non_leaf": True,
            # },
            # "Tr-5": {
            #     "--task.rewrite_args.add_branch": True,
            #     "--task.rewrite_args.add_branch_epsilon": True,
            #     "--task.rewrite_args.split_branch": True,
            #     "--task.rewrite_args.remove_branch": True,
            #     "--task.rewrite_args.remove_non_leaf": True,
            #     "--task.rewrite_args.add_anywhere": True,
            # },
            "Tr-F": {
                "--task.rewrite_args.add_branch": True,
                "--task.rewrite_args.add_branch_epsilon": True,
                "--task.rewrite_args.split_branch": True,
                "--task.rewrite_args.remove_branch": True,
                "--task.rewrite_args.remove_non_leaf": True,
                "--task.rewrite_args.add_anywhere": True,
                "--task.rewrite_args.add_anywhere_last_r": 0.03,
                "--task.cleanup_small_leaves": 0.03,
                "--task.tree_args.optimize_rs": True,
                "--task.tree_args.rs_max": 0.075,
                "--task.init_strategy": "frame-bottom",
            },
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
        }
    )

    b.add_sweep_set(
        {
            "": {
                "--task.tree_args.theta_mode": "rel",
                "--task.tree_args.theta_min": -math.radians(15),
                "--task.tree_args.theta_max": math.radians(15),
                "--task.rewrite_args.random_angle": False,
                "--task.rewrite_args.default_angle": math.radians(15),
            },
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

    gen_dir = GEN_DIR / "trees"
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

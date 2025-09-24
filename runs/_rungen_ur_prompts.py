from pathlib import Path
from confify.builder import CLIBuilder
import shutil
import math


PROJ_DIR = Path(__file__).parent.parent
GEN_DIR = folder = Path(__file__).parent / "_generated"


def main():
    b = CLIBuilder("UR-F_SDS")
    b.add("---", "configs/sds_600.yaml")
    b.add("--save_path", "output/###JOB_NAME###")
    b.add("--render.blur", 1 / math.sqrt(2))
    # task
    b.add("---task", "configs/tasks/ur.yaml")
    b.add("--task.rewrite_args.add_hole_weight", 1.0)
    b.add("--task.ur_args.offset_scale", 0.0)
    b.add("--task.cleanup_strategy", "smarter")
    b.add("--task.node_weight", 1e-6)
    b.add(
        "--prompts",
        [
            "horse",
            "bull",
            "skull",
            "wineglass",
            "cat",
            "tree",
            "flower",
            "einstein",
            "astronaut",
            "dog",
            "dolphin",
            "whale",
            "puppy",
            "cloth",
            "chicken",
            "car",
            "boat",
            "scissor",
            "pants",
            "heart",
            "racket",
            "man standing",
            "bottle",
            "tripod",
        ],
    )

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
                "--task.ur_args.theta_scale": math.pi,
            },
        }
    )

    # optim
    b.add("--optim.proposal_trigger", "step")
    b.add("--optim.propose_every", 75)
    b.add("--optim.proposal_size", 64)
    b.add("--optim.scheduler", "none")
    b.add("--optim.lr", 0.01)
    b.add("--optim.batch_size", 4)
    b.add("--optim.n_steps", 1500)
    b.add("--restart", False)

    gen_dir = GEN_DIR / "ur_prompts"
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

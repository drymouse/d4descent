from pathlib import Path
from confify.builder import CLIBuilder
import shutil
import math

PROJ_DIR = Path(__file__).parent.parent
GEN_DIR = folder = Path(__file__).parent / "_generated"


def main():
    b = CLIBuilder("AL-F_SDS")
    b.add("---", "configs/sds_600.yaml")
    b.add("--save_path", "output/###JOB_NAME###")
    b.add("--render.blur", 1 / math.sqrt(2))
    # task
    b.add("---task", "configs/tasks/arclines.yaml")
    b.add("--task.rewrite_args.add_hole_random", True)
    b.add("--task.arclines_args.ks_scale", 1 / math.sqrt(2))
    # optim
    b.add("--optim.proposal_trigger", "step")
    b.add("--optim.propose_every", 75)
    b.add("--optim.proposal_size", 64)
    b.add("--optim.cleanup_every", 5)
    b.add("--optim.scheduler", "none")
    b.add("--optim.lr", 0.02)
    b.add("--optim.batch_size", 4)
    b.add("--optim.n_steps", 1500)
    b.add("--optim.proposal_criterion", "loss")
    b.add("--optim.proposal_steps", 1)
    # script
    b.add("--restart", False)
    # prompts
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
                "--task.line_weight": 1e-7,
                "--task.arc_weight": 2e-7,
            },
            # "_w-125": {
            #     "--task.line_weight": 1e-5,
            #     "--task.arc_weight": 2e-5,
            # },
            # "_w-126": {
            #     "--task.line_weight": 1e-6,
            #     "--task.arc_weight": 2e-6,
            # },
        }
    )

    gen_dir = GEN_DIR / "arclines_prompts"
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

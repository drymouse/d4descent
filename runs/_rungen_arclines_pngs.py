from pathlib import Path
from confify.builder import CLIBuilder
import shutil
import math

PROJ_DIR = Path(__file__).parent.parent
GEN_DIR = folder = Path(__file__).parent / "_generated"


def main():
    b = CLIBuilder("AL-F")
    b.add("--save_path", "output/###JOB_NAME###")
    b.add("--render.blur", 1 / math.sqrt(2))
    # loss
    b.add("---loss", "configs/losses/raster.yaml")
    # task
    b.add("---task", "configs/tasks/arclines.yaml")
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
    # script
    b.add("--restart", False)

    b.add_sweep_set(
        {
            "": {
                "--optim.scheduler": "AdaptiveLR",
                "--optim.lr": 0.5,
            },
        }
    )
    b.add_sweep_set(
        {
            "_SIGLOGO": {
                "--task.line_weight": 1e-7,
                "--task.arc_weight": 1e-7,
                "--png_path": "data/pngs/siggraph",
            },
            "_SIGGRAPH": {
                "--png_path": "data/pngs/letters",
                "--task.line_weight": 1e-5,
                "--task.arc_weight": 1e-5,
            },
        }
    )

    gen_dir = GEN_DIR / "arclines_pngs"
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

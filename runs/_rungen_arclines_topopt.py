from pathlib import Path
from confify.builder import CLIBuilder
import shutil
import math

PROJ_DIR = Path(__file__).parent.parent
GEN_DIR = folder = Path(__file__).parent / "_generated"


def main():
    b = CLIBuilder("AL-F_Topopt")
    b.add("--save_path", "output/###JOB_NAME###")
    b.add("--render.blur", 1 / math.sqrt(2))
    b.add("---", "configs/arclines_topopt.yaml")

    b.add_sweep_set(
        {
            "": {
                "--loss.occ_blur": 1 / math.sqrt(2),
                "--optim.proposal_criterion": "loss",
                "--optim.proposal_steps": 1,
            },
        }
    )

    b.add_sweep_set(
        {
            "_Cantilever": {
                "--loss.init_setup": "cantileaver",
            },
            "_MBB": {
                "--loss.init_setup": "mbb",
            },
            "_MBBHalf": {
                "--loss.init_setup": "mbb_half",
            },
        }
    )

    b.add("--task.line_weight", 1e-6)
    b.add("--task.arc_weight", 2e-6)
    b.add("--task.rewrite_args.add_hole_random", True)
    b.add_sweep_set(
        {
            "": {
                "--task.arclines_args.ks_scale": 1 / math.sqrt(2),
            },
        }
    )

    # optim
    b.add("--optim.proposal_trigger", "step")
    b.add("--optim.propose_every", 25)
    b.add("--optim.proposal_size", 64)
    b.add("--optim.scheduler", "none")
    b.add("--optim.lr", 0.01)
    b.add("--optim.n_steps", 2000)
    b.add("--restart", False)

    gen_dir = GEN_DIR / "arclines_topopt"
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

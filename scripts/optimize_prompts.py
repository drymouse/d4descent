from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Callable, Literal, Union, cast
import torch
import yaml
import shutil
import numpy as np
from rich.pretty import pprint
from rich.traceback import install
from confify import read_config_from_cli, config_dump_yaml
from dataclasses import replace
from matplotlib.figure import Figure

install()  # Enable rich traceback

from d4descent.tasks._base import Task, TaskArgs, RenderArgs
from d4descent.losses.sds import SDSLossArgs
from d4descent.object_collection import ObjectCollection
from d4descent.util import torch_load, save_rgb8, save_video, read_points_npz, register_slurm_signal_handlers_auto
from d4descent.optimizer import OptimizeArgs, optimize, OnVisualizeFunc


@dataclass
class Args:
    task: TaskArgs
    loss: SDSLossArgs
    prompts: list[str]
    prompt_suffix: Optional[str] = None
    render: RenderArgs = field(default_factory=RenderArgs)
    save_path: Optional[Path] = None
    optim: OptimizeArgs = field(default_factory=OptimizeArgs)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    restart: bool = False
    skip: int = 0


register_slurm_signal_handlers_auto()


def main():
    args = read_config_from_cli(Args)
    pprint(args)
    if args.save_path is not None:
        save_path = Path(args.save_path)
        if args.restart and save_path.exists():
            shutil.rmtree(save_path)
        save_path.mkdir(exist_ok=True, parents=True)
        config_dump_yaml(args, save_path / "config.yaml")

    for i, prompt in enumerate(args.prompts):
        if args.prompt_suffix is not None:
            prompt += f", {args.prompt_suffix}"
        if i < args.skip:
            continue
        name = f"{i:02d}_{prompt.replace(',', '').replace(' ', '-')[:50]}"
        print(f"==== Processing {name} ({i + 1}/{len(args.prompts)}) ====")
        save_path = Path(args.save_path) / name if args.save_path is not None else None
        on_visualize: Optional[OnVisualizeFunc] = None

        if save_path is not None:
            if not args.restart and (save_path / "video.mp4").exists():
                print("--> Skipping")
                continue
            save_path.mkdir(exist_ok=True, parents=True)

            def on_visualize_(img: np.ndarray, step: int, loss: float):
                assert save_path is not None
                all_imgs.append(img)
                save_rgb8(save_path / "last.png", img)

            on_visualize = on_visualize_

        cur_loss = replace(args.loss, prompt=prompt)
        task = args.task.create(args.render, cur_loss, args.device, None)

        optim = replace(args.optim)
        retry = 5
        while retry > 0:
            try:
                all_imgs: list[np.ndarray] = []

                top_shape, loss, all_objects, all_metrics = optimize(task, optim, on_visualize)
                if save_path is not None:
                    Collection = task.get_collection_constructor()
                    torch.save(Collection.from_object(top_shape).to_savable(), save_path / "topshape.objc")
                    torch.save(all_objects.to_savable(), save_path / "all_objects.objc")
                    torch.save(all_metrics, save_path / "metrics.pt")
                    save_video(save_path / "video.mp4", all_imgs, fps=5)
                break
            except torch.cuda.OutOfMemoryError:
                print("--> Out of memory, retrying")
                if optim.batch_size is not None:
                    if optim.batch_size == 1:
                        raise RuntimeError("Out of memory")
                    optim.batch_size = optim.batch_size // 2
                else:
                    optim.batch_param_count = optim.batch_param_count // 2
                retry -= 1


if __name__ == "__main__":
    main()

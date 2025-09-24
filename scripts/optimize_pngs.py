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
import functools
from PIL import Image
import torchvision.transforms.functional as TF
import time

install()  # Enable rich traceback

from d4descent.tasks._base import Task, TaskArgs, RenderArgs, ExtraMetrics, update_extra_metrics
from d4descent.losses._base import LossArgs
from d4descent.object_collection import ObjectCollection
from d4descent.util import torch_load, save_rgb8, save_video, read_points_npz, register_slurm_signal_handlers_auto
from d4descent.optimizer import OptimizeArgs, optimize, OnVisualizeFunc


@dataclass
class Args:
    task: TaskArgs
    loss: LossArgs
    png_path: Path
    render: RenderArgs = field(default_factory=RenderArgs)
    img_mode: Literal["bow", "wob"] = "bow"  # bow: black on white, wob: white on black
    save_path: Optional[Path] = None
    optim: OptimizeArgs = field(default_factory=OptimizeArgs)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    restart: bool = False


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

    name = f"0"
    print(f"==== Processing {name} (1/1) ====")
    save_path = Path(args.save_path) / name if args.save_path is not None else None
    on_visualize: Optional[OnVisualizeFunc] = None

    if save_path is not None:
        if not args.restart and (save_path / "video.mp4").exists():
            print("--> Skipping")
            return
        save_path.mkdir(exist_ok=True, parents=True)

        def on_visualize_(img: np.ndarray, step: int, loss: float):
            assert save_path is not None
            all_imgs.append(img)
            save_rgb8(save_path / "last.png", img)

        on_visualize = on_visualize_

    pngs = sorted(args.png_path.iterdir())
    print([p.name for p in pngs])
    target_imgs = [
        TF.rgb_to_grayscale(
            TF.resize(TF.to_tensor(Image.open(png)).to(args.device), [args.render.size, args.render.size])
        )
        .squeeze(0)
        .flip(0)
        for png in pngs
    ]  # (size, size)
    if args.img_mode == "bow":
        target_imgs = [1 - img for img in target_imgs]

    optim = replace(args.optim)
    retry = 5
    while retry > 0:
        try:
            all_imgs: list[np.ndarray] = []
            all_all_objects: list[ObjectCollection] = []
            all_all_metrics: ExtraMetrics = {}
            n_steps: list[int] = []

            top_shape = None
            task = None

            def initialize_object():
                return top_shape

            start_time = time.time()

            for target_img in target_imgs:
                task = args.task.create(args.render, args.loss, args.device, target_img)
                task.start_time = start_time
                if top_shape is not None:
                    task.initialize_object = initialize_object
                top_shape, loss, all_objects, all_metrics = optimize(task, optim, on_visualize)

                all_all_objects.append(all_objects)
                all_all_metrics = update_extra_metrics(all_all_metrics, all_metrics)
                n_steps.append(len(all_objects))

            if save_path is not None:
                assert task is not None
                Collection = task.get_collection_constructor()
                torch.save(Collection.from_object(top_shape).to_savable(), save_path / "topshape.objc")
                torch.save(Collection.cat(all_all_objects).to_savable(), save_path / "all_objects.objc")
                torch.save(all_all_metrics, save_path / "metrics.pt")
                torch.save(n_steps, save_path / "n_steps.pt")
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

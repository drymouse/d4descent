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
import time
import json

install()  # Enable rich traceback

from d4descent.tasks._base import Task, TaskArgs, RenderArgs, update_extra_metrics
from d4descent.losses._base import LossArgs
from d4descent.object_collection import ObjectCollection
from d4descent.objects.prim2 import ShapeCollection, Shape
from d4descent.util import torch_load, save_rgb8, save_video, read_points_npz, register_slurm_signal_handlers_auto
from d4descent.optimizer import OptimizeArgs, optimize, OnVisualizeFunc


@dataclass
class Args:
    task: TaskArgs
    loss: LossArgs
    target_points_path: Path
    target_pairs: list[tuple[int, int]]
    save_path: Path
    render: RenderArgs = field(default_factory=RenderArgs)
    optim: OptimizeArgs = field(default_factory=OptimizeArgs)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    restart: bool = False
    skip: int = 0


register_slurm_signal_handlers_auto()


def main():
    args = read_config_from_cli(Args)
    pprint(args)
    target_points_path = Path(args.target_points_path)
    shc = torch_load(target_points_path, ShapeCollection).to(args.device)
    shc_id_map: dict[int, int] = {id_: i for i, id_ in enumerate(shc.shape_ids)}

    if args.save_path is not None:
        save_path = Path(args.save_path)
        if args.restart and save_path.exists():
            shutil.rmtree(save_path)
        save_path.mkdir(exist_ok=True, parents=True)
        config_dump_yaml(args, save_path / "config.yaml")

    metrics_path = args.save_path / "metrics.json"

    if metrics_path.exists():
        cum_metrics: dict[str, float] = json.loads(metrics_path.read_text())
    else:
        cum_metrics: dict[str, float] = {
            "total_objects": len(args.target_pairs),
            "$completed": 0,
            "sum_loss": 0,
            "sum_loss_cont": 0,
            "sum_loss_simp": 0,
            "sum_time": 0,
            "sum_pct_matches": 0,
            "sum_pct_perfect_matches": 0,
        }

    for i, (id1, id2) in enumerate(args.target_pairs):
        if i < args.skip:
            continue
        name = f"{id1}_{id2}"
        print(f"==== Processing {name} ({i + 1}/{len(args.target_pairs)}) ====")
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

        target_shape1 = shc[shc_id_map[id1]]
        target_shape2 = shc[shc_id_map[id2]]

        optim = replace(args.optim)
        retry = 5
        while retry > 0:
            try:
                all_imgs: list[np.ndarray] = []
                target_img = ShapeCollection.from_shape(target_shape1).render01(
                    args.render.size, args.render.lim, center_pixel=args.render.center_pixel, blur=args.render.blur
                )[0]
                start_time = time.time()
                task = args.task.create(args.render, args.loss, args.device, target_img)
                task.start_time = start_time

                top_shape, loss, all_objects, all_metrics = optimize(task, optim, on_visualize)

                target_img2 = ShapeCollection.from_shape(target_shape2).render01(
                    args.render.size, args.render.lim, center_pixel=args.render.center_pixel, blur=args.render.blur
                )[0]
                task2 = args.task.create(args.render, args.loss, args.device, target_img2)
                task2.start_time = start_time

                def initialize_object2():
                    return top_shape

                task2.initialize_object = initialize_object2
                top_shape2, loss, all_objects2, all_metrics2 = optimize(task2, optim, on_visualize)

                if save_path is not None:
                    cum_metrics["$completed"] += 1
                    cum_metrics["sum_loss"] += all_metrics2["$loss"][-1]
                    cum_metrics["sum_loss_cont"] += all_metrics2["$loss_cont"][-1]
                    cum_metrics["sum_loss_simp"] += all_metrics2["$loss_simp"][-1]
                    cum_metrics["sum_time"] += all_metrics2["$timestamp"][-1]
                    Collection = task.get_collection_constructor()
                    torch.save(Collection.from_object(top_shape).to_savable(), save_path / "topshape.objc")
                    torch.save(Collection.cat([all_objects, all_objects2]).to_savable(), save_path / "all_objects.objc")
                    torch.save(update_extra_metrics(all_metrics, all_metrics2), save_path / "metrics.pt")
                    torch.save([len(all_objects), len(all_objects2)], save_path / "n_steps.pt")
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

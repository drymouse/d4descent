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
import json

install()  # Enable rich traceback

from d4descent.tasks._base import Task, TaskArgs, RenderArgs
from d4descent.losses._base import LossArgs
from d4descent.objects.arclines import ShapeCollection, Shape, ShapeCollectionArgs
from d4descent.util import (
    torch_load,
    save_rgb8,
    save_video,
    read_points_npz,
    register_slurm_signal_handlers_auto,
    get_default_device,
)
from d4descent.metrics import compute_metric
from d4descent.optimizer import OptimizeArgs, optimize, OnVisualizeFunc


@dataclass
class Args:
    task: TaskArgs
    loss: LossArgs
    save_path: Path
    render: RenderArgs = field(default_factory=RenderArgs)
    target_points_path: Optional[Path] = None
    optim: OptimizeArgs = field(default_factory=OptimizeArgs)
    device: str = get_default_device()
    restart: bool = False
    skip: int = 0
    until: Optional[int] = None


register_slurm_signal_handlers_auto()


def main():
    args = read_config_from_cli(Args)
    pprint(args)
    if args.target_points_path is not None:
        target_points_path = Path(args.target_points_path)
        shc = torch_load(target_points_path, ShapeCollection).to(args.device)
    else:
        # dummy shape
        shc = ShapeCollection.from_shapes([Shape.create_circle_lines(4, (0, 0), 1)]).to(args.device)
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
            "total_objects": len(shc),
            "$completed": 0,
            "sum_loss": 0,
            "sum_loss_cont": 0,
            "sum_loss_simp": 0,
            "sum_time": 0,
            "sum_pct_matches": 0,
            "sum_pct_perfect_matches": 0,
        }

    for i, target_shape in enumerate(shc):
        if i < args.skip:
            continue
        if args.until is not None and i >= args.until:
            break
        name = str(target_shape.id)
        print(f"==== Processing {name} ({i + 1}/{len(shc)}) ====")
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

        optim = replace(args.optim)
        retry = 5
        while retry > 0:
            try:
                all_imgs: list[np.ndarray] = []
                target_img = ShapeCollection.from_shape(target_shape).render01(
                    args.render.size, args.render.lim, center_pixel=args.render.center_pixel, blur=args.render.blur
                )[0]
                task = args.task.create(args.render, args.loss, args.device, target_img)

                top_shape, loss, all_objects, all_metrics = optimize(task, optim, on_visualize)
                if save_path is not None:
                    cum_metrics["$completed"] += 1
                    cum_metrics["sum_loss"] += all_metrics["$loss"][-1]
                    cum_metrics["sum_loss_cont"] += all_metrics["$loss_cont"][-1]
                    cum_metrics["sum_loss_simp"] += all_metrics["$loss_simp"][-1]
                    cum_metrics["sum_time"] += all_metrics["$timestamp"][-1]
                    if isinstance(top_shape, Shape):
                        metrics = compute_metric(top_shape, target_shape)
                        summary_ = metrics.summarize(0.8)
                        cum_metrics["sum_pct_matches"] += summary_.pct_matches
                        cum_metrics["sum_pct_perfect_matches"] += summary_.pct_perfect_matches
                        torch.save(
                            (ShapeCollection.from_shapes([target_shape, top_shape]).to_savable(), loss, metrics),
                            save_path / "results.bsr",
                        )

                    Collection = task.get_collection_constructor()
                    metrics_path.write_text(json.dumps(cum_metrics))
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

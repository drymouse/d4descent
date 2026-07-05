"""
標準入力からプロンプトを1行ずつ受け取り、その都度SDS最適化を実行するインタラクティブ版。
`optimize_prompts.py`(config内の固定リストを順に処理)の対話版で、文法(--task)には依存しない
(City/UR/ArcLinesどれでも---taskの指定次第で動く)。

使い方:
    # 対話的に入力(Ctrl-Dで終了)
    uv run python scripts/optimize_prompt_stdin.py ---task configs/tasks/city.yaml --save_path output/interactive

    # ファイルから流し込む(1行1プロンプト)
    cat prompts.txt | uv run python scripts/optimize_prompt_stdin.py ---task configs/tasks/city.yaml --save_path output/interactive

各プロンプトの出力フォルダに last.png / video.mp4 / topshape.objc 等に加えて、
実際に使われた完全なプロンプト文字列(prompt_suffix込み)を prompt.txt として保存する。
"""

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional
import sys
import gc
import shutil
import numpy as np
import torch
from rich.pretty import pprint
from rich.traceback import install
from confify import read_config_from_cli, config_dump_yaml

install()  # Enable rich traceback

from d4descent.tasks._base import TaskArgs, RenderArgs
from d4descent.losses.sds import SDSLossArgs
from d4descent.util import save_rgb8, save_video, register_slurm_signal_handlers_auto
from d4descent.optimizer import OptimizeArgs, optimize, OnVisualizeFunc


@dataclass
class Args:
    task: TaskArgs
    loss: SDSLossArgs
    prompt_suffix: Optional[str] = None
    render: RenderArgs = field(default_factory=RenderArgs)
    save_path: Optional[Path] = None
    optim: OptimizeArgs = field(default_factory=OptimizeArgs)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    restart: bool = False


register_slurm_signal_handlers_auto()


def read_prompts_from_stdin():
    """標準入力から1行1プロンプトで読み続けるジェネレータ。空行は無視、EOF(Ctrl-D)で終了する。"""
    interactive = sys.stdin.isatty()
    i = 0
    while True:
        try:
            line = input(f"[{i}] prompt> " if interactive else "")
        except EOFError:
            return
        line = line.strip()
        if not line:
            continue
        yield i, line
        i += 1


def main():
    args = read_config_from_cli(Args)
    pprint(args)
    if args.save_path is not None:
        save_path = Path(args.save_path)
        save_path.mkdir(exist_ok=True, parents=True)
        config_dump_yaml(args, save_path / "config.yaml")

    for i, raw_prompt in read_prompts_from_stdin():
        prompt = raw_prompt if args.prompt_suffix is None else f"{raw_prompt}, {args.prompt_suffix}"
        name = f"{i:02d}_{raw_prompt.replace(',', '').replace(' ', '-')[:50]}"
        print(f"==== Processing {name} ====")
        print(f"    prompt: {prompt!r}")
        save_path = Path(args.save_path) / name if args.save_path is not None else None
        on_visualize: Optional[OnVisualizeFunc] = None

        if save_path is not None:
            if not args.restart and (save_path / "video.mp4").exists():
                print("--> Skipping (already exists)")
                continue
            if args.restart and save_path.exists():
                shutil.rmtree(save_path)
            save_path.mkdir(exist_ok=True, parents=True)
            # 出力結果だけを見た時にどのプロンプトで生成したか分かるよう、そのまま保存しておく
            (save_path / "prompt.txt").write_text(prompt + "\n")

            def on_visualize_(img: np.ndarray, step: int, loss: float):
                assert save_path is not None
                all_imgs.append(img)
                save_rgb8(save_path / "last.png", img)

            on_visualize = on_visualize_

        cur_loss = replace(args.loss, prompt=prompt)
        task = args.task.create(args.render, cur_loss, args.device, None)

        try:
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
                        print(f"--> Saved to {save_path}")
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
        finally:
            # 次のプロンプトに移る前にSD一式(数GB)を確実に解放する。対話ループでプロンプトを
            # 何本も続けて処理するため、ここで解放しないとVRAMが積み上がっていく恐れがある。
            del task
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

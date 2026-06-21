"""Utility functions"""

import torch
from typing import Tuple, Type, TypeVar, cast, Union, Callable, ParamSpec, Optional
import yaml
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import skvideo.io
import torch.nn as nn
import random
import os
import logging
import re
import signal
from subprocess import call
from collections import deque
import logging
from torch.serialization import MAP_LOCATION


# Logging, updated from sd-scripts /library/utils.py
def setup_logging(args=None, log_level=None, reset=False):
    if logging.root.handlers:
        if reset:
            # remove all handlers
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
        else:
            return

    # ! log level is set as env variable, default is INFO
    _log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, _log_level)

    msg_init = None
    if args is not None and args.console_log_file:
        handler = logging.FileHandler(args.console_log_file, mode="w")
    else:
        handler = None
        if not args or not args.console_log_simple:
            try:
                from rich.logging import RichHandler
                from rich.console import Console
                from rich.logging import RichHandler

                handler = RichHandler(console=Console(stderr=True))
            except ImportError:
                msg_init = "rich is not installed, using basic logging"

        if handler is None:
            handler = logging.StreamHandler(sys.stdout)  # same as print
            handler.propagate = False  # type: ignore

    formatter = logging.Formatter(
        fmt="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logging.root.setLevel(log_level)
    logging.root.addHandler(handler)

    if msg_init is not None:
        logger = logging.getLogger(__name__)
        logger.info(msg_init)


def seed_everything(seed):  # Set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.xpu.is_available():
        torch.xpu.manual_seed(seed)
    random.seed(seed)


def get_default_device() -> str:
    """Selects the best available accelerator: CUDA, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


_T = TypeVar("_T")


def torch_load(path: Union[str, Path], type: Type[_T], map_location: MAP_LOCATION = None) -> _T:
    return cast(_T, torch.load(path, weights_only=False, map_location=map_location))


def yaml_load(path: Union[str, Path], type: Type[_T]) -> _T:
    path = Path(path)
    return cast(_T, yaml.unsafe_load(path.open("r")))


def save_rgb8(path: Union[str, Path], image: np.ndarray) -> None:
    """
    image: (H, W, 3) uint8 array
    """
    Image.fromarray(image, mode="RGB").save(path)


def save_video(path: Union[str, Path], images: list[np.ndarray], fps: int = 15) -> None:
    """
    images: (N, H, W, 3) uint8 array
    """
    skvideo.io.vwrite(
        path,
        np.stack(images, axis=0),
        inputdict={"-r": f"{fps}"},
        outputdict={"-vcodec": "libx264", "-pix_fmt": "yuv420p"},
    )


def read_points_npz(path: Union[str, Path], device: Union[torch.device, str, None] = None) -> torch.Tensor:
    """
    path: npz file containing a dict of prim_name -> (N, 2) float32 np array
    returns: (N, 2) float32 array
    """
    path = Path(path)
    data = np.load(path)
    data = np.concat(list(data.values()))
    res = torch.as_tensor(data, dtype=torch.float32, device=device)
    mn = res.min(dim=0).values
    mx = res.max(dim=0).values
    res = (res - (mn + mx) / 2) / ((mx - mn).max() / 2)  # normalize to (-1, 1)
    return res


def torch_interp(x: torch.Tensor, xp: torch.Tensor, yp: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Same functionality of `numpy.interp`.

    One-dimensional (innermost dimension) linear interpolation for monotonically increasing sample points.

    Returns the one-dimensional (innermost dimension) piecewise linear interpolant to a function with given discrete data points (xp, fp), evaluated at x.

    Parameters:
    - x: (..., out). The x-coordinates at which to evaluate the interpolated values.
    - xp: (..., in). The x-coordinates of the data points. Must be increasing (no checks are performed).
    - yp: (..., in, ...1). The y-coordinates of the data points.
    - dim: dimension of the `in` in `yp`. Default: -1 (i.e., ...1 is empty)

    Returns:
    - y: (..., out, ...1). The interpolated values.
    """
    dim = dim % yp.ndim
    n_ydim = yp.ndim - dim - 1
    _ = torch.broadcast_shapes(x.shape[:-1], xp.shape[:-1])
    out = x.shape[-1]
    x = x.expand(*_, out)
    xp = xp.expand(*_, xp.shape[-1])
    unsqueezed_shapes = (*_, out, *(n_ydim * (1,)))
    expanded_shapes = (*_, out, *yp.shape[dim + 1 :])

    ind = torch.searchsorted(xp.contiguous(), x.contiguous(), right=True)  # (..., out)
    ind = torch.clamp(ind, min=1, max=xp.shape[-1] - 1)  # (..., out)
    prev = ind - 1  # (..., out)
    x0 = xp.gather(-1, prev)  # (..., out)
    x1 = xp.gather(-1, ind)  # (..., out)
    ind = ind.reshape(unsqueezed_shapes).expand(*expanded_shapes)  # (..., out, ...1)
    prev = prev.reshape(unsqueezed_shapes).expand(*expanded_shapes)  # (..., out, ...1)
    y0 = yp.gather(dim, prev)  # (..., out, ...1)
    y1 = yp.gather(dim, ind)  # (..., out, ...1)
    t = ((x - x0) / (x1 - x0 + 1e-12)).clamp(min=0, max=1).reshape(unsqueezed_shapes)  # (..., out, ...1)
    return y0 + t * (y1 - y0)


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _returns_nn_module_call(*args):
    return nn.Module.__call__


def patch_call(src_func: Callable[_P, _R]) -> Callable[..., Callable[_P, _R]]:
    return _returns_nn_module_call  # type: ignore


def slurm_sigusr_handler(signum, frame):
    logging.info(f"SIGUSR1 {signum} received")
    # find job id
    array_job_id = os.getenv("SLURM_ARRAY_JOB_ID")
    if array_job_id is not None:
        array_task_id = os.environ["SLURM_ARRAY_TASK_ID"]
        job_id = f"{array_job_id}_{array_task_id}"
    else:
        job_id = os.getenv("SLURM_JOB_ID")

    if job_id is None:
        logging.warning("No job ID found in environment variables")
        return

    assert re.match("[0-9_-]+", job_id)
    cmd = ["scontrol", "requeue", job_id]

    # requeue job
    logging.info(f"requeing job {job_id}...")
    try:
        result = call(cmd)
    except FileNotFoundError:
        # This can occur if a subprocess call to `scontrol` is run outside a shell context
        # Re-attempt call (now with shell context). If any error is raised, propagate to user.
        # When running a shell command, it should be passed as a single string.
        result = call(" ".join(cmd), shell=True)

    # print result text
    if result == 0:
        logging.info(f"Requeued SLURM job: {job_id}")
    else:
        logging.warning(f"Requeuing SLURM job {job_id} failed with error code {result}")


def slurm_sigterm_handler(signum, frame):
    logging.info(f"Bypassing SIGTERM: {signum}")


def register_slurm_signal_handlers_auto():
    # find job id
    array_job_id = os.getenv("SLURM_ARRAY_JOB_ID")
    if array_job_id is not None:
        array_task_id = os.environ["SLURM_ARRAY_TASK_ID"]
        job_id = f"{array_job_id}_{array_task_id}"
    else:
        job_id = os.getenv("SLURM_JOB_ID")

    if job_id is not None:
        logging.info("Registering SLURM signal handlers")
        signal.signal(signal.SIGUSR1, slurm_sigusr_handler)
        signal.signal(signal.SIGTERM, slurm_sigterm_handler)


class MovingAverage:
    def __init__(self, window_size: int):
        self.sum = 0.0
        self.values: deque[float] = deque(maxlen=window_size)

    def clear(self) -> None:
        self.sum = 0.0
        self.values.clear()

    def add(self, value: float) -> None:
        self.sum += value
        if len(self.values) == self.values.maxlen:
            self.sum -= self.values[0]
            self.values.popleft()
        self.values.append(value)

    def mean(self) -> float:
        return self.sum / len(self.values) if len(self.values) > 0 else 0.0


class SetGradient(torch.autograd.Function):
    @staticmethod
    def forward(input: torch.Tensor, target_grad: torch.Tensor, output: torch.Tensor):
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, target_grad, _ = inputs
        ctx.save_for_backward(target_grad.view_as(target_grad), output.view_as(output))

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor):
        (grad_output,) = grad_outputs
        target_grad, output = ctx.saved_tensors
        return (
            target_grad * grad_output.reshape(*output.shape, *(1,) * (len(target_grad.shape) - len(output.shape))),
            None,
            None,
        )


def set_grad(input: torch.Tensor, target_grad: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    """
    input: (...1, ...2)
    target_grad: (...1, ...2)
    output: (...1,)
    """
    return SetGradient.apply(input, target_grad, output)  # type: ignore[no-any-return]


def safe_stack(
    objs: Union[list[torch.Tensor], tuple[torch.Tensor, ...]],
    other_dim: tuple[int, ...],
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Stack tensors across dim=0. If list is empty, return empty tensor with (0, *other_dim) with device
    """
    if len(objs) == 0:
        return torch.empty((0, *other_dim), device=device, dtype=dtype)
    return torch.stack(objs, dim=0)


def safe_cat(
    objs: Union[list[torch.Tensor], tuple[torch.Tensor, ...]],
    other_dim: tuple[int, ...],
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Cat tensors across dim=0. If list is empty, return empty tensor with (0, *other_dim) with device
    """
    if len(objs) == 0:
        return torch.empty((0, *other_dim), device=device, dtype=dtype)
    return torch.cat(objs, dim=0)


def safe_tensor(
    obj,
    zero_shape: tuple[int, ...],
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    res = torch.tensor(obj, device=device, dtype=dtype)
    if res.numel() == 0:
        res = res.reshape(zero_shape)
    return res


def maybe_detach(x: torch.Tensor, detach: bool) -> torch.Tensor:
    if detach:
        return x.detach().clone()
    return x


def maybe_clamp(x: torch.Tensor, min: Optional[float] = None, max: Optional[float] = None) -> torch.Tensor:
    if min is None and max is None:
        return x
    return x.clamp(min=min, max=max)

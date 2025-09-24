import torch
from torch.optim import LBFGS
from typing import Generic, TypeVar, Sequence, Self, Optional, Union, Any, Mapping
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass, field
import random
import time
import math

from geocad.types import Device, Box
from geocad.object_collection import ObjectCollection
from geocad.losses._base import LossArgs


ObjectT = TypeVar("ObjectT")
RewriteSpecT = TypeVar("RewriteSpecT")
StateT = TypeVar("StateT")
ExtraMetrics = dict[str, tuple[float, ...]]


@dataclass
class SatisfyConstraintsArgs:
    lr: float = 0.0002
    steps: int = 5
    debug: bool = True


@dataclass
class RenderArgs:
    size: int = 256
    lim: tuple[float, float] = (-1.5, 1.5)
    center_pixel: bool = True
    blur: float = 1 / math.sqrt(2)


@dataclass
class TaskArgs(ABC):
    @abstractmethod
    def create(
        self,
        render_args: RenderArgs,
        loss_args: LossArgs,
        device: Union[torch.device, str],
        target_img: Optional[torch.Tensor] = None,
    ) -> "Task": ...


class Task(ABC, Generic[ObjectT, RewriteSpecT, StateT]):
    render_args: RenderArgs
    start_time: float

    def __init__(self, render_args: RenderArgs):
        self.render_args = render_args
        self.start_time = time.time()

    @abstractmethod
    def device(self) -> torch.device:
        """Returns the device."""
        ...

    @abstractmethod
    def get_collection_constructor(self) -> type[ObjectCollection[ObjectT]]:
        """Returns the collection constructor."""
        ...

    @abstractmethod
    def initialize_object(self) -> ObjectT:
        """Initialize an object."""
        ...

    @abstractmethod
    def _compute_losses(
        self, collection: ObjectCollection[ObjectT], state: StateT
    ) -> tuple[torch.Tensor, ExtraMetrics]:
        """
        Compute the losses for continuous optimization.

        - collection: [n_shapes,]

        Returns:
        - losses: (n_shapes,)
        - extra_metrics: key -> (n_shapes,)
        """
        ...

    def compute_losses(self, collection: ObjectCollection[ObjectT], state: StateT) -> tuple[torch.Tensor, ExtraMetrics]:
        """
        Compute the losses for continuous optimization.

        - collection: [n_shapes,]

        Returns: (n_shapes,)
        """
        return self._compute_losses(collection, state)

    @abstractmethod
    def compute_simplicity(self, collection: ObjectCollection[ObjectT]) -> list[float]:
        """
        Compute the simplicity metrics for discrete optimization

        Returns: (n,)
        """
        ...

    def make_proposals_ex(
        self, obj: ObjectT, num_proposals: int
    ) -> tuple[ObjectCollection[ObjectT], list[RewriteSpecT]]:
        proposals, specs = self.make_proposals(obj)
        if num_proposals > 0:
            sel_ids = random.sample(range(len(proposals)), k=min(num_proposals, len(proposals)))
            proposals = self.get_collection_constructor().from_objects([proposals[i] for i in sel_ids])
            specs = [specs[i] for i in sel_ids]
        return proposals, specs

    @abstractmethod
    def make_proposals(self, obj: ObjectT) -> tuple[ObjectCollection[ObjectT], list[RewriteSpecT]]:
        """Make proposals for the given object."""
        ...

    @abstractmethod
    def combine_proposals(
        self,
        base: ObjectT,
        proposals: ObjectCollection[ObjectT],
        base_loss: float,
        proposal_losses: list[float],
        proposal_specs: list[RewriteSpecT],
        accept_parallel: bool = True,
    ) -> tuple[ObjectT, bool]:
        """Combine proposals into a single object.

        Returns:
        - new object
        - whether the object was changed
        """
        ...

    @abstractmethod
    def initialize_state(self) -> StateT:
        """Returns kwargs for compute_losses"""

    def update_state_for_proposals(self, state: StateT, proposals: ObjectCollection[ObjectT]) -> StateT:
        """Returns old state"""
        return state

    def step_state(self, state: StateT) -> StateT:
        """Returns new state"""
        return state

    @abstractmethod
    def cleanup(self, collection: ObjectCollection[ObjectT]) -> ObjectCollection[ObjectT]:
        """Cleanup the collection"""
        ...

    @abstractmethod
    def visualize(self, collection: ObjectCollection[ObjectT], step: int, loss: float, state: StateT) -> np.ndarray:
        """Called after each optimization step"""
        ...

    def get_elapsed_time(self) -> float:
        """Returns the elapsed time in seconds"""
        return time.time() - self.start_time


def update_extra_metrics(input_: ExtraMetrics, other_: ExtraMetrics) -> ExtraMetrics:
    """
    Append other_ to input_. Creates a new dict.
    """
    if input_ == {}:
        return {**other_}
    input_keys = set(input_.keys())
    other_keys = set(other_.keys())
    assert input_keys == other_keys, f"input_keys != other_keys, got {input_keys} != {other_keys}"
    res: ExtraMetrics = {}
    input_keys_ = list(input_keys)
    if len(input_keys_) == 0:
        return res
    input_len = len(input_[input_keys_[0]])
    output_len = len(other_[input_keys_[0]])
    for key in input_keys:
        input__ = input_[key]
        other__ = other_[key]
        assert len(input__) == input_len, f"input_len != {input_len}, got {len(input__)} != {input_len}"
        assert len(other__) == output_len, f"output_len != {output_len}, got {len(other__)} != {output_len}"
        res[key] = input__ + other__
    return res

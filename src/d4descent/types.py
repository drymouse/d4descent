import torch
from typing import Union, Generic, TypeVar
from abc import ABC, abstractmethod

Vec2Like = Union[torch.Tensor, tuple[float, float]]
ScalarLike = Union[torch.Tensor, float]
Device = Union[str, torch.device, None]

_T = TypeVar("_T")


class Box(Generic[_T]):
    def __init__(self, value: _T):
        self.value = value

    def get(self) -> _T:
        return self.value

    def set(self, value: _T) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"Box({self.value})"

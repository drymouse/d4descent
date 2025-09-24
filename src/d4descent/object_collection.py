import torch
import torch.jit
from typing import TypeVar, Sequence, Self, Generic, TypeVar, Sequence, Self, Generic, overload, Optional, Union, cast
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial

from .util import safe_cat

ObjectT = TypeVar("ObjectT")


class Renderable(ABC):
    @abstractmethod
    def device(self) -> torch.device: ...

    @abstractmethod
    def rasterize(self, positions: torch.Tensor) -> torch.Tensor:
        """
        positions: (...A, 2)

        returns: (...B, ...A)
        """
        ...

    def render(
        self,
        size: int,
        lim: tuple[float, float] = (-1.5, 1.5),
        center_pixel: bool = True,
    ) -> torch.Tensor:
        """
        Render as a signed distance field.
        returns:
        - img: (...B, size, size)
        """
        device = self.device()
        lim0, lim1 = lim

        if center_pixel:
            basis = (torch.arange(size, device=device) + 0.5) / size * (lim1 - lim0) + lim0  # (size,)
        else:
            basis = torch.linspace(lim0, lim1, size, device=device)  # (size,)
        xs = basis.expand(size, -1)  # (size, size)
        ys = basis.unsqueeze(-1).expand(-1, size)  # (size, size)
        grid = torch.stack([xs, ys], dim=-1)  # (size, size, 2)
        res = self.rasterize(grid)
        return res

    def render01(
        self, size: int, lim: tuple[float, float] = (-1.5, 1.5), center_pixel: bool = True, blur: float = 1.0
    ) -> torch.Tensor:
        """Render the collection as a signed distance field.

        Returns:
        - img: (...B, size, size)
        """
        vlim = blur * (lim[1] - lim[0]) / size
        imgs = self.render(size, lim, center_pixel=center_pixel)  # (n_shapes, size, size)
        imgs = (-imgs.clamp(-vlim, vlim) + vlim) / (2 * vlim)  # (n_shapes, size, size)
        return imgs


class ObjectCollection(Renderable, Sequence[ObjectT]):
    @classmethod
    @abstractmethod
    def from_object(cls, object: ObjectT, **kwargs) -> Self:
        """Create a collection from an object."""
        ...

    @classmethod
    def from_objects(cls, objects: Sequence[ObjectT], **kwargs) -> Self:
        """Create a collection from a list of objects."""
        return cls.cat([cls.from_object(x, **kwargs) for x in objects])

    @classmethod
    @abstractmethod
    def cat(cls, collections: list[Self], **kwargs) -> Self:
        """Concatenate collections."""
        ...

    @abstractmethod
    def get_object(self, idx: int, detach: bool = True) -> ObjectT: ...

    @abstractmethod
    def parameters(self) -> list[torch.Tensor]:
        """Returns a list of parameters."""
        ...

    def scale_grads_(self) -> bool:
        """Scale gradients in-place. Returns True if any scaling was performed."""
        return False

    @abstractmethod
    def parameter_names(self) -> list[str]:
        """Returns a list of parameter names."""
        ...

    @abstractmethod
    def per_object_grads(self) -> list[torch.Tensor]:
        """Returns a list of grads for each object."""
        ...

    @abstractmethod
    def requires_grad_(self, requires_grad: bool = True) -> Self:
        """Set requires_grad."""
        ...

    @abstractmethod
    def clone(self) -> Self: ...

    @abstractmethod
    def to(self, device: Union[str, torch.device, None] = None) -> Self: ...

    def batchify(
        self, param_count: Optional[int], batch_size: Optional[int] = None, requires_grad: bool = True
    ) -> list[Self]:
        """Batchify the collection
        If batch_size is set, then the batch size is `batch_size`.
        If param_count is set, then the collection is batched so that each collection has at most `param_count` parameters.
        """
        Collection = type(self)
        proposal_shcs: list[Self] = []
        if batch_size is not None:
            for i in range(0, len(self), batch_size):
                proposal_shcs.append(self[i : i + batch_size].requires_grad_(requires_grad=requires_grad))
        elif param_count is not None:
            stk__: list[Self] = []
            stk_param_count__: int = 0
            for shape_ in self:
                shc__ = Collection.from_object(shape_)
                cnt__ = sum(x.numel() for x in shc__.parameters())
                if stk_param_count__ + cnt__ > param_count:
                    proposal_shcs.append(Collection.cat(stk__).requires_grad_(requires_grad=requires_grad))
                    stk__ = []
                    stk_param_count__ = 0
                stk__.append(shc__)
                stk_param_count__ += cnt__

            if len(stk__) > 0:
                proposal_shcs.append(Collection.cat(stk__).requires_grad_(requires_grad=requires_grad))
        else:
            raise ValueError("Either batch_size or param_count must be set.")
        return proposal_shcs

    @overload
    def __getitem__(self, idx: int) -> ObjectT: ...
    @overload
    def __getitem__(self, idx: slice) -> Self: ...

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.get_object(idx, True)
        elif isinstance(idx, slice):
            return self.__class__.from_objects([self.get_object(i, True) for i in range(*idx.indices(len(self)))])
        raise TypeError(f"Invalid index type: {type(idx)}")

    def project_to_valid_(self) -> Self:
        """Projects the collection to the valid set"""
        return self  # default to no-op

    def to_savable(self) -> Self:
        return self


class StdObject(Renderable):
    """
    A standard object that can be rendered. render() returns (1, size, size)
    """

    @abstractmethod
    def parameters(self) -> list[torch.Tensor]:
        """Returns a list of parameters."""
        ...

    def scale_grads_(self) -> bool:
        """Scale gradients in-place. Returns True if any scaling was performed."""
        return False

    @abstractmethod
    def parameter_names(self) -> list[str]:
        """Returns a list of parameter names."""
        ...

    @abstractmethod
    def set_parameters_(self, parameters: list[torch.Tensor]):
        """Set parameters."""
        ...

    def requires_grad_(self, requires_grad: bool = True) -> Self:
        """Set requires_grad."""
        self.set_parameters_([p.detach().clone().requires_grad_(requires_grad) for p in self.parameters()])
        return self

    @abstractmethod
    def clone(self) -> Self: ...

    @abstractmethod
    def device(self) -> torch.device: ...

    @abstractmethod
    def to(self, device: Union[str, torch.device, None] = None) -> Self: ...


StdObjectT = TypeVar("StdObjectT", bound=StdObject)


@dataclass
class StdCollection(ObjectCollection[StdObjectT]):
    """
    A naive implementation of ObjectCollection for StdObject.
    """

    # params in objects and params here always refer to the same tensors
    objects: list[StdObjectT]
    params: list[list[torch.Tensor]]
    param_names: list[list[str]]

    _rasterize_fns: Optional[list[torch.jit.ScriptFunction]] = field(init=False, default=None)

    def __len__(self) -> int:
        return len(self.objects)

    def get_object(self, idx: int, detach: bool = True) -> StdObjectT:
        assert detach == True, "detach must be True"
        return self.objects[idx].clone()

    def parameters(self) -> list[torch.Tensor]:
        """Returns a list of parameters."""
        return [y for x in self.params for y in x]

    def scale_grads_(self) -> bool:
        """Scale gradients in-place. Returns True if any scaling was performed."""
        res = [obj.scale_grads_() for obj in self.objects]
        return any(res)

    def parameter_names(self) -> list[str]:
        """Returns a list of parameter names."""
        return [y for x in self.param_names for y in x]

    def per_object_grads(self) -> list[torch.Tensor]:
        """Returns a list of grads for each object."""
        device = self.device()
        return [
            safe_cat([x.grad.flatten() for x in obj.parameters() if x.grad is not None], (), device=device)
            for obj in self.objects
        ]

    @classmethod
    def from_object(cls, object: StdObjectT, **kwargs) -> Self:
        """Create a collection from an object."""
        object = object.clone()
        return cls(
            objects=[object],
            params=[object.parameters()],
            param_names=[object.parameter_names()],
        )

    @classmethod
    def cat(cls, collections: list["StdCollection[StdObjectT]"], **kwargs) -> Self:  # type: ignore[override]
        """Concatenate collections."""
        objects = [o.clone() for x in collections for o in x.objects]
        params = [obj.parameters() for obj in objects]
        param_names = sum([x.param_names for x in collections], [])
        return cls(
            objects=objects,
            params=params,
            param_names=param_names,
        )

    def requires_grad_(self, requires_grad: bool = True) -> Self:
        """Set requires_grad."""
        for obj, params in zip(self.objects, self.params):
            new_params = [param.detach().clone().requires_grad_(requires_grad) for param in params]
            obj.set_parameters_(new_params)
        return self

    def device(self) -> torch.device:
        return self.objects[0].device()

    def to(self, device: Union[str, torch.device, None] = None) -> Self:
        return self.__class__(
            objects=[obj.to(device=device) for obj in self.objects],
            params=[[p.to(device=device) for p in params] for params in self.params],
            param_names=list(self.param_names),
        )

    def clone(self) -> Self:
        return self.__class__(
            objects=[obj.clone() for obj in self.objects],
            params=[[p.detach().clone() for p in params] for params in self.params],
            param_names=list(self.param_names),
        )

    def rasterize(self, positions: torch.Tensor) -> torch.Tensor:
        """
        positions: (...A, 2)
        obj.rasterize: (...A, 2) -> (...B, ...A)
        returns: (n_objects, ...B, ...A)
        """
        if self._rasterize_fns is None:
            rasterize_fns: list[torch.jit.ScriptFunction] = []
            for obj in self.objects:

                def wrap(*args: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
                    obj.set_parameters_(list(args))
                    return obj.rasterize(positions)

                traced = torch.jit.trace(wrap, example_inputs=(*obj.parameters(), positions))
                assert isinstance(traced, torch.jit.ScriptFunction)
                rasterize_fns.append(traced)
            self._render_funcs = rasterize_fns

        promises = [torch.jit.fork(func, *params, positions) for params, func in zip(self.params, self._render_funcs)]
        results: list[torch.Tensor] = [promise.wait() for promise in promises]
        return torch.stack(results, dim=0)  # (n_shapes, ...B, ...A)

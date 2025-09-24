from contextlib import contextmanager
from typing import TypeVar, Generic


ObjectT = TypeVar("ObjectT")


class Context(Generic[ObjectT]):
    _id: int
    _objects: dict[int, ObjectT]
    _save_objects: bool

    def __init__(self, cur_id: int = 0, save_objects: bool = False) -> None:
        self._id = cur_id
        self._objects = {}
        self._save_objects = save_objects

    def gen_id(self) -> int:
        res = self._id
        self._id += 1
        return res

    def register(self, id: int, obj: ObjectT):
        if self._save_objects:
            self._objects[id] = obj

    def get_object(self, id: int) -> ObjectT:
        return self._objects[id]

    @classmethod
    def get(cls) -> "Context":
        return _global_context


_global_context = Context()


@contextmanager
def use_context(context: Context):
    global _global_context
    old_context = _global_context
    _global_context = context
    yield
    _global_context = old_context

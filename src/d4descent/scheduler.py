from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
from typing import Any, Dict, Literal, SupportsFloat


class AdaptiveLRScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        factor: float,
        reduce_patience: int,
        increase_patience: int,
        min_lr: float,
        max_lr: float,
        threshold: float = 1e-4,
        threshold_mode: Literal["rel", "abs"] = "rel",
        eps: float = 1e-8,
    ):
        if factor <= 0.0 or factor >= 1.0:
            raise ValueError(f"factor must be in (0, 1), got {factor}")
        self.factor = factor
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.reduce_patience = reduce_patience
        self.increase_patience = increase_patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.eps = eps
        self.best = float("inf")
        self.num_bad_epochs = 0
        self.num_good_epochs = 0
        self.optimizer = optimizer
        self._last_lr = [group["lr"] for group in optimizer.param_groups]

    def step(self, metrics: SupportsFloat):  # type: ignore[override]
        current = float(metrics)

        if not self.is_worse(current, self.best):
            self.num_good_epochs += 1
        else:
            self.num_good_epochs = 0

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.reduce_patience:
            self._adjust_lr(self.factor)
            self.num_bad_epochs = 0
            self.num_good_epochs = 0

        if self.num_good_epochs > self.increase_patience:
            self._adjust_lr((1 / self.factor) ** (1 / 2))
            self.num_bad_epochs = 0
            self.num_good_epochs = 0

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def _adjust_lr(self, factor: float):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = min(max(old_lr * factor, self.min_lr), self.max_lr)
            if abs(old_lr - new_lr) > self.eps:
                param_group["lr"] = new_lr

    def is_better(self, a: float, best: float) -> bool:
        if self.threshold_mode == "rel":
            rel_epsilon = 1.0 - self.threshold
            return a < best * rel_epsilon
        elif self.threshold_mode == "abs":
            return a < best - self.threshold
        else:
            raise ValueError(f"Unknown threshold_mode: {self.threshold_mode}")

    def is_worse(self, a: float, best: float) -> bool:
        if self.threshold_mode == "rel":
            rel_epsilon = self.threshold + 1.0
            return a > best * rel_epsilon
        elif self.threshold_mode == "abs":
            return a > best + self.threshold
        else:
            raise ValueError(f"Unknown threshold_mode: {self.threshold_mode}")

    def state_dict(self):
        raise NotImplementedError()

    def load_state_dict(self, state_dict: Dict[str, Any]):
        raise NotImplementedError()

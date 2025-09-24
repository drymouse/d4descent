import torch
from torch.optim.optimizer import Optimizer
from torch.optim import LBFGS
from torch.optim.lr_scheduler import ReduceLROnPlateau, LRScheduler, LinearLR, ExponentialLR
from dataclasses import dataclass, field
from typing import Literal, Optional, Callable, Protocol, Generic
from tqdm.auto import tqdm
import numpy as np
import random
import os

from .context import Context, use_context
from .object_collection import ObjectCollection
from .tasks._base import (
    Task,
    ObjectT,
    RewriteSpecT,
    StateT,
    SatisfyConstraintsArgs,
    ExtraMetrics,
    update_extra_metrics,
)
from .util import MovingAverage, maybe_clamp
from .scheduler import AdaptiveLRScheduler


@dataclass
class OptimizeArgs:
    n_steps: int = 4000
    # continuous optimization
    optimizer: Literal["Adam", "SGD"] = "SGD"
    scheduler: Literal["none", "ReduceLROnPlateau", "AdaptiveLR", "LinearLR", "ExponentialLR"] = "ReduceLROnPlateau"
    lr: float = 0.5
    clip_grad: Optional[float] = 2.0
    clip_grad_mode: Literal["abs", "rel"] = "abs"
    reduce_lr_factor: float = 0.5
    reduce_lr_patience: int = 2
    reduce_lr_min_lr: float = 1e-4
    increase_lr_patience: int = 2
    reset_lr_after_proposal: bool = False
    increase_lr_after_proposal: bool = True
    # cleanup
    cleanup_every: int = 10
    # proposal
    proposal_trigger: Literal["step", "rel_loss"] = "step"
    propose_every: int = 50
    proposal_rel_loss: float = 5e-3
    proposal_patience: int = 10
    proposal_criterion: Literal["loss", "grad", "grad_only"] = "loss"
    proposal_steps: int = 2  # not used if proposal_criterion == "grad"
    proposal_size: int = 0  # the number of proposals to evaluate (uniformly sampled). 0 means all
    proposal_clip_grad: bool = True
    proposal_accept_parallel: bool = True
    batch_param_count: int = 8192
    batch_size: Optional[int] = None  # batch_param_count is ignored if batch_size is set
    # simplicity
    w_simplicity: float = 1.0
    # visualization
    visualize_every: int = 10
    # satisfy constraints
    enable_satisfy_constraints: bool = False
    satisfy_constraints_args: SatisfyConstraintsArgs = field(default_factory=SatisfyConstraintsArgs)
    # early stopping
    stopping_eps: float = 5e-3
    stopping_patience: Optional[int] = None  # number of rewrites without improvement
    # fix bug
    fix_bug: bool = False


class OnVisualizeFunc(Protocol):
    def __call__(self, img: np.ndarray, step: int, loss: float) -> None: ...


def optimize(
    task: Task[ObjectT, RewriteSpecT, StateT],
    args: OptimizeArgs,
    on_visualize: Optional[OnVisualizeFunc] = None,
    debug: Optional[bool] = None,
    **tqdm_kwargs,
) -> tuple[ObjectT, float, ObjectCollection[ObjectT], ExtraMetrics]:
    """
    Returns:
    - best_object: the best object
    - best_loss: the best loss
    - all_objects: a collection of objects. (n_steps,)
    - all_metrics: a dictionary of metrics. each entry has length n_steps
        - $loss: loss with simplicity
        - $loss_ma: loss after moving average
        - $loss_cont: loss without simplicity
        - $loss_simp: simplicity
        - $lr: learning rate
        - $timestamp: timestamp
        - extra metrics from task.compute_losses
    """
    if debug is None:
        debug = os.environ.get("GEOCAD_DEBUG", "0").strip().lower() in ["1", "true"]
    ctx = Context(save_objects=True)
    with use_context(ctx):
        Collection = task.get_collection_constructor()
        population: ObjectCollection[ObjectT] = Collection.from_object(task.initialize_object())
        state = task.initialize_state()

        def setup_optimizer(
            parameters: list[torch.Tensor],
            lr: Optional[float] = None,
        ) -> tuple[Optimizer, Optional[LRScheduler]]:
            lr = args.lr if lr is None else lr
            if args.optimizer == "Adam":
                optimizer = torch.optim.Adam(parameters, lr=lr)
            elif args.optimizer == "SGD":
                optimizer = torch.optim.SGD(parameters, lr=lr)
            else:
                raise ValueError(f"unknown optimizer {args.optimizer}")

            scheduler = None
            if args.scheduler == "ReduceLROnPlateau":
                scheduler = ReduceLROnPlateau(
                    optimizer,
                    factor=args.reduce_lr_factor,
                    patience=args.reduce_lr_patience,
                    min_lr=args.reduce_lr_min_lr,
                )
            elif args.scheduler == "AdaptiveLR":
                scheduler = AdaptiveLRScheduler(
                    optimizer,
                    factor=args.reduce_lr_factor,
                    reduce_patience=args.reduce_lr_patience,
                    increase_patience=args.increase_lr_patience,
                    min_lr=args.reduce_lr_min_lr,
                    max_lr=args.lr,
                )
            elif args.scheduler == "LinearLR":
                scheduler = LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor=0.0,
                    total_iters=args.n_steps,
                )
            elif args.scheduler == "ExponentialLR":
                scheduler = ExponentialLR(
                    optimizer,
                    gamma=0.01 ** (1 / args.n_steps),
                )

            return optimizer, scheduler

        optimizer, scheduler = setup_optimizer(population.parameters())
        cur_lr = scheduler.get_last_lr()[0] if scheduler is not None else args.lr
        losses, _ = task.compute_losses(population, state=state)
        loss_since_last_rewrite = float("inf")
        stopping_patience = 0
        patience = 0
        ma_loss = MovingAverage(window_size=8)

        all_ma_losses: list[float] = []
        all_losses: list[float] = []  # loss with simplicity
        all_cont_losses: list[float] = []  # loss without simplicity
        all_simplicities: list[float] = []  # simplicity
        all_extra_metrics: ExtraMetrics = {}
        all_objects: list[ObjectCollection[ObjectT]] = []
        all_lrs: list[float] = []
        all_timestamps: list[float] = []

        for step in (pbar := tqdm(range(args.n_steps), **tqdm_kwargs)):
            if step == 0:
                rewrite = False
            elif args.proposal_trigger == "step":
                rewrite = step % args.propose_every == 0
            elif args.proposal_trigger == "rel_loss":
                rewrite = patience >= args.proposal_patience
                if rewrite:
                    patience = 0
            else:
                raise ValueError(f"Unknown rewrite_trigger: {args.proposal_trigger}")

            if step % args.cleanup_every == 0:
                population = task.cleanup(population).requires_grad_()
                if not rewrite:
                    optimizer, scheduler = setup_optimizer(population.parameters(), lr=cur_lr)
                # else: optimizer is recreated during rewrite

            if rewrite:
                print(f"Rewriting step {step}")

                cur_loss = all_ma_losses[-1]
                if cur_loss > loss_since_last_rewrite * (1 - args.stopping_eps):
                    stopping_patience += 1
                else:
                    stopping_patience = 0
                loss_since_last_rewrite = min(loss_since_last_rewrite, cur_loss)
                ma_loss.clear()

                if args.stopping_patience is not None and stopping_patience >= args.stopping_patience:
                    break

                og = population[0]
                proposals, specs = task.make_proposals_ex(og, num_proposals=args.proposal_size)

                proposals_og = Collection.cat([proposals, population])
                batched_proposals = proposals_og.batchify(
                    param_count=args.batch_param_count, batch_size=args.batch_size
                )

                # print(f"proposal size: {len(proposals)}, batches: {len(batched_proposals)}")
                # print([sum(x.numel() for x in proosal_shc.parameters()) for proosal_shc in batched_proposals])

                if args.proposal_criterion == "loss":
                    for proposal_ in tqdm(batched_proposals):
                        optimizer, scheduler = setup_optimizer(
                            proposal_.parameters(), lr=cur_lr if args.fix_bug else None  # HACK: RIP
                        )
                        proposal_state = task.update_state_for_proposals(state, proposal_)
                        for _ in range(args.proposal_steps):
                            optimizer.zero_grad()
                            losses_, _ = task.compute_losses(proposal_, state=proposal_state)
                            loss = losses_.sum()
                            loss.backward()
                            proposal_.scale_grads_()
                            if args.clip_grad is not None:
                                torch.nn.utils.clip_grad_value_(
                                    proposal_.parameters(),
                                    args.clip_grad if args.clip_grad_mode == "abs" else args.clip_grad / cur_lr,
                                )
                            optimizer.step()
                            proposal_.project_to_valid_()

                    with torch.no_grad():
                        all_losses_1: list[float] = []
                        for proposal_ in batched_proposals:
                            proposal_state = task.update_state_for_proposals(state, proposal_)
                            losses__, _ = task.compute_losses(proposal_, state=proposal_state)
                            all_losses_1.extend(losses__.tolist())
                        simplicity = task.compute_simplicity(proposals_og)
                        all_losses_ = [l_ + s_ * args.w_simplicity for l_, s_ in zip(all_losses_1, simplicity)]
                        losses_ = all_losses_[:-1]
                        og_loss = all_losses_[-1]

                    if debug:
                        og_s = simplicity[-1]
                        og_l = all_losses_1[-1]
                        ls = [
                            (l_ - og_loss, l1_, sp_, s_)
                            for l_, l1_, sp_, s_ in zip(losses_, all_losses_1, specs, simplicity)
                        ]
                        ls.sort(key=lambda x: x[0])
                        for loss_, l_, spec_, s_ in ls:
                            print(f"loss_d: {loss_:.2e}, l: {l_ - og_l:.2e}, s: {s_ - og_s:.2e} spec: {spec_}")
                        print("===")
                elif args.proposal_criterion == "grad" or args.proposal_criterion == "grad_only":
                    all_losses_1: list[float] = []
                    all_grads_: list[float] = []
                    for proposal_ in tqdm(batched_proposals):
                        proposal_state = task.update_state_for_proposals(state, proposal_)
                        for p_ in proposal_.parameters():
                            p_.grad = None
                        losses_, _ = task.compute_losses(proposal_, state=proposal_state)
                        all_losses_1.extend(losses_.tolist())
                        loss = losses_.sum()
                        loss.backward()

                        per_object_grads = proposal_.per_object_grads()
                        if proposal_.scale_grads_():
                            per_object_scaled_grads = proposal_.per_object_grads()
                        else:
                            per_object_scaled_grads = per_object_grads

                        all_grads_.extend(
                            [
                                (
                                    maybe_clamp(
                                        sp_,
                                        min=(
                                            -args.clip_grad
                                            if args.clip_grad is not None and args.proposal_clip_grad
                                            else None
                                        ),
                                        max=args.clip_grad if args.proposal_clip_grad else None,
                                    )
                                    * p_
                                )
                                .sum()
                                .item()
                                for p_, sp_ in zip(per_object_grads, per_object_scaled_grads)
                            ]
                        )
                    simplicity = task.compute_simplicity(proposals_og)
                    loss_w = 0 if args.proposal_criterion == "grad_only" else 1
                    all_losses_ = [
                        loss_w * l_ - cur_lr * g_ + s_ * args.w_simplicity
                        for l_, g_, s_ in zip(all_losses_1, all_grads_, simplicity)
                    ]
                    losses_ = all_losses_[:-1]
                    og_loss = all_losses_[-1]

                    # if isinstance(proposals, ShapeCollection):
                    if debug:
                        og_s = simplicity[-1]
                        og_g = all_grads_[-1]
                        og_l = all_losses_1[-1]
                        ls = [
                            (l_ - og_loss, l1_, sp_, g_, s_)
                            for l_, l1_, sp_, g_, s_ in zip(losses_, all_losses_1, specs, all_grads_, simplicity)
                        ]
                        ls.sort(key=lambda x: x[0])
                        for loss_, l_, spec_, g_, s_ in ls:
                            print(
                                f"loss_d: {loss_:.2e}, l: {l_ - og_l:.2e} g: {g_ - og_g:.2e}, s: {s_ - og_s:.2e} spec: {spec_}"
                            )
                        print("===")
                        print(all_grads_)
                else:
                    raise ValueError(f"Unknown proposal_criterion: {args.proposal_criterion}")

                # for loss_, spec_ in zip(losses_, specs):
                #     print(f"{loss_ - og_loss:.2e}, {spec_}")

                new_shape, changed = task.combine_proposals(
                    og, proposals, og_loss, losses_, specs, accept_parallel=args.proposal_accept_parallel
                )
                population = Collection.from_object(new_shape).requires_grad_()
                # img = task.visualize(population, step, losses.sum().item(), state=state)
                # if on_visualize is not None:
                #     on_visualize(img, step, losses.sum().item())
                # import pdb; pdb.set_trace()
                if args.scheduler == "ReduceLROnPlateau":
                    if args.increase_lr_after_proposal:
                        # increase lr one step
                        cur_lr = min(args.lr, cur_lr / args.reduce_lr_factor)
                    if args.reset_lr_after_proposal:
                        cur_lr = args.lr
                optimizer, scheduler = setup_optimizer(population.parameters(), lr=cur_lr)

            simplicity = task.compute_simplicity(population)
            losses, extra = task.compute_losses(population, state=state)
            loss = losses.sum()
            loss_with_simplicity = [l_ + s_ * args.w_simplicity for l_, s_ in zip(losses.tolist(), simplicity)]

            all_objects.append(population.clone().to("cpu"))
            all_cont_losses.append(loss.item())
            all_simplicities.append(sum(simplicity))
            all_losses.append(sum(loss_with_simplicity))
            all_extra_metrics = update_extra_metrics(all_extra_metrics, extra)
            all_lrs.append(cur_lr)
            all_timestamps.append(task.get_elapsed_time())

            if on_visualize is not None and (step % args.visualize_every == 0 or rewrite):
                img = task.visualize(population, step, losses.sum().item(), state=state)
                on_visualize(img, step, losses.sum().item())

            optimizer.zero_grad()
            loss.backward()
            population.scale_grads_()
            if debug and step % 50 == 0:
                params_ = population.parameters()
                for name_, p_ in zip(population.parameter_names(), params_):
                    if p_.grad is not None:
                        print(f"{name_}: {p_.grad.tolist()}")
                print("====")

            # if step % 5 == 0:
            #     print(population.per_object_grads())
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_value_(
                    population.parameters(),
                    args.clip_grad if args.clip_grad_mode == "abs" else args.clip_grad / cur_lr,
                )
            optimizer.step()
            population.project_to_valid_()
            # if step == 100:
            #     import pdb; pdb.set_trace()

            mean_loss = sum(loss_with_simplicity) / len(loss_with_simplicity)
            ma_loss.add(mean_loss)
            all_ma_losses.append(ma_loss.mean())

            if scheduler is not None:
                if isinstance(scheduler, (ReduceLROnPlateau, AdaptiveLRScheduler)):
                    scheduler.step(loss)
                else:
                    scheduler.step()
                cur_lr = scheduler.get_last_lr()[0]

            state = task.step_state(state)

            if len(all_ma_losses) >= 2:
                if (1 - args.proposal_rel_loss) * all_ma_losses[-2] <= all_ma_losses[-1]:
                    patience += 1
                else:
                    patience = 0

            if len(all_ma_losses) > 2:
                diff = 1 - all_ma_losses[-1] / max(all_ma_losses[-2], 1e-12)
            else:
                diff = 0

            pbar.set_description(f"loss={mean_loss:.3e}, diff={diff:.3e}, lr={cur_lr:.2e}")

        return (
            population[0],
            losses.item(),
            Collection.cat(all_objects),
            {
                "$loss": tuple(all_losses),
                "$loss_ma": tuple(all_ma_losses),
                "$loss_cont": tuple(all_cont_losses),
                "$loss_simp": tuple(all_simplicities),
                "$lr": tuple(all_lrs),
                "$timestamp": tuple(all_timestamps),
                **all_extra_metrics,
            },
        )

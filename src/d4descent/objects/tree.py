from typing import Optional, Self, Union, Type, Literal
from dataclasses import dataclass, field, replace
from enum import Enum
import math
import random
from functools import partialmethod

import torch
from matplotlib.patches import Rectangle, Circle, Arc
from matplotlib.lines import Line2D

from ..context import Context
from ..object_collection import ObjectCollection
from ..util import safe_stack, safe_cat, maybe_detach
from ..visualizer import LineStyle, MPLVisualizerAxes


@dataclass
class TreePayload:
    pass


@dataclass
class TreeRewriteArgs:
    add_branch: bool = True
    add_branch_epsilon: bool = True
    add_both_branch: bool = False
    split_branch: bool = False
    split_add_branch: bool = False
    remove_branch: bool = True
    remove_branch_epsilon: bool = False
    remove_non_leaf: bool = False
    add_anywhere: bool = False
    # --- Misc params ---
    default_angle: float = math.radians(15.0)  # degrees
    default_length: float = 0.2
    eps_length: float = 0.03
    default_r: float = 0.075
    random_angle: bool = False
    add_anywhere_strategy: Literal["nearest", "fewest"] = "fewest"
    add_anywhere_last_r: float = 0.075

    def __post_init__(self):
        pass
        # assert self.add_branch_epsilon == self.remove_branch_epsilon, "add_branch_epsilon and remove_branch_epsilon must be the same"


class TreeRewriteType(Enum):
    SplitBranch = 0
    AddBranch = 1
    AddBranchBoth = 2
    RemoveBranch = 3
    AddAnywhere = 4


ThetaMode = Literal["rel", "abs"]


@dataclass
class TreeCollectionArgs:
    # Selective optimize flags
    optimize_roots: bool = True
    optimize_ls: bool = True
    optimize_thetas: bool = True
    optimize_rs: bool = False
    # clamp
    ls_min: Optional[float] = 0.0
    ls_max: Optional[float] = None
    theta_min: Optional[float] = None
    theta_max: Optional[float] = None
    rs_min: Optional[float] = 0.0
    rs_max: Optional[float] = 1.0
    #
    theta_mode: ThetaMode = "rel"
    render_only_leaves: bool = False
    render_stem: bool = False
    stem_size: float = 0.015
    #
    scale_strategy: Literal["none", "linear", "square"] = "none"
    leaf_shape: Literal["square", "leaf1"] = "square"


_LEAF1_HE_RATIO = 1.6
_LEAF1_CE = (_LEAF1_HE_RATIO * _LEAF1_HE_RATIO - 1) / 2
_LEAF1_ANG = math.degrees(math.atan(_LEAF1_HE_RATIO / _LEAF1_CE))


@dataclass
class TreeRewriteSpec:
    rewrite_type: TreeRewriteType
    rewrite_args: tuple[float, ...]
    """ 
    AddBranch/AddBranchBoth spec: (parent_id, length, theta, r)  
    RemoveBranch: (child_id,)
    SplitBranch: (child_id, frac, l, theta, r)
    AddAnywhere: (parent_id, l1, theta1, r1, l2, theta2, r2, ...)
    """

    def __repr__(self) -> str:
        return f"{self.rewrite_type.name}({self.rewrite_args})"


@dataclass
class Tree:
    root: torch.Tensor  # (2,)
    ls: torch.Tensor  # (n_nodes - 1,)
    thetas: torch.Tensor  # (n_nodes - 1,), radians
    # domain and range for parents are different i.e, 0 <= par[x] < n_nodes, but 0 <= x < n_nodes - 1
    # if par of x is root then, par[x] == 0;
    # parents[i - 1] < i;
    parents: tuple[int, ...]  # (n_nodes - 1,)
    rs: torch.Tensor  # (n_nodes,) [root_rs, *rest]
    args: TreeCollectionArgs
    id: int = field(default_factory=lambda: Context.get().gen_id())
    payload: TreePayload = field(default_factory=TreePayload)
    # precomputed values
    _precomputed: bool = field(default=False, init=False)
    _xs: torch.Tensor = field(default=torch.empty(0), init=False)
    _thetas: torch.Tensor = field(default=torch.empty(0), init=False)

    def __post_init__(self):
        assert self.root.shape == (2,), f"root must have shape (2,), got {self.root.shape}"
        assert self.ls.ndim == 1, f"ls must be a 1D tensor, got {self.ls.ndim}"
        assert self.thetas.ndim == 1, f"thetas must be a 1D tensor, got {self.thetas.ndim}"
        assert self.rs.ndim == 1, f"rs must be a 1D tensor, got {self.rs.ndim}"
        assert (
            len(self.ls) == len(self.thetas) == len(self.parents) == len(self.rs) - 1
        ), f"len(ls) != len(thetas) != len(parents) != len(rs) - 1"
        return

    def _forward_kinematics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        returns:
        - positions: (n_nodes, 2)
        - thetas: (n_nodes,)
        """
        if self._precomputed:
            return self._xs, self._thetas
        theta_mode = self.args.theta_mode
        n_nodes = len(self.rs)  # number of nodes
        positions = torch.zeros((n_nodes, 2), dtype=self.root.dtype, device=self.root.device)
        thetas_ = torch.zeros((n_nodes,), device=self.thetas.device)
        # Set root position
        positions[0] = self.root
        thetas_[0] = torch.pi / 2
        # theta_i is angle of node_{i+1}
        # true angle of node_{i+1} is angle of parent[node_i+1] + theta_i
        for i in range(1, n_nodes):
            if theta_mode == "rel":
                thetas_[i] = thetas_[self.parents[i - 1]] + self.thetas[i - 1]
            elif theta_mode == "abs":
                thetas_[i] = self.thetas[i - 1]
            else:
                raise ValueError(f"Unknown theta_mode: {theta_mode}")
        cs = torch.stack([thetas_[1:].cos(), thetas_[1:].sin()], dim=-1)  # (n_nodes-1, 2)
        # Compute positions
        for i in range(1, n_nodes):
            parent_idx = self.parents[i - 1]
            positions[i] = positions[parent_idx] + cs[i - 1] * self.ls[i - 1]
        self._xs = positions
        self._thetas = thetas_
        self._precomputed = True
        return positions, thetas_

    def visualize(
        self,
        ax: MPLVisualizerAxes,
        line_style: LineStyle = LineStyle(),
        leaf_style: LineStyle = LineStyle(color="black", linewidth=1),
        show_indices: bool = False,
        node_index_color: str = "black",
        edge_index_color: str = "blue",
        show_leaf: bool = False,
        show_stem: bool = True,
        leaf_alpha: float = 0.2,
        node_radius: float = 0.01,
    ) -> None:
        """
        Visualize the tree.
        Parameters
        ----------
        ax : MPLVisualizerAxes
            Target visualizer / Matplotlib axes wrapper.
        line_style : LineStyle
            Colour / width used for the geometry.
        show_indices : bool
            If True (default) write node and edge indices.
        node_index_color : str
            Colour for node labels (default: red).
        edge_index_color : str
            Colour for edge labels (default: blue).
        """
        # --- forward kinematics
        node_positions, node_thetas = self._forward_kinematics()
        # --- draw nodes
        for i, (x, y) in enumerate(node_positions.tolist()):
            if show_stem:
                ax.ax.add_patch(
                    Circle(
                        (x, y),
                        node_radius,
                        color=line_style.color,
                        linewidth=line_style.linewidth,
                        fill=True,
                        zorder=3,
                    )
                )
            if show_leaf:
                if self.args.leaf_shape == "square":
                    sz = self.rs[i].item()
                    ax.ax.add_patch(
                        Rectangle(
                            (x - sz, y - sz),
                            sz * 2,
                            sz * 2,
                            angle=node_thetas[i].item() * 180 / math.pi,
                            rotation_point="center",
                            color=leaf_style.color,
                            linewidth=leaf_style.linewidth,
                            fill=False,
                            alpha=leaf_alpha,
                            zorder=1,
                        )
                    )
                elif self.args.leaf_shape == "leaf1":
                    sz = self.rs[i].item()
                    ang = node_thetas[i].item()
                    dyx = math.cos(ang)
                    dyy = math.sin(ang)
                    dxx = dyy
                    dxy = -dyx
                    ce = sz * _LEAF1_CE
                    ax.ax.add_patch(
                        Arc(
                            (x + dyx * sz, y + dyy * sz),
                            sz * 2,
                            sz * 2,
                            angle=ang * 180 / math.pi - 90,
                            theta1=180,
                            theta2=0,
                            color=leaf_style.color,
                            linewidth=leaf_style.linewidth,
                            fill=False,
                            zorder=1,
                            alpha=leaf_alpha,
                        )
                    )
                    ax.ax.add_patch(
                        Arc(
                            (x + sz * dyx + ce * dxx, y + sz * dyy + ce * dxy),
                            (ce + sz) * 2,
                            (ce + sz) * 2,
                            angle=ang * 180 / math.pi - 90,
                            theta1=180 - _LEAF1_ANG,
                            theta2=180,
                            color=leaf_style.color,
                            linewidth=leaf_style.linewidth,
                            fill=False,
                            zorder=1,
                            alpha=leaf_alpha,
                        )
                    )
                    ax.ax.add_patch(
                        Arc(
                            (x + sz * dyx - ce * dxx, y + sz * dyy - ce * dxy),
                            (ce + sz) * 2,
                            (ce + sz) * 2,
                            angle=ang * 180 / math.pi - 90,
                            theta1=0,
                            theta2=_LEAF1_ANG,
                            color=leaf_style.color,
                            linewidth=leaf_style.linewidth,
                            fill=False,
                            zorder=1,
                            alpha=leaf_alpha,
                        )
                    )

            if show_indices:
                ax.ax.text(
                    x,
                    y,
                    str(i),
                    color=node_index_color,
                    fontsize=8,
                    ha="center",
                    va="center",
                    zorder=4,
                )

        # --- draw edges --------------------------------------------------------
        n_nodes = len(self.rs)
        for child_id in range(1, n_nodes):
            parent_id = self.parents[child_id - 1]
            (x1, y1), (x2, y2) = node_positions[parent_id].tolist(), node_positions[child_id].tolist()
            if show_stem:
                ax.ax.add_line(
                    Line2D(
                        [x1, x2],
                        [y1, y2],
                        color=line_style.color,
                        linewidth=line_style.linewidth,
                        zorder=2,
                    )
                )
            if show_indices:
                # mid-point
                mx, my = (x1 + x2) * 0.5, (y1 + y2) * 0.5
                # small offset perpendicular to the segment (for readability)
                dx, dy = x2 - x1, y2 - y1
                seg_len = math.hypot(dx, dy)
                if seg_len > 1e-6:
                    # normalised perpendicular
                    ox, oy = -dy / seg_len, dx / seg_len
                    mx += 0.02 * ox
                    my += 0.02 * oy
                ax.ax.text(
                    mx,
                    my,
                    str(child_id),
                    color=edge_index_color,
                    fontsize=8,
                    ha="center",
                    va="center",
                    zorder=4,
                )
        return

    @torch.no_grad()
    def cleanup(self, cleanup_small_leaves: Optional[float] = None, max_iter: int = 1) -> "Tree":
        tree = self
        if cleanup_small_leaves is not None:
            tree = tree._cleanup_small_leaves(cleanup_small_leaves)
        if max_iter > 0:
            iter_ = 0
            for iter_ in range(max_iter):
                pos, _ = self._forward_kinematics()  # (n,2)
                edges = [(tree.parents[i - 1], i) for i in range(1, len(tree.rs))]  # (parent, child)
                changed = False
                # Sweep all unordered edge pairs
                for (pa, ca), (pb, cb) in (
                    (edges[i], edges[j]) for i in range(len(edges)) for j in range(i + 1, len(edges))
                ):
                    # skip if edges share a node
                    if len({pa, ca, pb, cb}) < 4:
                        continue
                    # Positions of the four nodes
                    A, B = pos[pa], pos[ca]
                    C, D = pos[pb], pos[cb]
                    hit, t_ab, u_cd = _seg_intersect(A, B, C, D)  # intersect, pos on AB, pos on CD
                    if not hit:
                        continue
                    # Angle between v1 and v2
                    v1 = B - A
                    v2 = D - C
                    v1_n = v1 / v1.norm()
                    v2_n = v2 / v2.norm()
                    dot = torch.clamp(torch.dot(v1_n, v2_n), -1.0, 1.0)
                    cross_z = v1_n[0] * v2_n[1] - v1_n[1] * v2_n[0]
                    ang_signed = math.atan2(cross_z, dot)
                    # def wrap(a: float) -> float:
                    #     return (a + math.pi) % (2 * math.pi) - math.pi
                    # Split the larger id first:
                    if ca > cb:
                        first, second = (pa, ca, t_ab), (pb, cb, u_cd)
                    else:
                        first, second = (pb, cb, u_cd), (pa, ca, t_ab)
                    # --- First split
                    # print(tree.parents)
                    # print(f"{first[0]} {first[1]} {first[2]}")
                    # print(f"{second[0]} {second[1]} {second[2]}")
                    tree = tree.apply_rewrite(TreeRewriteSpec(TreeRewriteType.SplitBranch, (first[1], first[2])))
                    # --- second split
                    tree = tree.apply_rewrite(TreeRewriteSpec(TreeRewriteType.SplitBranch, (second[1], second[2])))
                    parents = list(tree.parents)
                    parents[first[1] + 1] = second[1]  # parent of tail1 is new2
                    parents[second[1]] = first[1] + 1  # parent of tail2 is new1
                    # print(parents)
                    # print(second[1], first[1] + 1)
                    thetas = tree.thetas.clone()
                    thetas[first[1] + 1] += ang_signed
                    thetas[second[1]] -= ang_signed
                    tree = Tree(
                        root=tree.root,
                        ls=tree.ls,
                        thetas=thetas,
                        parents=tuple(parents),
                        rs=tree.rs,
                        args=tree.args,
                    )

                    changed = True
                    break  # restart sweep
                if not changed:  # no more crossings
                    break
            tree = renumber(tree)
            print(f"Tree {self.id} renumbered")
        return tree

    def gen_rewrite_specs(self, args: TreeRewriteArgs, lim: tuple[float, float]) -> list[TreeRewriteSpec]:
        """
        AddBranch: We search through the nodes and collect ones without enough children
        RemoveBranch: We search through the nodes and collect leaves (no descendants)
        """
        _EPS = args.eps_length  # epsilon 0.01 for AddBranch
        tree_args = self.args
        specs: list[TreeRewriteSpec] = []
        # Children list
        children: list[list[int]] = [[] for _ in range(len(self.rs))]
        for child in range(1, len(self.rs)):
            parent = self.parents[child - 1]
            children[parent].append(child)
        # Collect nodes amenable to AddBranch
        n_children = [len(x) for x in children]
        add1_candidates = [i for i, x in enumerate(n_children) if x == 1]
        add2_candidates = [i for i, x in enumerate(n_children) if x == 0]

        if args.add_branch:
            len_ = _EPS if args.add_branch_epsilon else args.default_length
            # Branches already with one branch, we add another
            for add_candidate in add1_candidates:
                if args.random_angle:
                    new_angle = random.random() * 2 * args.default_angle - args.default_angle
                else:
                    new_angle = args.default_angle if self.thetas[add_candidate] < 0 else -args.default_angle
                specs.append(
                    TreeRewriteSpec(
                        TreeRewriteType.AddBranch,
                        (add_candidate, len_, new_angle, args.default_r),
                    )
                )
            # Branches with no children, we can
            # 1) Add left branch, 2) Add right branch, 3) Add both branches
            for add_candidate in add2_candidates:
                # Add left
                new_angle = random.random() * args.default_angle if args.random_angle else args.default_angle
                specs.append(
                    TreeRewriteSpec(
                        TreeRewriteType.AddBranch,
                        (add_candidate, len_, new_angle, args.default_r),  # length  # radius
                    )
                )
                # Add right
                new_angle = random.random() * args.default_angle if args.random_angle else args.default_angle
                specs.append(
                    TreeRewriteSpec(
                        TreeRewriteType.AddBranch,
                        (add_candidate, len_, -new_angle, args.default_r),  # length  # radius
                    )
                )
                # Add both
                if args.add_both_branch:
                    new_angle = random.random() * args.default_angle if args.random_angle else args.default_angle
                    specs.append(
                        TreeRewriteSpec(
                            TreeRewriteType.AddBranchBoth,
                            (add_candidate, len_, new_angle, args.default_r),  # length  # radius
                        )
                    )
        # Branch removal at leaves
        if args.remove_branch and len(self.ls) > 2:
            xs_, thetas_ = self._forward_kinematics()

            for i_ in range(1, len(self.rs)):
                children_ = children[i_]
                if len(children_) == 0:
                    to_remove = (
                        (not args.remove_branch_epsilon) or (self.ls[i_ - 1] < 10 * _EPS) or (self.rs[i_] < _EPS)
                    )
                    if to_remove:
                        specs.append(TreeRewriteSpec(TreeRewriteType.RemoveBranch, (i_,)))
                elif len(children_) == 1 and args.remove_non_leaf:
                    if args.remove_branch_epsilon and self.ls[i_ - 1] > 10 * _EPS:
                        continue
                    parent_ = self.parents[i_ - 1]
                    grandchild_ = children_[0]
                    diff = xs_[grandchild_] - xs_[parent_]
                    new_l = diff.norm().item()
                    if tree_args.ls_max is not None and new_l > tree_args.ls_max:
                        continue
                    new_theta_abs = torch.atan2(diff[1], diff[0]).item()
                    if tree_args.theta_mode == "rel":
                        new_theta = (new_theta_abs - thetas_[parent_] + math.pi) % (2 * math.pi) - math.pi
                    elif tree_args.theta_mode == "abs":
                        new_theta = new_theta_abs % (2 * math.pi)
                    else:
                        raise ValueError(f"Unknown theta_mode: {tree_args.theta_mode}")
                    if (tree_args.theta_min is not None and new_theta < tree_args.theta_min) or (
                        tree_args.theta_max is not None and new_theta > tree_args.theta_max
                    ):
                        continue
                    specs.append(TreeRewriteSpec(TreeRewriteType.RemoveBranch, (i_, grandchild_)))
        if args.split_branch:
            for child_id in range(1, len(self.rs)):
                frac = (_EPS / self.ls[child_id - 1]).item()
                frac = min(max(0.01, frac), 0.99)
                if args.split_add_branch:
                    if args.random_angle:
                        new_angle = random.random() * 2 * args.default_angle - args.default_angle
                    else:
                        new_angle = args.default_angle if random.random() < 0.5 else -args.default_angle
                    specs.append(
                        TreeRewriteSpec(
                            TreeRewriteType.SplitBranch,
                            (child_id, frac, _EPS, new_angle, args.default_r),
                        )
                    )  # split midway
                else:
                    specs.append(TreeRewriteSpec(TreeRewriteType.SplitBranch, (child_id, frac)))
        if args.add_anywhere:
            device = self.ls.device
            n_pts = max(len(self.rs), 32)  # TODO: use this so that the weight is about the same as others
            rs = self.args.rs_max or args.default_r
            pts = torch.rand((n_pts, 2), device=device) * (lim[1] - lim[0] - 2 * rs) + lim[0] + rs  # (n_pts, 2)
            specs.extend(self._gen_add_anywhere_specs(pts, args))

        # for spec in specs:
        #     if spec.rewrite_type == TreeRewriteType.RemoveBranch:
        #         print(spec)

        return specs

    def apply_rewrite(self, spec: TreeRewriteSpec) -> "Tree":
        dtype = self.ls.dtype
        device = self.ls.device
        theta_mode = self.args.theta_mode
        angle_offset = 0.0 if theta_mode == "rel" else torch.pi / 2
        match spec.rewrite_type:
            case TreeRewriteType.AddBranch:
                assert len(spec.rewrite_args) == 4, f"AddBranch args must be (parent_id, length, theta, r)"
                parent_id, length, theta, r = spec.rewrite_args
                parent_id = int(parent_id)
                return Tree(
                    root=self.root,
                    ls=torch.cat([self.ls, torch.tensor([length], dtype=dtype, device=device)]),
                    thetas=torch.cat([self.thetas, torch.tensor([theta + angle_offset], dtype=dtype, device=device)]),
                    parents=self.parents + (parent_id,),
                    rs=torch.cat([self.rs, torch.tensor([r], dtype=dtype, device=device)]),
                    args=self.args,
                )
            case TreeRewriteType.AddBranchBoth:
                assert len(spec.rewrite_args) == 4, f"AddBranch args must be (parent_id, length, theta, r)"
                # We add two nodes with opposite angles
                parent_id, length, theta, r = spec.rewrite_args
                parent_id = int(parent_id)
                return Tree(
                    root=self.root,
                    ls=torch.cat([self.ls, torch.tensor([length, length], dtype=dtype, device=device)]),
                    thetas=torch.cat(
                        [
                            self.thetas,
                            torch.tensor([theta + angle_offset, -theta + angle_offset], dtype=dtype, device=device),
                        ]
                    ),
                    parents=self.parents + (parent_id, parent_id),
                    rs=torch.cat([self.rs, torch.tensor([r, r], dtype=dtype, device=device)]),
                    args=self.args,
                )
            case TreeRewriteType.RemoveBranch:
                # 1 child) For C ---- B ---- A, remove B, assign C's parent to A
                # 0 child) For        B ---- A, remove B
                # child_id is B's id
                assert len(spec.rewrite_args) in {1, 2}, f"RemoveBranch args must be (child_id, [grandchild_id])"
                if len(spec.rewrite_args) == 1:
                    child_id = int(spec.rewrite_args[0])
                    assert child_id > 0, f"child_id must be > 0, got {child_id}"
                    ls = torch.cat([self.ls[: child_id - 1], self.ls[child_id:]])
                    thetas = torch.cat([self.thetas[: child_id - 1], self.thetas[child_id:]])
                    rs = torch.cat([self.rs[:child_id], self.rs[child_id + 1 :]])
                    parents = list(self.parents[: child_id - 1] + self.parents[child_id:])
                    # Since we removed a node, any node with higher index needs to be decremented
                    parents = [id_ if id_ < child_id else id_ - 1 for id_ in parents]
                    parents = tuple(parents)
                    return Tree(
                        root=self.root,
                        ls=ls,
                        thetas=thetas,
                        parents=parents,
                        rs=rs,
                        args=self.args,
                    )
                elif len(spec.rewrite_args) == 2:
                    # print(self.parents)
                    child_id = int(spec.rewrite_args[0])
                    assert child_id > 0, f"child_id must be > 0, got {child_id}"
                    old_parent = self.parents[child_id - 1]
                    grandchild_id = -1

                    for i, p in enumerate(self.parents):
                        if p == child_id:
                            grandchild_id = i
                            break
                    grandchild_id += 1
                    # if grandchild_id != -1:
                    assert grandchild_id == int(
                        spec.rewrite_args[1]
                    ), f"grandchild_id must be != spec.rewrite_args[1], got {grandchild_id} != {spec.rewrite_args[1]}"
                    # print(f"xx: {old_parent} -> {child_id} -> {grandchild_id}")
                    thetas = self.thetas.clone()
                    ls = self.ls.clone()
                    if theta_mode == "rel":
                        # thetas[grandchild_id - 1] += self.thetas[child_id - 1]
                        a_ = ls[child_id - 1]
                        b_ = ls[grandchild_id - 1]
                        ls[grandchild_id - 1] = (
                            a_.square() + b_.square() + 2 * a_ * b_ * thetas[grandchild_id - 1].cos()
                        ).sqrt()
                        thetas[grandchild_id - 1] = torch.atan(
                            thetas[grandchild_id - 1].sin() / (thetas[grandchild_id - 1].cos() + self.ls[child_id - 1])
                        )
                    elif theta_mode == "abs":
                        raise NotImplementedError()
                        thetas[grandchild_id - 1] = self.thetas[child_id - 1]
                    ls = torch.cat([ls[: child_id - 1], ls[child_id:]])
                    thetas = torch.cat([thetas[: child_id - 1], thetas[child_id:]])
                    rs = torch.cat([self.rs[:child_id], self.rs[child_id + 1 :]])
                    parents = list(self.parents)
                    # print(f"-1-> {self.parents}")
                    parents[grandchild_id - 1] = old_parent
                    # print(f"-2-> {parents}")
                    parents = list(parents[: child_id - 1] + parents[child_id:])
                    # print(f"-3-> {parents}")
                    # Since we removed a node, any node with higher index needs to be decremented
                    parents = [id_ if id_ < child_id else id_ - 1 for id_ in parents]
                    parents = tuple(parents)
                    # print(f"-4-> {parents}")
                    return Tree(
                        root=self.root,
                        ls=ls,
                        thetas=thetas,
                        parents=parents,
                        rs=rs,
                        args=self.args,
                    )
                else:
                    raise ValueError(f"Unknown RemoveBranch arg: {spec.rewrite_args[1]}")
            case TreeRewriteType.SplitBranch:
                # ------------------------------------------------------------------
                # Args:
                #   (child_id, frac)                                   – just split
                #   (child_id, frac, new_len, new_theta, new_r)        – split *and*
                #                                                      sprout a child
                # ------------------------------------------------------------------
                if len(spec.rewrite_args) not in (2, 5):
                    raise ValueError(
                        "SplitBranch expects 2 or 5 args: " "(child_id, frac [, new_len, new_theta, new_r])"
                    )
                # print(spec.rewrite_args)
                child_id, frac, *extra = spec.rewrite_args
                child_id = int(child_id)
                frac = float(frac)
                assert 0.01 <= frac <= 0.99, "frac must lie in (0,1)"

                parent_id = self.parents[child_id - 1]
                n_nodes_orig = len(self.rs)  # id of the *new* mid‑node
                L_old = self.ls[child_id - 1]
                theta_old = torch.tensor(0.0, device=device) if theta_mode == "rel" else self.thetas[child_id - 1]

                # --- split lengths -------------------------------------------------
                L_parent_new = L_old * frac  # P → N
                L_child_new = L_old * (1.0 - frac)  # N → C

                # --- radii: simple average for the mid‑node ------------------------
                r_mid = 0.5 * (self.rs[parent_id] + self.rs[child_id])

                # 1) update existing branch (now N → C)
                ls_new = torch.cat([self.ls[:child_id], L_child_new.unsqueeze(0), self.ls[child_id:]])
                ls_new[child_id - 1] = L_parent_new
                thetas_new = torch.cat([self.thetas[:child_id], theta_old.unsqueeze(0), self.thetas[child_id:]])
                # 3) parents list
                # if parent is greater than or equal to child_id, increment
                parents_new = [p + 1 if p >= child_id else p for p in self.parents]
                # parent of new_id is child_id
                parents_new = parents_new[:child_id] + [child_id] + parents_new[child_id:]

                # print("NEW parents")
                # print(parents_new)
                # 4) radii
                rs_new = torch.cat(
                    [
                        self.rs[:child_id],
                        torch.tensor([r_mid], dtype=self.rs.dtype, device=self.rs.device),
                        self.rs[child_id:],
                    ]
                )

                # ---- extra child sprouting off the new mid‑node -------------------
                if extra:  # len == 3
                    new_len, new_theta, new_r = extra
                    ls_new = torch.cat([ls_new, torch.tensor([new_len], dtype=self.ls.dtype, device=self.ls.device)])
                    thetas_new = torch.cat(
                        [
                            thetas_new,
                            torch.tensor(
                                [new_theta + angle_offset], dtype=self.thetas.dtype, device=self.thetas.device
                            ),
                        ]
                    )
                    parents_new.append(child_id)  # parent = mid‑node
                    rs_new = torch.cat([rs_new, torch.tensor([new_r], dtype=self.rs.dtype, device=self.rs.device)])
                return Tree(
                    root=self.root,
                    ls=ls_new,
                    thetas=thetas_new,
                    parents=tuple(parents_new),
                    rs=rs_new,
                    args=self.args,
                )
            case TreeRewriteType.AddAnywhere:
                par_id, *extra = spec.rewrite_args
                assert len(extra) % 3 == 0, f"AddAnywhere args must be (parent_id, l1, theta1, r1, l2, theta2, r2, ...)"
                ls_new_: list[float] = []
                thetas_new_: list[float] = []
                rs_new_: list[float] = []
                parents_new_: list[int] = list(self.parents)
                last_id = int(par_id)
                while len(extra) > 0:
                    l_, theta_, r_, *extra = extra
                    ls_new_.append(l_)
                    thetas_new_.append(theta_)
                    rs_new_.append(r_)
                    parents_new_.append(last_id)
                    last_id = len(parents_new_)
                return Tree(
                    root=self.root,
                    ls=torch.cat([self.ls, torch.tensor(ls_new_, dtype=self.ls.dtype, device=self.ls.device)]),
                    thetas=torch.cat(
                        [self.thetas, torch.tensor(thetas_new_, dtype=self.thetas.dtype, device=self.thetas.device)]
                    ),
                    parents=tuple(parents_new_),
                    rs=torch.cat([self.rs, torch.tensor(rs_new_, dtype=self.rs.dtype, device=self.rs.device)]),
                    args=self.args,
                )
            case _:
                raise ValueError(f"Unknown rewrite type {spec.rewrite_type}")

    def apply_rewrite_each(self, specs: list[TreeRewriteSpec]) -> list["Tree"]:
        return [self.apply_rewrite(spec) for spec in specs]

    def apply_rewrite_all(self, specs: list[TreeRewriteSpec], scores: list[float]) -> "Tree":
        """
        Assumes specs are not conflicting
        """
        # ! do `apply_rewrite_each`
        tree = Tree(
            root=self.root.clone(),
            ls=self.ls.clone(),
            thetas=self.thetas.clone(),
            rs=self.rs.clone(),
            parents=self.parents,
            args=self.args,
        )
        # og_specs = [replace(x) for x in specs]
        # Rewrite
        # RemoveBranch will invalidate node indices in future rewrites
        # Need to update all future rewrites for impacted indices by
        # decrementing them by 1
        for i in range(len(specs)):
            # print(f"{og_specs[i]} -> {specs[i]}")
            spec = specs[i]
            if spec.rewrite_type == TreeRewriteType.RemoveBranch:
                for j in range(i + 1, len(specs)):
                    # if specs[j].rewrite_type in [
                    #     TreeRewriteType.AddBranch,
                    #     TreeRewriteType.AddBranchBoth,
                    #     TreeRewriteType.RemoveBranch,
                    #     TreeRewriteType.SplitBranch,
                    #     TreeRewriteType.AddAnywhere,
                    # ]:
                    args_ = list(specs[j].rewrite_args)
                    if args_[0] > spec.rewrite_args[0]:
                        args_[0] -= 1
                    if (
                        specs[j].rewrite_type == TreeRewriteType.RemoveBranch
                        and len(specs[j].rewrite_args) == 2
                        and args_[1] > spec.rewrite_args[0]
                    ):
                        args_[1] -= 1
                    specs[j].rewrite_args = tuple(args_)
            elif spec.rewrite_type == TreeRewriteType.SplitBranch:
                # split will add a node
                for j in range(i + 1, len(specs)):
                    # if specs[j].rewrite_type in [
                    #     TreeRewriteType.AddBranch,
                    #     TreeRewriteType.AddBranchBoth,
                    #     TreeRewriteType.RemoveBranch,
                    #     TreeRewriteType.SplitBranch,
                    #     TreeRewriteType.AddAnywhere,
                    # ]:
                    args_ = list(specs[j].rewrite_args)
                    if args_[0] > spec.rewrite_args[0]:
                        args_[0] += 1
                    if (
                        specs[j].rewrite_type == TreeRewriteType.RemoveBranch
                        and len(specs[j].rewrite_args) == 2
                        and args_[1] > spec.rewrite_args[0]
                    ):
                        args_[1] += 1
                    specs[j].rewrite_args = tuple(args_)
            tree = tree.apply_rewrite(spec)
        return tree

    def _gen_add_anywhere_specs(self, pts: torch.Tensor, args: TreeRewriteArgs) -> list[TreeRewriteSpec]:
        specs: list[TreeRewriteSpec] = []
        xs, thetas = self._forward_kinematics()  # (n_nodes, 2), (n_nodes,)
        tree_args = self.args

        diff = pts.unsqueeze(-2) - xs  # (n_pts, n_nodes, 2)
        dnorm = diff.norm(dim=-1)  # (n_pts, n_nodes,)
        dx, dy = diff.unbind(dim=-1)  # (n_pts, n_nodes,)
        ang = torch.atan2(dy, dx)  # (n_pts, n_nodes,)
        dnorm_np = dnorm.tolist()

        r_ = args.default_r if not tree_args.optimize_rs else tree_args.rs_min if tree_args.rs_min is not None else 0.0

        if tree_args.theta_mode == "abs":
            # find the closest point within the angle range
            if tree_args.theta_min is not None and tree_args.theta_max is not None:
                assert (
                    tree_args.theta_min < tree_args.theta_max
                ), f"theta_min must be less than theta_max, got {tree_args.theta_min} >= {tree_args.theta_max}"
                theta_min = tree_args.theta_min + torch.pi
                theta_range = tree_args.theta_max - tree_args.theta_min
                ang_ = (ang - theta_min).remainder(torch.pi * 2)  # (n_pts, n_nodes,)
                mask = ang_ < theta_range  # (n_pts, n_nodes,)
                dnorm = torch.where(mask, dnorm, torch.inf)
            min_, amin = dnorm.min(dim=-1)  # (n_pts,)
            ang = ang.cpu().numpy()
            for i, (min__, amin__) in enumerate(zip(min_.tolist(), amin.tolist())):
                if min__ != torch.inf:
                    ang_ = ang[i][amin__]
                    if tree_args.theta_min is not None:
                        ang_ = (ang_ - tree_args.theta_min) % (2 * torch.pi) + tree_args.theta_min
                    num_ = math.ceil(dnorm[i][amin__] / tree_args.ls_max) if tree_args.ls_max is not None else 1
                    l_ = dnorm_np[i][amin__] / num_
                    specs.append(
                        TreeRewriteSpec(
                            TreeRewriteType.AddAnywhere,
                            (amin__, *(l_, ang_, r_) * (num_ - 1), l_, ang_, args.add_anywhere_last_r),
                        )
                    )
        elif tree_args.theta_mode == "rel":
            theta_d = (ang - thetas + torch.pi).remainder(torch.pi * 2) - torch.pi  # (n_pts, n_nodes) (-pi, pi)
            theta_d_sgn = theta_d.sign().tolist()  # (n_pts, n_nodes)
            theta_d = theta_d.abs()  # (n_pts, n_nodes)
            theta_d_np = theta_d.tolist()
            # Heuristics to find points with fewest intermediate points
            n_heur = torch.where(theta_d < torch.pi / 2, 1.0, torch.inf)  # (n_pts, n_nodes)
            theta_max = None
            d_max = None
            if tree_args.theta_min is not None and tree_args.theta_max is not None:
                assert tree_args.theta_min < 0 and tree_args.theta_max > 0
                theta_max = min(abs(tree_args.theta_min), abs(tree_args.theta_max))
                n_heur = torch.maximum(n_heur, 2 * theta_d / theta_max - 1)
            if tree_args.ls_max is not None:
                d_max = tree_args.ls_max
                # magic
                n_cur = (dnorm / d_max - 1) / theta_d.sinc() + 1
                n_heur = torch.maximum(n_heur, n_cur)

            if args.add_anywhere_strategy == "fewest":
                _, amin = n_heur.min(dim=-1)  # (n_pts,)
            elif args.add_anywhere_strategy == "nearest":
                _, amin = torch.where(n_heur == torch.inf, torch.inf, dnorm).min(dim=-1)  # (n_pts,)
            else:
                raise ValueError(f"Unknown add_anywhere_strategy: {args.add_anywhere_strategy}")

            for i, amin__ in enumerate(amin.tolist()):
                n = n_heur[i][amin__]
                if n == torch.inf:
                    continue
                n = math.ceil(n)
                theta_d_ = theta_d_np[i][amin__]
                d_ = dnorm_np[i][amin__]
                n_iters = 0
                work = False
                while n_iters < 10:
                    work = True
                    if theta_max is not None and not (n >= 2 * theta_d_ / theta_max - 1):
                        work = False
                    if d_max is not None and not (d_ / d_max <= _the_func(theta_d_, n)):
                        work = False
                    if not work:
                        n += 1
                    else:
                        break
                    n_iters += 1
                if not work:
                    continue

                # actual params for intermediate nodes
                theta_ = 2 * theta_d_ / (n + 1) * theta_d_sgn[i][amin__]
                l_ = d_ / _the_func(theta_d_, n)
                assert (
                    theta_max is None or abs(theta_) <= theta_max + 1e-6
                ), f"theta_ {theta_} must be <= theta_max {theta_max}"
                assert d_max is None or l_ <= d_max + 1e-6, f"l_ {l_} must be <= d_max {d_max}"
                specs.append(
                    TreeRewriteSpec(
                        TreeRewriteType.AddAnywhere,
                        (amin__, *(l_, theta_, r_) * (n - 1), l_, theta_, args.add_anywhere_last_r),
                    )
                )
        else:
            raise ValueError(f"Unknown theta_mode: {tree_args.theta_mode}")
        return specs

    def _cleanup_small_leaves(self, threshold: float) -> "Tree":
        """ """
        n_nodes = len(self.rs)
        chd = [0 for _ in range(n_nodes)]
        deleted: set[int] = set()
        for i in range(n_nodes - 1, 0, -1):
            if chd[i] == 0 and self.rs[i] < threshold:
                deleted.add(i)
            chd[self.parents[i - 1]] += 1

        if len(deleted) == 0:
            return self

        ids: list[int] = []
        new_id_map: dict[int, int] = {0: 0}
        new_parents: list[int] = []
        for i in range(1, n_nodes):
            if i in deleted:
                continue
            ids.append(i - 1)
            new_id_map[i] = len(ids)
            new_parents.append(new_id_map[self.parents[i - 1]])

        print(f"Tree {self.id} cleaned up, removed {len(deleted)} leaves")

        return Tree(
            root=self.root,
            ls=self.ls[ids],
            thetas=self.thetas[ids],
            parents=tuple(new_parents),
            rs=self.rs[[0] + [i + 1 for i in ids]],
            id=self.id,
            payload=self.payload,
            args=self.args,
        )

    @classmethod
    def create_random(cls, max_nodes: int, max_depth: int, args: TreeCollectionArgs, seed: int = 0) -> "Tree":
        random.seed(seed)
        torch.manual_seed(seed)

        root = torch.rand(2) * 2 - 1  # Random position in [-1, 1]
        root_rs = torch.rand(1).item() * 0.1 + 0.05  # Random radius

        ls: list[torch.Tensor] = []
        thetas: list[torch.Tensor] = []
        parents: list[int] = []
        rs: list[torch.Tensor] = [torch.tensor(root_rs)]  # Start with root radius

        nodes = [(0, 0)]  # (node_index, depth) - root is node 0 at depth 0
        node_count = 1

        while nodes and node_count < max_nodes:
            parent_idx, current_depth = nodes.pop(0)  # BFS

            if current_depth >= max_depth:
                continue

            num_children = random.randint(0, min(3, max_nodes - node_count))  # Add up to 3 children or remaining nodes

            for _ in range(num_children):
                if node_count >= max_nodes:
                    break

                l_val = torch.rand(1).item() * 0.5 + 0.1  # Random length
                theta_val = (torch.rand(1).item() * 2 - 1) * torch.pi  # Random angle in [-pi, pi]
                r_val = torch.rand(1).item() * 0.1 + 0.02  # Random radius, slightly smaller

                ls.append(torch.tensor(l_val))
                thetas.append(torch.tensor(theta_val))
                parents.append(parent_idx)
                rs.append(torch.tensor(r_val))

                nodes.append((node_count, current_depth + 1))
                node_count += 1

        device = root.device
        return cls(
            root=root,
            ls=safe_stack(ls, (), device=device),
            thetas=safe_stack(thetas, (), device=device),
            parents=tuple(parents),
            rs=safe_stack(rs, (), device=device),
            args=args,
        )


def _the_func(theta_d_: float, n: int) -> float:
    """
    sin(n * theta_d_ / (n + 1)) / sin(theta_d_ / (n + 1))
    """
    if theta_d_ < 1e-12:
        return n
    return math.sin(n * theta_d_ / (n + 1)) / math.sin(theta_d_ / (n + 1))


def _seg_intersect(
    p: torch.Tensor, q: torch.Tensor, r: torch.Tensor, s: torch.Tensor  # (2,), (2,)
) -> tuple[bool, float, float]:
    """Return (do_intersect, t, u) where
    p + t (q-p)  ==  r + u (s-r)  for t,u in (0,1).
    If no proper intersection, returns (False, 0, 0).
    """
    v = q - p
    w = s - r
    denom = v[0] * w[1] - v[1] * w[0]
    if denom.abs() < 1e-6:
        return False, 0.0, 0.0  # Parallel or collinear – ignore
    diff = r - p
    t = (diff[0] * w[1] - diff[1] * w[0]) / denom
    u = (diff[0] * v[1] - diff[1] * v[0]) / denom
    return (0.01 < t.item() < 0.99 and 0.01 < u.item() < 0.99), t.item(), u.item()


@torch.no_grad()
def renumber(tree: Tree) -> Tree:
    """
    Return same tree whose indices satisfy parent < child.
    """
    n = len(tree.rs)
    if n <= 1:
        return tree
    # Children list
    children: list[list[int]] = [[] for _ in range(n)]
    for child in range(1, n):
        parent = tree.parents[child - 1]
        children[parent].append(child)
    # BFS
    order: list[int] = []
    queue: list[int] = [0]
    while queue:
        u = queue.pop(0)
        order.append(u)
        queue.extend(children[u])
    assert len(order) == n, "broken tree?"
    # old_idx -> new_idx
    new_idx = [-1] * n
    for new_i, old_i in enumerate(order):
        new_idx[old_i] = new_i
    device, dtype = tree.ls.device, tree.ls.dtype
    new_rs = tree.rs[order].clone()
    new_ls_: list[torch.Tensor] = []
    new_thetas_: list[torch.Tensor] = []
    new_parents: list[int] = []
    # Remap
    for pos_in_order in range(1, n):  # reduced coordinates
        old_child = order[pos_in_order]
        edge_idx = old_child - 1
        new_ls_.append(tree.ls[edge_idx])
        new_thetas_.append(tree.thetas[edge_idx])
        old_parent = tree.parents[edge_idx]
        new_parents.append(new_idx[old_parent])
    new_ls = torch.stack(new_ls_).to(device=device, dtype=dtype)
    new_thetas = torch.stack(new_thetas_).to(device=device, dtype=tree.thetas.dtype)
    return Tree(
        root=tree.root,
        ls=new_ls,
        thetas=new_thetas,
        parents=tuple(new_parents),
        rs=new_rs,
        id=tree.id,
        payload=tree.payload,
        args=tree.args,
    )


@dataclass
class TreeCollection(ObjectCollection[Tree]):
    roots: torch.Tensor  # (n_trees, 2)
    root_rs: torch.Tensor  # (n_trees,)
    ls: tuple[torch.Tensor, ...]  # n_depth * (n_nodes,); no roots
    thetas: tuple[torch.Tensor, ...]  # n_depth * (n_nodes,); no roots
    rs: tuple[torch.Tensor, ...]  # n_depth * (n_nodes,); no roots
    parents: tuple[
        tuple[int, ...], ...
    ]  # n_depth * (n_nodes,) storing the index of the parent node in the previous depth; no roots
    tree_ids: tuple[int, ...]  # n_trees
    tree_node_ords: tuple[tuple[tuple[int, int], ...], ...]  # n_trees x n_nodes; (depth, index); no roots
    tree_payloads: tuple[TreePayload, ...]  # n_trees
    args: TreeCollectionArgs
    # precomputed index
    _computed: bool = field(init=False, default=False)
    _par: torch.Tensor = field(
        init=False, default=torch.empty(0, dtype=torch.long)
    )  # (total_node - n_trees) (index - n_trees) to cat'd nodes
    _idx_of: torch.Tensor = field(
        init=False, default=torch.empty(0, dtype=torch.long)
    )  # (total_node) (index) to tree_idx
    _is_leaf: torch.Tensor = field(
        init=False, default=torch.empty(0, dtype=torch.bool)
    )  # (total_node) whether node is leaf
    _tree_node_idx: torch.Tensor = field(
        init=False, default=torch.empty(0, dtype=torch.long)
    )  # (n_trees, max_nodes) (index - 1) to cat'd nodes; 0 means pad
    _param_scales: list[Optional[torch.Tensor]] = field(init=False, default_factory=list)

    def __post_init__(self):
        assert (
            self.roots.ndim == 2 and self.roots.shape[1] == 2
        ), f"roots must have shape (n_trees, 2), got {self.roots.shape}"
        n_trees = self.roots.shape[0]

        assert (
            self.root_rs.ndim == 1 and self.root_rs.shape[0] == n_trees
        ), f"root_rs must have shape ({n_trees},), got {self.root_rs.shape}"

        assert len(self.tree_ids) == n_trees, f"len(tree_ids) must be {n_trees}, got {len(self.tree_ids)}"
        assert (
            len(self.tree_node_ords) == n_trees
        ), f"len(tree_node_ords) must be {n_trees}, got {len(self.tree_node_ords)}"
        assert (
            len(self.tree_payloads) == n_trees
        ), f"len(tree_payloads) must be {n_trees}, got {len(self.tree_payloads)}"

        n_depth = len(self.ls)
        assert len(self.thetas) == n_depth, f"len(thetas) must be {n_depth}, got {len(self.thetas)}"
        assert len(self.rs) == n_depth, f"len(rs) must be {n_depth}, got {len(self.rs)}"
        assert len(self.parents) == n_depth, f"len(parents) must be {n_depth}, got {len(self.parents)}"

        # nodes_per_depth: list[int] = []
        for d in range(n_depth):
            assert self.ls[d].ndim == 1, f"ls[{d}] must be 1D, got {self.ls[d].ndim}"
            n_nodes_d = len(self.ls[d])
            # nodes_per_depth.append(n_nodes_d)

            assert (
                self.thetas[d].ndim == 1 and len(self.thetas[d]) == n_nodes_d
            ), f"thetas[{d}] must have shape ({n_nodes_d},), got {self.thetas[d].shape}"
            assert (
                self.rs[d].ndim == 1 and len(self.rs[d]) == n_nodes_d
            ), f"rs[{d}] must have shape ({n_nodes_d},), got {self.rs[d].shape}"
            assert (
                len(self.parents[d]) == n_nodes_d
            ), f"len(parents[{d}]) must be {n_nodes_d}, got {len(self.parents[d])}"

            # if d == 0:
            #     assert all(0 <= p < n_trees for p in self.parents[d]), f"parents[0] indices must be in [0, {n_trees})"
            # else:
            #     assert all(0 <= p < nodes_per_depth[d - 1] for p in self.parents[d]), f"parents[{d}] indices must be in [0, {nodes_per_depth[d-1]})"

        # for t in range(n_trees):
        #     for d, i in self.tree_node_ords[t]:
        #         assert 0 <= d < n_depth, f"Node depth must be in [0, {n_depth}), got {d} for tree {t}"
        #         assert 0 <= i < nodes_per_depth[d], f"Node index at depth {d} must be in [0, {nodes_per_depth[d]}), got {i} for tree {t}"

    def _precompute_index(self):
        if self._computed:
            return
        self._computed = True
        device = self.roots.device

        # flattened par
        par: list[int] = []
        cnt = 0
        for i, pars in enumerate(self.parents):
            par.extend([p + cnt for p in pars])
            if i > 0:
                cnt += len(self.parents[i - 1])
            else:
                cnt += len(self.roots)
        self._par = torch.tensor(par, dtype=torch.long, device=device)

        # subtree_size
        n_trees = len(self.roots)
        subtree_size: list[int] = [1] * (len(par) + n_trees)
        for i in range(len(par) - 1, -1, -1):
            subtree_size[par[i]] += subtree_size[i + n_trees]
        subtree_size_ = torch.tensor(subtree_size[n_trees:], device=device)
        scales: list[Optional[torch.Tensor]] = []
        if self.args.scale_strategy != "none":
            if self.args.scale_strategy == "linear":
                scales_ = 1 / subtree_size_
            elif self.args.scale_strategy == "square":
                scales_ = 1 / subtree_size_.square()
            else:
                raise ValueError(f"Unknown scale_strategy: {self.args.scale_strategy}")
            scales__: list[torch.Tensor] = []
            offset_ = 0
            for i, ls_ in enumerate(self.ls):
                scales__.append(scales_[offset_ : offset_ + len(ls_)])
                offset_ += len(ls_)
            if self.args.optimize_roots:
                scales.append(None)  # roots
            if self.args.optimize_rs:
                scales.append(None)  # root_rs
                scales.extend(scales__)  # rs
            if self.args.optimize_ls:
                scales.extend(scales__)  # ls
            if self.args.optimize_thetas:
                scales.extend(scales__)  # thetas
            self._param_scales = scales

        # index of
        total_node = len(self.roots) + len(par)
        idx_of: list[int] = [-1 for _ in range(total_node)]  # total_node
        cnt = [len(self.roots), *(len(x) for x in self.ls)]
        for i in range(1, len(cnt)):
            cnt[i] += cnt[i - 1]
        for i_, node_ords_ in enumerate(self.tree_node_ords):
            idx_of[i_] = i_  # root
            for a, b in node_ords_:
                idx_of[cnt[a] + b] = i_
        self._idx_of = torch.tensor(idx_of, dtype=torch.long, device=device)

        # is leaf
        chd_cnt: list[int] = [0 for _ in range(total_node)]
        for p in par:
            chd_cnt[p] += 1
        self._is_leaf = torch.tensor([c == 0 for c in chd_cnt], dtype=torch.bool, device=device)

        # node_idx
        max_nodes = max(len(node_ords_) for node_ords_ in self.tree_node_ords) + 1
        tree_node_idx: list[list[int]] = []
        for i_, node_ords_ in enumerate(self.tree_node_ords):
            cur_: list[int] = [i_]  # root
            for a, b in node_ords_:
                cur_.append(cnt[a] + b)
            cur_.extend([-1] * (max_nodes - len(cur_)))
            tree_node_idx.append(cur_)
        self._tree_node_idx = torch.tensor(tree_node_idx, dtype=torch.long, device=device)

    def __len__(self) -> int:
        return len(self.roots)

    def parameters(self) -> list[torch.Tensor]:
        params = []
        if self.args.optimize_roots:
            params.append(self.roots)
        if self.args.optimize_rs:
            params.append(self.root_rs)
            params.extend(self.rs)
        if self.args.optimize_ls:
            params.extend(self.ls)
        if self.args.optimize_thetas:
            params.extend(self.thetas)
        return params

    def parameter_names(self) -> list[str]:
        params: list[str] = []
        if self.args.optimize_roots:
            params.append("roots")
        if self.args.optimize_rs:
            params.append("root_rs")
            params.extend(f"rs_{i}" for i in range(len(self.rs)))
        if self.args.optimize_ls:
            params.extend(f"ls_{i}" for i in range(len(self.ls)))
        if self.args.optimize_thetas:
            params.extend(f"thetas_{i}" for i in range(len(self.thetas)))
        return params

    def scale_grads_(self) -> bool:
        if self.args.scale_strategy == "none":
            return False
        for p, s in zip(self.parameters(), self._param_scales):
            if s is not None and p.grad is not None:
                p.grad.mul_(s)
        return True

    def per_object_grads(self) -> list[torch.Tensor]:
        self._precompute_index()
        results: list[torch.Tensor] = []
        device = self.roots.device

        roots = None
        root_rs = None
        rs = None
        ls = None
        thetas = None

        if self.args.optimize_roots:
            assert self.roots.grad is not None
            roots = self.roots.grad
        if self.args.optimize_rs:
            assert self.root_rs.grad is not None
            root_rs = self.root_rs.grad
            assert all(x.grad is not None for x in self.rs)
            rs = safe_cat([x.grad for x in self.rs if x.grad is not None], (), device=device)  # (n_nodes,)
        if self.args.optimize_ls:
            assert all(x.grad is not None for x in self.ls)
            ls = safe_cat([x.grad for x in self.ls if x.grad is not None], (), device=device)  # (n_nodes,)
        if self.args.optimize_thetas:
            assert all(x.grad is not None for x in self.thetas)
            thetas = safe_cat([x.grad for x in self.thetas if x.grad is not None], (), device=device)  # (n_nodes,)

        for i, tree_ord in enumerate(self._tree_node_idx[:, 1:] - len(self.tree_ids)):
            res: list[torch.Tensor] = []
            tree_ord = tree_ord[tree_ord != -1]
            if roots is not None:
                res.append(roots[i])
            if root_rs is not None:
                res.append(root_rs[i].unsqueeze(0))
            if rs is not None:
                res.append(rs[tree_ord])
            if ls is not None:
                res.append(ls[tree_ord])
            if thetas is not None:
                res.append(thetas[tree_ord])
            results.append(safe_cat(res, (), device=device))
        return results

    def device(self) -> torch.device:
        return self.roots.device

    def requires_grad_(self, requires_grad: bool = True) -> Self:
        self.roots = self.roots.detach().clone().requires_grad_(requires_grad and self.args.optimize_roots)
        self.root_rs = self.root_rs.detach().clone().requires_grad_(requires_grad and self.args.optimize_rs)
        self.ls = tuple(ls.detach().clone().requires_grad_(requires_grad and self.args.optimize_ls) for ls in self.ls)
        self.thetas = tuple(
            thetas.detach().clone().requires_grad_(requires_grad and self.args.optimize_thetas)
            for thetas in self.thetas
        )
        self.rs = tuple(rs.detach().clone().requires_grad_(requires_grad and self.args.optimize_rs) for rs in self.rs)
        return self

    def clone(self) -> "TreeCollection":
        """
        Detach and clone
        """
        return TreeCollection(
            roots=self.roots.detach().clone(),
            root_rs=self.root_rs.detach().clone(),
            ls=tuple(ls.detach().clone() for ls in self.ls),
            thetas=tuple(thetas.detach().clone() for thetas in self.thetas),
            rs=tuple(rs.detach().clone() for rs in self.rs),
            parents=self.parents,
            tree_ids=self.tree_ids,
            tree_node_ords=self.tree_node_ords,
            tree_payloads=self.tree_payloads,
            args=self.args,
        )

    def to(self, device: Union[str, torch.device, None] = None) -> "TreeCollection":
        """
        Detach and clone
        """
        return TreeCollection(
            roots=self.roots.to(device=device),
            root_rs=self.root_rs.to(device=device),
            ls=tuple(ls.to(device=device) for ls in self.ls),
            thetas=tuple(thetas.to(device=device) for thetas in self.thetas),
            rs=tuple(rs.to(device=device) for rs in self.rs),
            parents=self.parents,
            tree_ids=self.tree_ids,
            tree_node_ords=self.tree_node_ords,
            tree_payloads=self.tree_payloads,
            args=self.args,
        )

    def get_object(self, idx: int, detach: bool = True) -> Tree:
        ls: list[torch.Tensor] = []
        thetas: list[torch.Tensor] = []
        rs: list[torch.Tensor] = []
        parents: list[int] = []
        tree_ords = self.tree_node_ords[idx]

        imap: dict[tuple[int, int], int] = {}  # (depth, i_) -> i
        for i, (depth, i_) in enumerate(tree_ords):
            ls.append(self.ls[depth][i_])
            thetas.append(self.thetas[depth][i_])
            rs.append(self.rs[depth][i_])
            imap[(depth, i_)] = i + 1

        for depth, i_ in tree_ords:
            if depth == 0:
                parents.append(0)
            else:
                parents.append(imap[(depth - 1, self.parents[depth][i_])])

        device = self.roots.device

        return Tree(
            root=maybe_detach(self.roots[idx], detach),
            ls=maybe_detach(safe_stack(ls, (), device=device), detach),
            thetas=maybe_detach(safe_stack(thetas, (), device=device), detach),
            parents=tuple(parents),
            rs=maybe_detach(safe_stack([self.root_rs[idx], *rs], (), device=device), detach),
            id=self.tree_ids[idx],
            payload=self.tree_payloads[idx],
            args=self.args,
        )

    @classmethod
    def from_object(cls, object: Tree, **kwargs) -> "TreeCollection":
        depths = [-1]
        for i, p in enumerate(object.parents):
            assert p < i + 1, f"parent must have index less than self, got {p} >= {i + 1}"
            depths.append(depths[p] + 1)
        depths = depths[1:]  # (n_nodes - 1,)
        max_depth = max(depths) + 1 if len(depths) > 0 else 0
        ls: list[list[torch.Tensor]] = [[] for _ in range(max_depth)]
        thetas: list[list[torch.Tensor]] = [[] for _ in range(max_depth)]
        rs: list[list[torch.Tensor]] = [[] for _ in range(max_depth)]
        parents: list[list[int]] = [[] for _ in range(max_depth)]

        ids_: list[int] = []
        for i, (l_, theta_, r_) in enumerate(zip(object.ls, object.thetas, object.rs[1:])):
            d_ = depths[i]
            ls[d_].append(l_)
            thetas[d_].append(theta_)
            rs[d_].append(r_)
            if d_ == 0:
                parents[d_].append(0)
            else:
                parents[d_].append(ids_[object.parents[i] - 1])
            ids_.append(len(rs[d_]) - 1)

        device = object.root.device
        return cls(
            roots=object.root.unsqueeze(0),  # (1, 2)
            root_rs=object.rs[:1],  # (1,)
            ls=tuple(safe_stack(l_, (), device=device) for l_ in ls),  # (n_depth, n_nodes)
            thetas=tuple(safe_stack(t_, (), device=device) for t_ in thetas),  # (n_depth, n_nodes)
            rs=tuple(safe_stack(r_, (), device=device) for r_ in rs),  # (n_depth, n_nodes)
            parents=tuple(tuple(p_) for p_ in parents),  # (n_depth, n_nodes)
            tree_ids=(object.id,),
            tree_node_ords=(tuple((depths[i], ids_[i]) for i in range(len(object.ls))),),  # (1, all_nodes - 1)
            tree_payloads=(object.payload,),
            args=object.args,
        )

    @classmethod
    def cat(cls, collections: list["TreeCollection"], **kwargs) -> "TreeCollection":  # type: ignore[override]
        assert len(collections) > 0, "collections must not be empty"
        device = collections[0].device()
        args = collections[0].args
        max_depth = max(len(x.ls) for x in collections)
        ls: list[list[torch.Tensor]] = [[] for _ in range(max_depth)]
        thetas: list[list[torch.Tensor]] = [[] for _ in range(max_depth)]
        rs: list[list[torch.Tensor]] = [[] for _ in range(max_depth)]
        parents: list[list[int]] = [[] for _ in range(max_depth)]

        tree_node_ords: list[tuple[tuple[int, int], ...]] = []

        offset: list[int] = [0 for _ in range(max_depth + 1)]
        for ic_, collection in enumerate(collections):
            assert (
                collection.args is args
            ), f"args must be the same for all collections, got {collection.args} != {args}"
            tree_node_ords.extend(
                tuple((d__, i__ + offset[d__ + 1]) for d__, i__ in node_ords_)
                for node_ords_ in collection.tree_node_ords
            )

            for i, (ls_, thetas_, rs_, pars_) in enumerate(
                zip(collection.ls, collection.thetas, collection.rs, collection.parents)
            ):
                ls[i].append(ls_)
                thetas[i].append(thetas_)
                rs[i].append(rs_)
                parents[i].extend(p_ + offset[i] for p_ in pars_)

            for i, ls_ in enumerate(collection.ls):
                offset[i + 1] += len(ls_)
            offset[0] += 1

        return cls(
            roots=safe_cat([x.roots for x in collections], (2,), device),
            root_rs=safe_cat([x.root_rs for x in collections], (), device),
            ls=tuple(safe_cat(l_, (), device=device) for l_ in ls),  # (n_depth, n_nodes)
            thetas=tuple(safe_cat(t_, (), device=device) for t_ in thetas),  # (n_depth, n_nodes)
            rs=tuple(safe_cat(r_, (), device=device) for r_ in rs),  # (n_depth, n_nodes)
            parents=tuple(tuple(p_) for p_ in parents),  # (n_depth, n_nodes)
            tree_ids=sum([x.tree_ids for x in collections], ()),
            tree_node_ords=tuple(tree_node_ords),  # (n_trees, n_nodes)
            tree_payloads=sum([x.tree_payloads for x in collections], ()),
            args=args,
        )

    def rasterize(
        self,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        positions: (..., 2)

        returns: (n_trees, ...)
        """
        self._precompute_index()
        *shape, _ = positions.shape
        device = self.roots.device
        theta_mode = self.args.theta_mode
        xs_: list[torch.Tensor] = [self.roots]  # d x (*, 2)
        thetas_: list[torch.Tensor] = [torch.full((len(self.roots),), torch.pi / 2, device=device)]
        thetas_cs_: list[torch.Tensor] = [torch.tensor([0.0, 1.0], device=device).expand(len(self.roots), -1)]
        for i_, (l_, t_, p_) in enumerate(zip(self.ls, self.thetas, self.parents)):
            xs_par = xs_[-1][list(p_)]  # (*, 2)
            if theta_mode == "rel":
                thetas_cur = thetas_[-1][list(p_)] + t_  # (*,)
            elif theta_mode == "abs":
                thetas_cur = t_  # (*,)
            else:
                raise ValueError(f"Unknown theta_mode: {theta_mode}")
            thetas_cs_cur = torch.stack([thetas_cur.cos(), thetas_cur.sin()], dim=-1)  # (*, 2)
            xs_cur = xs_par + thetas_cs_cur * l_.unsqueeze(-1)  # (*, 2)
            thetas_.append(thetas_cur)
            xs_.append(xs_cur)
            thetas_cs_.append(thetas_cs_cur)

        xs = safe_cat(xs_, (2,), device)  # (total_nodes, 2)
        rs = safe_cat([self.root_rs, *self.rs], (), device).unsqueeze(-1)  # (total_nodes, 1)
        cos_, sin_ = safe_cat(thetas_cs_, (2,), device).unbind(dim=-1)  # (total_nodes,)

        # render stem

        res = torch.full((*shape, len(self.roots)), torch.inf, device=self.roots.device)  # (..., n_trees)
        if self.args.render_stem:
            n_shapes = len(self.roots)
            # https://iquilezles.org/articles/distfunctions2d/  Segment
            a = xs[n_shapes:]  # (n_edges, 2)
            b = xs[self._par]  # (n_edges, 2)
            pa = positions.unsqueeze(-2) - a  # (..., n_edges, 2)
            ba = b - a  # (n_edges, 2)
            h = ((pa * ba).sum(dim=-1) / ba.square().sum(dim=-1).clamp(min=1e-12)).clamp(0, 1)  # (..., n_edges)
            sdf_stem = (pa - ba * h.unsqueeze(-1)).norm(dim=-1) - self.args.stem_size  # (..., n_edges)
            res = torch.scatter_reduce(
                res, -1, self._idx_of[n_shapes:].expand(*shape, -1), sdf_stem, reduce="amin"
            )  # (..., n_trees)

        # render nodes

        if self.args.render_only_leaves:
            mask = self._is_leaf
            idx_of = self._idx_of[mask]
            xs = xs[mask]
            rs = rs[mask]
            cos_ = cos_[mask]
            sin_ = sin_[mask]
        else:
            idx_of = self._idx_of

        # idx_of  # (render_nodes,)

        irot = torch.stack([cos_, sin_, -sin_, cos_], dim=-1).reshape(-1, 2, 2)  # (render_nodes, 2, 2)
        diff = (irot @ (positions.unsqueeze(-2) - xs).unsqueeze(-1)).squeeze(-1)  # (..., render_nodes, 2)
        if self.args.leaf_shape == "square":
            d = diff.abs() - rs  # (..., render_nodes, 2)
            sdf_rects = d.clamp(min=0).norm(dim=-1) + d.max(dim=-1).values.clamp(max=0)  # (..., render_nodes)
        elif self.args.leaf_shape == "leaf1":
            # https://www.shadertoy.com/view/Wdjfz3
            # ra = rs, rb = 0, he = rs * 1.6
            rs = rs.squeeze(-1)  # (render_nodes,)
            ce = rs * (1.6 * 1.6 - 1) / 2  # (render_nodes,)
            dy, dx = diff.unbind(-1)  # (..., render_nodes)
            dx = dx.abs()
            dy = dy - rs
            case1_ = torch.stack([dx, dy], dim=-1).norm(dim=-1) - rs  # (..., render_nodes)
            case2_ = torch.stack([dx + ce, dy], dim=-1).norm(dim=-1) - (rs + ce)  # (..., render_nodes)
            sdf_rects = torch.where(dy < 0, case1_, case2_)  # (..., render_nodes)
        else:
            raise ValueError(f"Unknown leaf_shape: {self.args.leaf_shape}")

        res = torch.scatter_reduce(res, -1, idx_of.expand(*shape, -1), sdf_rects, reduce="amin")  # (..., n_trees)
        return res.permute(-1, *range(positions.ndim - 1))  # (n_trees, ...)

    def get_sizes(self) -> list[int]:
        return [len(x) for x in self.tree_node_ords]

    def project_to_valid_(self) -> Self:
        """
        Projects the collection to the valid set
        """
        if self.args.ls_min is not None or self.args.ls_max is not None:
            for ls_ in self.ls:
                ls_.data.clamp_(min=self.args.ls_min, max=self.args.ls_max)
        if self.args.theta_min is not None or self.args.theta_max is not None:
            for thetas_ in self.thetas:
                thetas_.data.clamp_(min=self.args.theta_min, max=self.args.theta_max)
        if self.args.rs_min is not None or self.args.rs_max is not None:
            for rs_ in self.rs:
                rs_.data.clamp_(min=self.args.rs_min, max=self.args.rs_max)
            self.root_rs.data.clamp_(min=self.args.rs_min, max=self.args.rs_max)
        return self


# static type checking
if __name__ == "__main__":
    TreeCollection(
        roots=torch.rand(10, 2),
        root_rs=torch.rand(10),
        ls=(torch.rand(10, 10), torch.rand(10, 10)),
        thetas=(torch.rand(10, 10), torch.rand(10, 10)),
        rs=(torch.rand(10, 10), torch.rand(10, 10)),
        parents=(),
        tree_ids=(),
        tree_node_ords=(),
        tree_payloads=(),
        args=TreeCollectionArgs(),
    )

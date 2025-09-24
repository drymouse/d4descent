import torch
import torch.sparse
import torch.nn.functional as TF
import numpy as np
from scipy.sparse import coo_matrix, linalg, csr_matrix
from matplotlib.figure import Figure
from pathlib import Path
from typing import Optional, Union
import math
from dataclasses import dataclass


def levelset88(nelx: int, nely: int, Vmax: float, tau: float, savedir: Optional[Path] = None):
    """
    Based on:
    Matlab code for a level set-based topology optimization using a reaction diffusion equation
    """
    if savedir is not None:
        savedir.mkdir(exist_ok=True, parents=True)
    # Parameter definition
    E0 = 1
    Emin = 1e-4
    nu = 0.3
    nvol = 100
    dt = 0.1
    d = -0.02
    p = 4
    phi = np.ones((nely + 1) * (nelx + 1))  # ((nely + 1) * (nelx + 1),)
    struc = np.ones((nely, nelx))  # (nely, nelx)
    volInit = np.sum(struc) / (nelx * nely)  # ()

    # Finite element analysis preparation
    # For displacement field
    A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]])  # (4, 4)
    A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]])  # (4, 4)
    B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]])  # (4, 4)
    B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]])  # (4, 4)
    KE = (1 / (1 - nu**2) / 24) * (
        np.block([[A11, A12], [A12.T, A11]]) + nu * np.block([[B11, B12], [B12.T, B11]])
    )  # (8, 8)

    # For topological derivative
    a1 = (3 * (1 - nu) / (2 * (1 + nu) * (7 - 5 * nu))) * (-(1 - 14 * nu + 15 * nu**2) * E0) / (1 - 2 * nu) ** 2  # ()
    a2 = (3 * (1 - nu) / (2 * (1 + nu) * (7 - 5 * nu))) * 5 * E0  # ()
    A = (
        (a1 + 2 * a2)
        / 24
        * (np.block([[A11, A12], [A12.T, A11]]) + (a1 / (a1 + 2 * a2)) * np.block([[B11, B12], [B12.T, B11]]))
    )  # (8, 8)

    # Nodal numbers and element connectivity
    nodenrs = np.arange((1 + nelx) * (1 + nely)).reshape((1 + nely, 1 + nelx))  # (nely + 1, nelx + 1)
    edofVec = (2 * nodenrs[:-1, :-1].ravel()).reshape(nelx * nely, 1)  # (nely * nelx, 1)
    edofMat = edofVec + np.array(
        [(nelx + 1) * 2, (nelx + 1) * 2 + 1, (nelx + 1) * 2 + 2, (nelx + 1) * 2 + 3, 2, 3, 0, 1]
    )  # (nely * nelx, 8)

    iK = np.kron(edofMat, np.ones((8, 1), dtype=np.int64)).ravel()  # ((nely * nelx * 8) * 8,)
    jK = np.kron(edofMat, np.ones((1, 8), dtype=np.int64)).ravel()  # ((nely * nelx) * (8 * 8),)

    # Reaction diffusion equation setup
    NNdif_e = (1 / 6) * np.array([[4, -1, -2, -1], [-1, 4, -1, -2], [-2, -1, 4, -1], [-1, -2, -1, 4]])  # (4, 4)
    NN_e = (1 / 36) * np.array([[4, 2, 1, 2], [2, 4, 2, 1], [1, 2, 4, 2], [2, 1, 2, 4]])  # (4, 4)

    edofVec2 = (nodenrs[:-1, :-1]).ravel().reshape(nelx * nely, 1)  # (nely * nelx, 1)
    edofMat2 = edofVec2 + np.array([nelx + 1, nelx + 2, 1, 0])  # (nely * nelx, 4)

    iN = np.kron(edofMat2, np.ones((4, 1), dtype=np.int64)).ravel()  # ((nely * nelx * 4) * 4,)
    jN = np.kron(edofMat2, np.ones((1, 4), dtype=np.int64)).ravel()  # ((nely * nelx) * (4 * 4),)
    sNN = np.tile(NN_e.ravel(), (nely * nelx))  # (nely * nelx * 16)
    NN = coo_matrix((sNN, (iN, jN)))  # ((nely + 1) * (nelx + 1), (nely + 1) * (nelx + 1))
    sNNdif = np.tile(NNdif_e.ravel(), (nely * nelx))  # (nely * nelx, 4)
    NNdif = coo_matrix((sNNdif, (iN, jN)))  # ((nely + 1) * (nelx + 1), (nely + 1) * (nelx + 1))

    # Load and boundary conditions
    # middle nely / 16 on the rightmost edge
    iF = (nelx + 1) * 2 * np.arange(nely // 2 - round(nely / 32), nely // 2 + round(nely / 32) + 1) + nelx * 2 + 1
    F = csr_matrix((np.ones(len(iF)), (iF, np.zeros(len(iF), np.int64))), ((nely + 1) * (nelx + 1) * 2, 1))
    U = np.zeros((nely + 1) * (nelx + 1) * 2)  # (2 * (nely + 1) * (nelx + 1), 1)

    fixeddofs = np.concat([np.arange(nely + 1) * (nelx + 1) * 2, np.arange(nely + 1) * (nelx + 1) * 2 + 1], axis=-1)
    alldofs = np.arange((nely + 1) * (nelx + 1) * 2)
    freedofs = np.setdiff1d(alldofs, fixeddofs, assume_unique=True)
    T = NN / dt + tau * (nely * nelx) * NNdif
    fixeddofs_phi = np.array(
        [
            *range(nelx + 2),
            *range(nelx + 1, nely * (nelx + 1), nelx + 1),
            *range(nelx + 1 + nelx, nely * (nelx + 1) + nelx, nelx + 1),
            *range(nely * (nelx + 1), (nely + 1) * (nelx + 1)),
        ]
    )
    fixeddofs_phi.sort()
    phi[fixeddofs_phi] = 0
    alldofs_phi = np.arange((nely + 1) * (nelx + 1))
    freedofs_phi = np.setdiff1d(alldofs_phi, fixeddofs_phi, assume_unique=True)

    # Main loop
    objective: list[float] = []
    for iterNum in range(200):
        sK = (KE.ravel() * (Emin + struc.ravel() * (E0 - Emin))[..., None]).flatten()
        K = coo_matrix((sK, (iK, jK))).tocsr()
        K = (K + K.T) / 2
        U[freedofs] = linalg.spsolve(K[freedofs[:, None], freedofs], F[freedofs])  # type: ignore

        SED = (Emin + struc * (E0 - Emin)) * ((U[edofMat] @ KE) * U[edofMat]).sum(axis=-1).reshape(nely, nelx)
        TD = (1e-4 + struc * (1 - 1e-4)) * ((U[edofMat] @ A) * U[edofMat]).sum(axis=-1).reshape(nely, nelx)
        td2 = np.pad(TD, ((1, 1), (1, 1)), mode="edge")
        TDN = (td2[:-1, :-1] + td2[1:, :-1] + td2[1:, 1:] + td2[:-1, 1:]) / 4

        objective.append(SED.sum())
        vol = struc.sum() / (nelx * nely)

        # print results
        print(f"It.: {iterNum} Compl.: {SED.sum() / (nelx * nely):.4e} Vol.: {vol:.2f}")

        if savedir is not None:
            fig = Figure(figsize=(6, 4))
            ax = fig.subplots(1, 1)
            ax.imshow(struc, cmap="gray")
            fig.savefig(savedir / f"{iterNum:04d}.png")
            last = savedir / "_last.png"
            if last.exists():
                last.unlink()
            last.symlink_to(f"{iterNum:04d}.png")
            del fig

        # Check for convergence
        if iterNum > nvol and (
            abs(vol - Vmax) < 0.005
            and (np.abs(objective[-1] - np.array(objective[-5:])) < 0.01 * abs(objective[-1])).all()
        ):
            break

        # Set augmented Lagrangian parameters
        ex = Vmax + (volInit - Vmax) * max(0, 1 - (iterNum + 1) / nvol)  # ()
        lmbda = TDN.mean() * np.exp(p * ((vol - ex) / ex + d))  # ()
        C = (nely * nelx) / np.abs(TDN).sum()  # ()
        g2 = TDN.flatten()  # ((nely + 1) * (nelx + 1),)

        # Update level set function
        Y = NN @ (C * (g2 - lmbda) + phi / dt)
        phi[freedofs_phi] = linalg.spsolve(T[freedofs_phi[:, None], freedofs_phi], Y[freedofs_phi])  # type: ignore
        phi = phi.clip(-1, 1)
        phin = phi.reshape(nely + 1, nelx + 1)
        phie = (phin[:-1, :-1] + phin[1:, :-1] + phin[:-1, 1:] + phin[1:, 1:]) / 4
        struc = phie > 0

    return struc, phi


@dataclass
class SensitivityAnalysisArgs:
    nelx: int
    nely: int
    E0: float = 1.0
    Emin: float = 1e-4
    nu: float = 0.3
    p: int = 2


class SensitivityAnalysis:
    def __init__(self, args: SensitivityAnalysisArgs, boundary_cond: np.ndarray, load: np.ndarray):
        """
        - boundary_cond: (#cond, 3) (x, y, 01 (xy)) nodal
        - load: (#load, 3) (x, y, 01 (xy), load) nodal
        """
        self.args = args

        nu = args.nu
        nelx = args.nelx
        nely = args.nely

        # Finite element analysis preparation
        A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]])  # (4, 4)
        A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]])  # (4, 4)
        B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]])  # (4, 4)
        B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]])  # (4, 4)
        self.KE = (1 / (1 - nu**2) / 24) * (
            np.block([[A11, A12], [A12.T, A11]]) + nu * np.block([[B11, B12], [B12.T, B11]])
        )  # (8, 8)

        # Nodal numbers and element connectivity
        nodenrs = np.arange((1 + nelx) * (1 + nely)).reshape((1 + nely, 1 + nelx))  # (nely + 1, nelx + 1)
        edofVec = (2 * nodenrs[:-1, :-1].ravel()).reshape(nelx * nely, 1)  # (nely * nelx, 1)
        edofMat = edofVec + np.array(
            [(nelx + 1) * 2, (nelx + 1) * 2 + 1, (nelx + 1) * 2 + 2, (nelx + 1) * 2 + 3, 2, 3, 0, 1]
        )  # (nely * nelx, 8)
        self.edofMat = edofMat

        self.iK = np.kron(edofMat, np.ones((8, 1), dtype=np.int64)).ravel()  # ((nely * nelx * 8) * 8,)
        self.jK = np.kron(edofMat, np.ones((1, 8), dtype=np.int64)).ravel()  # ((nely * nelx) * (8 * 8),)

        # Load and boundary conditions
        lx, ly, lxy, ll = np.unstack(load, axis=-1)  # (#load,)
        self.F = csr_matrix(
            (ll, (ly * (nelx + 1) * 2 + lx * 2 + lxy, np.zeros(len(load), np.int64))), ((nely + 1) * (nelx + 1) * 2, 1)
        )  # ((nely + 1) * (nelx + 1) * 2, 1)

        bx, by, bxy = np.split(boundary_cond, 3, axis=-1)  # (#cond,)
        fixeddofs = by * (nelx + 1) * 2 + bx * 2 + bxy  # (#cond,)
        alldofs = np.arange((nely + 1) * (nelx + 1) * 2)
        assert fixeddofs.min() >= 0 and fixeddofs.max() < (nely + 1) * (nelx + 1) * 2, f"invalid fixeddofs {fixeddofs}"
        self.freedofs = np.setdiff1d(alldofs, fixeddofs)

    def get_sensitivity_mat(self, occ: np.ndarray) -> np.ndarray:
        """
        occ: (nely, nelx) [0, 1] occupancy

        returns:
        - SED_mat: (nely, nelx): U @ K @ U.T
        """
        assert occ.shape == (
            self.args.nely,
            self.args.nelx,
        ), f"input shape {occ.shape} != {(self.args.nely, self.args.nelx)}"

        sK = (self.KE.ravel() * (self.args.Emin + (occ.ravel()) * (self.args.E0 - self.args.Emin))[..., None]).flatten()
        K = coo_matrix((sK, (self.iK, self.jK))).tocsr()
        K = (K + K.T) / 2
        U = np.zeros((self.args.nely + 1) * (self.args.nelx + 1) * 2)  # (2 * (nely + 1) * (nelx + 1), 1)
        U[self.freedofs] = linalg.spsolve(K[self.freedofs[:, None], self.freedofs], self.F[self.freedofs])  # type: ignore

        # strain energy density
        return ((U[self.edofMat] @ self.KE) * U[self.edofMat]).sum(axis=-1).reshape(self.args.nely, self.args.nelx)

    def get_sensitivity(self, occ: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        no gradient

        returns:
        - sed: (nely, nelx)
        - sed_grad: (nely, nelx): d sum(sed) / d occ
        """
        occ = occ.detach()
        occ_p = occ**self.args.p
        sens_mat = torch.from_numpy(self.get_sensitivity_mat(occ_p.cpu().numpy())).to(device=occ.device)
        sed_grad = self.args.p * (occ ** (self.args.p - 1)) * (self.args.E0 - self.args.Emin) * sens_mat
        sed = (self.args.Emin + occ_p * (self.args.E0 - self.args.Emin)) * sens_mat
        return sed, sed_grad

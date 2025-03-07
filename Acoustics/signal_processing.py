import math
import random
from typing import Callable

import matplotlib.pyplot as plt
from torch import Tensor
import torch
from torch.nn import functional as f
import scipy.linalg as la
import numpy as np

from utils.utils import is_probability
from torch_framework.config import GlobalConfig


class NoStatisticsException(Exception):
    pass


def snr(audio: Tensor, vad: Tensor, allow_nan: bool = False) -> tuple[float, float]:
    assert len(audio.shape) == len(vad.shape) == 1, (audio.shape, vad.shape)
    assert audio.shape[0] == vad.shape[0] * GlobalConfig.win_size, (audio.shape, vad.shape, vad.shape[0] * GlobalConfig.win_size)

    x = audio.view(-1, GlobalConfig.win_size)
    N0, N1 = torch.count_nonzero(vad == 0.), torch.count_nonzero(vad == 1.)
    if (not allow_nan) and (not (N0 > 2 and N1 > 2)):
        raise NoStatisticsException()

    S = float(torch.var(x[vad == 1, :])) if N1 > 2 else math.nan
    N = float(torch.var(x[vad == 0, :])) if N0 > 2 else math.nan

    return S, N


def get_correlation_matrices(x: Tensor, vad: Tensor, lag: int = None) -> tuple[Tensor, Tensor]:
    assert is_probability(vad)

    if lag is None:  # Frequency domain
        assert len(x.shape) == 3 and len(vad.shape) == 2
        assert x.shape[-1] == vad.shape[-1], (x.shape, vad.shape)

        vad = vad.unsqueeze(1)
        Px, Pn = vad, (1 - vad)  # M, 1, T

        Nx = torch.sum(Px.detach()) / vad.shape[0]
        Nn = torch.sum(Pn.detach()) / vad.shape[0]
        if not (Nx > 1 and Nn > 1):
            raise NoStatisticsException(f"{Nx=}, {Nn=}")
        assert torch.isclose(Nx + Nn, torch.tensor(vad.shape[2], dtype=torch.float)), (Nx + Nn, vad.shape)
        # print(Nx, Nn, vad.shape[1], Nx + Nn)

        x = x - torch.mean(x, dim=2, keepdim=True)

        x, n = (Px * x), (Pn * x)

        # x = x - torch.mean(x, dim=2, keepdim=True)
        # n = n - torch.mean(n, dim=2, keepdim=True)

        x, n = x.swapdims(0, 1), n.swapdims(0, 1)  # F, M, T
        Rxx = x @ x.mH
        Rnn = n @ n.mH  # F, M, M

        # print(Rxx.shape, Rnn.shape, Nx.shape, Nn.shape)

        return Rxx / Nx, Rnn / Nn

    else:  # Time domain
        assert len(x.shape) == len(vad.shape) == 2
        assert lag > 0, lag
        M = x.shape[0]  # M, T'

        vad = vad.unsqueeze(-1)  # M, T, 1
        Px, Pn = vad, 1 - vad

        Nx = torch.sum(Px.detach()) / M
        Nn = torch.sum(Pn.detach()) / M
        if not (Nx > 1 and Nn > 1):
            raise NoStatisticsException(f"{Nx=}, {Nn=}")
        assert torch.isclose(Nx + Nn, vad.shape[1])
        # print(Nx, Nn, vad.shape[1], Nx + Nn)

        x = x.view(M, -1, GlobalConfig.win_size)  # M, T, C
        # print(Px.shape, x.shape)
        x, n = (Px * x).view(M, -1), (Pn * x).view(M, -1)  # M, T'

        x = x - torch.mean(x, dim=1, keepdim=True)
        n = n - torch.mean(n, dim=1, keepdim=True)

        Rxx, Rnn = torch.empty([M * lag, M * lag]), torch.empty([M * lag, M * lag])
        xi = xj = x
        ni = nj = n
        for li in range(lag):
            Rxx_ = xi @ xj.mT
            Rnn_ = ni @ nj.mT

            for lk in range(lag - li):
                Rxx[lk * M:(lk + 1) * M, (li + lk) * M: (li + lk + 1) * M] = Rxx_
                Rxx[(li + lk) * M:(li + lk + 1) * M, lk * M: (lk + 1) * M] = Rxx_
                Rnn[lk * M:(lk + 1) * M, (li + lk) * M: (li + lk + 1) * M] = Rnn_
                Rnn[(li + lk) * M:(li + lk + 1) * M, lk * M: (lk + 1) * M] = Rnn_

            xi, xj = xi[:, :-1], xj[:, 1:]
            ni, nj = ni[:, :-1], nj[:, 1:]

        # print(Rxx.shape, Rnn.shape, Nx.shape, Nn.shape)

        return Rxx / Nx, Rnn / Nn


def _mwf_numpy_implementation(Rxx: np.ndarray, Rnn: np.ndarray, ref_mic: int):
    F, M, _ = Rxx.shape

    Eval, V = np.empty((F, M), dtype=np.float_), np.empty((F, M, M), dtype=np.cfloat)

    # Compute optimal filter - non causal
    for f in range(F):
        Eval[f], V[f] = la.eigh(Rnn[f], Rxx[f], type=1)
        # print(lambd.dtype, Vf.dtype)
        # print(lambd)
        # Eval[f] = max(lambd)
    # print(Eval.shape, V.shape)

    ek = np.zeros((M, 1), dtype=np.cfloat)
    ek[ref_mic] = 1.
    w = ((Eval[:, :] - 1) * la.solve(V, ek)[:, 0, :]) * V[:, :, 0]
    return w


def _mwf_torch_implementation(Rxx: Tensor, Rnn: Tensor, ref_mic: int):
    F, M, _ = Rxx.shape

    # Compute optimal filter - non causal
    try:
        # Should be generalized eigenvalue decomposition, but no support for that in pytorch ):
        Eval, V = torch.linalg.eig(torch.linalg.solve(Rnn, Rxx))
        print(Eval)
    except RuntimeError as e:
        print(Rnn, Rxx)
        raise e
    # print(Eval.shape, V.shape)
    for fi in range(F):
        sorted_idx = torch.argsort(Eval[fi].abs(), descending=True)
        Eval[fi], V[fi] = Eval[fi, sorted_idx], V[fi, :, sorted_idx]
    # print(Eval.shape, V.shape)

    ek = torch.zeros(M, 1, dtype=torch.cfloat)
    ek[ref_mic] = 1.
    assert torch.all(Eval[:, 0].abs() > 1.), Eval[:, 0]
    print(Eval)
    w = (((Eval[:, 0:1].abs() - 1.) * Eval[:, 0:1] / Eval[:, 0:1].abs()) * torch.linalg.solve(V, ek)[:, 0, :]) * V[:, :, 0]
    # w = torch.linalg.solve(Rxx, Rxx - Rnn)[:, :, ref_mic]
    # print(x.shape, w.shape)
    return w


def frequency_domain_mwf(x: Tensor, vad: Tensor, ref_mic: int):
    assert len(x.shape) == 3, x.shape
    M, F, T = x.shape
    assert 0 <= ref_mic < M, (ref_mic, x.shape)
    # print(x.shape, x.dtype)

    # Get correlation matrices
    Rxx, Rnn = get_correlation_matrices(x, vad)

    # plt.figure()
    # plt.title("Spatio-Temporal Noise Covariance Matrix")
    # plt.imshow(Rnn.detach().numpy(), interpolation='none')
    # plt.show()

    # w = torch.tensor(_mwf_numpy_implementation(Rxx.detach().numpy(), Rnn.detach().numpy(), ref_mic))
    # w = _mwf_torch_implementation(Rxx, Rnn, ref_mic)
    w = torch.linalg.solve(Rxx, Rxx - Rnn)[:, :, ref_mic]

    # w: (F, M) -> (F, M, 1); F groups
    w = w.view(F, M, 1)
    # x: (M, F, T) -> (F, M, T) -> (1, F*M, T)
    x = x.swapdims(0, 1).reshape(1, M * F, T)

    # Frequency domain -> filtering = depth wise convolution
    x = f.conv1d(x, w.conj(), groups=F)
    return x


def LCMV_beamformer(x: Tensor, vad: Tensor, A: Tensor, b: Tensor):
    assert len(x.shape) == 3 and len(vad.shape) == 2
    M, F, T = x.shape

    # 1) Get statistics
    Rxx, Rnn = get_correlation_matrices(x, vad)
    # print(Rnn.shape, A.shape, b.shape)

    # fig, axes = plt.subplots(2, 2)
    # fig.suptitle("Spatio-Temporal Noise Covariance Matrix")
    # indices = random.choices(range(Rnn.shape[0]), k=4)
    # for i in range(2):
    #     for j in range(2):
    #         k = indices[2*i + j]
    #         axes[i][j].imshow(Rnn[k].detach().abs().numpy(), interpolation='none')
    #         axes[i][j].set_title(f"{k=}")
    # plt.show()

    # 2) Compute LCMV filter
    C = torch.linalg.solve(Rnn, A)
    w = C @ torch.linalg.solve(A.mH @ C, b)  # F, M, 1
    # print(x.shape, w.shape)
    # x: (M, F, T) -> (F, M, T) -> (1, F*M, T)
    x = x.swapdims(0, 1).reshape(1, M * F, T)

    # 3) Apply filter
    return f.conv1d(x, w.conj(), groups=F)

import torch
from torch import Tensor
from torch.nn import Module, Sequential
from torch.nn import MSELoss
from torch.nn import functional as f

from abc import ABC, abstractmethod

from torch_framework.models.decoders import FinalLayer
from torch_framework.models.abstract_models import CoreNetwork, Encoder
import utils.utils as utils
from torch_framework.multi_channel_dataset import align_labels


class Model(ABC):
    enc: Module
    core: Module
    ff: Module

    model: Module

    @abstractmethod
    def apply_loss(self, y_pred, y_true, **kwargs) -> [Tensor, Tensor]:
        ...

    def finalize_output(self, y_pred):
        return y_pred


class VADModel(Model):
    def __init__(self, encoder: Encoder, core_network: CoreNetwork, gamma: float = 0., hidden: list[int] = None):
        self.enc = encoder
        self.core = core_network
        self.ff = FinalLayer(f.sigmoid, core_network.out_features, hidden=hidden)

        self.model = Sequential(self.enc, self.core, self.ff)

        self.loss = FocalLoss(.5, gamma)

    def apply_loss(self, y_pred, y_true, **kwargs):
        return self.loss(y_pred, y_true)


class EnergyModel(Model):
    def __init__(self, encoder: Encoder, core_network: CoreNetwork, hidden: list[int]):
        self.model = Sequential(encoder, core_network, FinalLayer(lambda x: x, core_network.out_features, hidden=hidden))
        self.loss = MSELoss()

        # from vad_labeling.simpleVAD import EnergyVAD
        # self.vad = EnergyVAD(causal=True)

    def apply_loss(self, y_pred, y_true, **meta):
        return self.loss(y_pred, torch.log(y_true + 1e-6))


class FocalLoss:
    def __init__(self, alpha: float = .5, gamma: float = 0., eps=1e-8):
        assert 0. < alpha < 1.
        self.alpha, self.gamma = torch.tensor(alpha, dtype=torch.float), torch.tensor(gamma, dtype=torch.float)
        self.eps = torch.tensor(eps, dtype=torch.float)

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        assert utils.is_probability(y_pred), (f"finite: {torch.all(torch.isfinite(y_pred))}, "
                                              f"pos: {torch.all(0 <= y_pred)}, <1: {torch.all(y_pred <= 1)}\n"
                                              f"{y_pred}")
        assert y_pred.shape == y_true.shape, (y_pred.shape, y_true.shape)
        assert len(y_pred.shape) == 2, y_pred.shape

        # print(y_pred.shape, y_true.shape, eps)
        # y_pred = torch.clip(y_pred, self.eps, 1. - self.eps)
        loss = y_true * (1 - self.alpha) * (1 - y_pred) ** self.gamma * torch.log(y_pred + self.eps) + \
            (1 - y_true) * self.alpha * y_pred ** self.gamma * torch.log(1 - y_pred + self.eps)
        # print(loss.shape)
        loss = -torch.mean(loss, dim=1)
        loss = torch.mean(loss, dim=0)
        assert len(loss.shape) == 0
        return loss

    def __repr__(self):
        return f"FocalLoss(alpha={self.alpha}, gamma={self.gamma})"


class FancyLossFocusingOnTheHardSamples:
    def __init__(self, alpha: float = .5, gamma: float = 0., eps: float = 1e-8):
        assert 0. < alpha < 1.
        assert eps > 0
        self.alpha, self.gamma = torch.tensor(alpha, dtype=torch.float), torch.tensor(gamma, dtype=torch.float)
        self.eps = torch.tensor(eps, dtype=torch.float)

        self.normalization = -.5 * (.5 ** gamma) * math.log(.5)
        # print(self.normalization)

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        assert len(y_pred.shape) == 1
        assert utils.is_probability(y_pred), \
            f"finite: {torch.all(torch.isfinite(y_pred))}, pos: {torch.all(0 <= y_pred)}, <1: {torch.all(y_pred <= 1)}"
        assert y_pred.shape == y_true.shape

        loss = y_true * (1 - self.alpha) * (1 - y_pred) ** self.gamma * torch.log(y_pred + self.eps) + \
            (1 - y_true) * self.alpha * y_pred ** self.gamma * torch.log(1 - y_pred + self.eps)
        loss = -torch.mean(loss, dim=-1) / self.normalization

        scale = loss.detach()

        return scale * loss

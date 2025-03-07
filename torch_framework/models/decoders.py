import torch
from torch import Tensor
from torch.nn import Module, ParameterList
from torch import nn
from torch.nn import functional as f

from typing import Literal, Callable, Sequence

from torch_framework.models.custom_layers import ConvTranspose1d, InstanceNorm1d, Norm1d

from torch_framework.config import GlobalConfig
from utils.utils import isPow2


class FinalLayer(Module):
    def __init__(self, act_fn: Callable[[Tensor], Tensor], in_features: int = GlobalConfig.win_size,
                 out_features: int = 1, hidden: list[int] = None):
        Module.__init__(self)
        if hidden is None:
            hidden = []
        hidden.append(out_features)

        self.norm = InstanceNorm1d(in_features)

        self.fc = nn.ParameterList()
        chin = in_features
        for chout in hidden:
            self.fc.append(nn.Conv1d(chin, chout, 1))
            chin = chout
        self.act_fn = act_fn

    def forward(self, x: Tensor):
        assert len(x.shape) == 3, x.shape

        x = self.norm(x)
        for layer in self.fc:
            x = self.act_fn(layer(x))

        return x.squeeze(1)


class ConvDecoder(Module):
    def __init__(self, kernel_sizes: Sequence[int], strides: Sequence[int], chin: int = GlobalConfig.win_size,
                 act_fn: Callable[[], Module] = nn.Identity):
        assert len(kernel_sizes) == len(strides) and len(kernel_sizes) > 0
        assert all(isPow2(s) for s in strides)
        assert all(0 < s <= k for s, k in zip(strides, kernel_sizes))
        Module.__init__(self)

        self.kernel_sizes = kernel_sizes
        self.strides = strides

        self.norm = ParameterList([])
        self.conv = ParameterList([])
        self.act_fn = ParameterList([])

        for k, s in zip(kernel_sizes, strides):
            self.norm.append(Norm1d.bn(chin, affine=False))
            assert chin % s == 0, (chin, s)
            self.conv.append(ConvTranspose1d(in_channels=chin, out_channels=chin // s, kernel_size=k, stride=s,
                                             groups=chin // s, padding='causal'))
            self.act_fn.append(act_fn())
            chin //= s
        assert chin == 1, (chin, strides, kernel_sizes)

    def __repr__(self):
        return f"ConvDecoder(kernel_sizes={self.kernel_sizes}, strides={self.strides}, act_fn={self.act_fn[0]})"

    def forward(self, x: Tensor):
        for norm, conv, act_fn in zip(self.norm, self.conv, self.act_fn):
            x = act_fn(conv(norm(x)))

        return x.squeeze(1)

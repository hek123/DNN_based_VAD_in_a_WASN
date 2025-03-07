import math
from typing import Union, Sequence, Callable, ClassVar, Final, Optional, List, Literal

import matplotlib.pyplot as plt
import torch
import torchaudio.functional
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as f
import torchaudio.functional as audio

from utils.utils import is_probability


class Conv1d(nn.Conv1d):
    __doc__ = "Extend Conv1d to support causal padding\n" + nn.Conv1d.__doc__

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int | tuple[int], stride: int | tuple[int] = 1, dilation: int | tuple[int] = 1,
                 padding: str | int | tuple[int] = 0, padding_mode: str = 'zeros',
                 bias: bool = True,
                 groups: int = 1,
                 weight_activation: nn.Module | Callable[[Tensor], Tensor] = nn.Identity(),
                 device=None, dtype=None):
        self._causal = (padding == 'causal')
        if self._causal:
            self.front_pad = kernel_size - stride
            assert self.front_pad >= 0, f"Causal padding only supported for overlapping convolution; {kernel_size=}, {stride=}"
            padding = 0
            assert padding_mode == 'zeros', padding_mode
            # assert padding % stride == 0, (padding, stride)
            # self.over_pad = padding // stride
        nn.Conv1d.__init__(self, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           stride=stride, dilation=dilation, bias=bias, groups=groups, padding=padding,
                           padding_mode=padding_mode, device=device, dtype=dtype)
        self._weight_activation = weight_activation

    def forward(self, x: Tensor) -> Tensor:
        if self._causal:
            x = f.pad(x, (self.front_pad, 0))
        return nn.Conv1d._conv_forward(self, x, self._weight_activation(self.weight), self.bias)
        # return x[:, :, :-self.over_pad] if self._causal else x

    def __repr__(self):
        if self._causal:
            p = self.padding
            self.padding = 'causal'
            out = nn.Conv1d.__repr__(self)
            self.padding = p
            return out
        return nn.Conv1d.__repr__(self)


class ConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1,
                 padding: int | str = 0, output_padding: int = 0,
                 groups: int = 1,
                 bias: bool = True,
                 dilation: int = 1,
                 padding_mode: str = 'zeros',
                 device=None, dtype=None
                 ):
        self._causal = False
        if padding == 'causal':
            # padding = kernel_size - 1
            self._causal = padding != 0
            self._over_pad = (kernel_size - stride)
            padding = 0
        nn.ConvTranspose1d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding,
                                    groups, bias, dilation, padding_mode, device, dtype)

    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        x = nn.ConvTranspose1d.forward(self, input, output_size)
        return x[:, :, self._over_pad:] if self._causal else x


class FFTConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int | tuple[int], stride: int | tuple[int] = 1, dilation: int | tuple[int] = 1,
                 padding: str | int | tuple[int] = 0, padding_mode: str = 'zeros',
                 bias: bool = True,
                 groups: int = 1,
                 device=None, dtype=None):
        nn.Module.__init__(self)
        assert in_channels > 0 and out_channels > 0
        assert stride == 1
        assert dilation == 1
        assert padding_mode == 'zeros'
        assert bias == False
        assert groups == 1
        assert padding in ('same', 'full', 'valid')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.padding = padding

        sigma = math.sqrt(2 / (in_channels * kernel_size + out_channels))
        self.W = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size) * sigma)

    def forward(self, x: Tensor) -> Tensor:
        B, Cin, T = x.shape
        y = torch.empty(B, self.out_channels, self.out_size(T))
        for chout in range(self.out_channels):
            y[:, chout, :] = torch.sum(
                torchaudio.functional.fftconvolve(x, self.W[None, chout, :, :], mode=self.padding), dim=1
            )
        return y

    def out_size(self, in_size: int):
        match self.padding:
            case 'same' | 'causal':
                return in_size
            case 'valid':
                return max(in_size, self.kernel_size) - min(in_size, self.kernel_size) + 1
            case 'full':
                return in_size + self.kernel_size - 1
            case _:
                raise AssertionError(self.padding)

    def __repr__(self):
        return f"FFTConvolve()"


class ExponentialMovingAverage(nn.Module):
    """
    Implements the running mean, defined as:
        y[t] = (1 - momentum) * y[t-1] + momentum * x[t]

    input: (..., time)
    output: (..., time) or (..., num_filters, time)
    """

    _inits: Final = ('zero', 'unbiased', 'reflect')
    min_momentum: Final = 1e-6

    def __init__(self, momentum: float | list[float] | None, trainable: bool = False,
                 initialization: str = 'unbiased', causal: bool = True):
        """
        :param momentum: The momentum used in the EMA. If momentum is `None`, this layer turns into a no-op.
        :param initialization: How to initialize the EMA:
            - `zero`: Initialize with zero. This creates an initial bias towards zero
            - `unbiased`: Use the unbiased EMA, which takes into account that there is no information before time zero.
            - `reflect`: Initialize with the average by reflecting the first `1 / momentum` samples. This makes that the
                output of the EMA for the first `1/momentum` samples will be non-causal.
        :param trainable: Set to True if `momentum` should be a trainable parameter
        :param causal: If False, apply the EMA filter forward and backward
        """
        nn.Module.__init__(self)
        assert initialization in self._inits, initialization

        if momentum is None:
            self._momentum = None
            self.unbiased = True
            self.initialization = 'unbiased'
            self.causal = True
            self.trainable = False
        else:
            assert isinstance(momentum, float) or isinstance(momentum, list), type(momentum)
            self._momentum = torch.tensor(momentum, dtype=torch.float)
            self._check_momentum(self._momentum)

            self.trainable = trainable
            if trainable:
                # Inverse sigmoid function
                self._momentum = nn.Parameter(- torch.log(1. / self._momentum - 1.))
            else:
                self.a, self.b = self._get_coeff(self._momentum)

            self.causal = causal
            self.initialization = initialization
            if initialization == 'unbiased' and not self.trainable:
                # cache the correction, because is always the same
                self.correction = self._bias_correction(0, self.momentum.detach())  # shape: (num_filters, time)
            elif initialization == 'reflect':
                self.T60 = math.ceil(1. / torch.min(self.momentum).item())
                # print(self.T60)

    @property
    def momentum(self):
        return f.sigmoid(self._momentum) if self.trainable else self._momentum

    @classmethod
    def _check_momentum(cls, momentum: Tensor):
        assert len(momentum.shape) == 0 or len(momentum.shape) == 1
        assert (0 < momentum).all() and (momentum <= 1).all(), momentum
        torch.clip_(momentum, cls.min_momentum, None)

    @staticmethod
    def _get_coeff(momentum: Tensor):
        a = torch.stack([torch.ones_like(momentum), momentum - 1.], dim=-1)
        b = torch.stack([momentum, torch.zeros_like(momentum)], dim=-1)
        return a, b

    def __repr__(self):
        return (f"RunningMean({self._momentum}, init={self.initialization}, trainable={self.trainable}, "
                f"causal={self.causal})")

    def _lfilter(self, x):
        if self.initialization == 'reflect':
            num_samples = min(self.T60, x.shape[-1])
            x = torch.cat([x[..., :num_samples].flip(-1), x], dim=-1)
            x = audio.lfilter(x, self.a, self.b, batching=True, clamp=False)
            return x[..., num_samples:]
        elif self.initialization == 'unbiased':
            if self.correction.shape[1] < x.shape[-1]:
                # Recompute if cached correction is not long enough
                self.correction = self._bias_correction(x.shape[-1], self.momentum)
            x = audio.lfilter(x, self.a, self.b, batching=True, clamp=False)
            # print(x.shape, self.correction.shape)
            return x * self.correction[:, :x.shape[-1]]
        elif self.initialization == 'zeros':
            return audio.lfilter(x, self.a, self.b, batching=True, clamp=False)
        else:
            raise AssertionError(self.initialization)

    def forward(self, x):
        if self.momentum is None:
            return x

        if self.trainable and self.training:
            self.a, self.b = self._get_coeff(self.momentum)
        if self.initialization == 'unbiased' and \
                ((self.trainable and self.training) or self.correction.shape[1] < x.shape[-1]):
            self.correction = self._bias_correction(x.shape[-1], self.momentum)

        if self.causal:
            x_original = x
            x = self._lfilter(x)

            assert torch.all(torch.isfinite(x)), str(self)

            # plt.figure()
            # idx = [i for i in [0, 10, 100] if i < x.shape[1]]
            # plt.plot(x_original[0, idx, :].mT.detach().numpy())
            # plt.plot(x[0, idx, :].mT.detach().numpy())
            # plt.show()

            return x
        else:
            x = self._lfilter(x)
            return self._lfilter(x.flip(-1)).flip(-1)

    @staticmethod
    def _bias_correction(num_samples: int, momentum: Tensor) -> Tensor:
        # x.shape: (..., num_filters, time) or (..., time)
        power = torch.arange(1, 1 + num_samples)
        correction = torch.pow((1. - momentum).view(-1, 1), power.view(1, -1))  # (num_filters, time)
        return 1. / (1. - correction)


class Integrator(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.a = torch.tensor([1., -1.])
        self.b = torch.tensor([1., 0.])

    def forward(self, x):
        return torchaudio.functional.lfilter(x, self.a, self.b, clamp=False)


class EMACell(nn.Module):
    """
    Implements the running mean, defined as:
        y[t] = (1 - momentum) * y[t-1] + momentum * x[t]

    input: (...)
    output: (...)
    """

    def __init__(self, momentum: float, unbiased: bool = True):
        assert isinstance(momentum, float), type(momentum)
        assert 0. < momentum <= 1.

        nn.Module.__init__(self)

        self.momentum = torch.tensor(momentum, dtype=torch.float)
        self.momentum_1 = 1. - self.momentum
        self.unbiased = unbiased

        self.state = None
        self.correction = 1.

    def reset_state(self):
        self.state = None
        self.correction = 1.

    def forward(self, x):
        if self.state is None:
            self.state = torch.zeros_like(x)

        # Exponential Moving Average
        self.state = self.momentum_1 * self.state + self.momentum * x

        # Bias correction
        self.correction *= self.momentum_1
        return self.state / (1. - self.correction)


class TimeSeriesNorm(nn.Module):
    def __init__(self, feature_dim: int, momentum: float = .05, affine: bool = False,
                 eps: float = 1e-6, adaptive: bool = False, trainable_momentum: bool = False):
        nn.Module.__init__(self)
        assert 0. < momentum <= 1., momentum
        assert eps > 0.

        self._feature_dim = feature_dim
        self._stat_dims = 1
        self._momentum = momentum
        self._ema = ExponentialMovingAverage(momentum, trainable=trainable_momentum, initialization='reflect')
        self._adaptive = adaptive
        self._affine = affine

        if adaptive:
            self.gamma = nn.Conv1d(feature_dim, feature_dim, 1, groups=feature_dim) if affine else \
                nn.Conv1d(1, feature_dim, 1)
            nn.init.zeros_(self.gamma.weight)
            nn.init.ones_(self.gamma.bias)
            self.beta = nn.Conv1d(feature_dim, feature_dim, 1, groups=feature_dim) if affine else \
                nn.Conv1d(1, feature_dim, 1)
            nn.init.zeros_(self.beta.weight)
            nn.init.zeros_(self.beta.bias)
        else:
            self.gamma = nn.Parameter(torch.ones([1, feature_dim, 1]))
            self.beta = nn.Parameter(torch.zeros([1, feature_dim, 1]))

        self.eps = torch.tensor(eps)

    def forward(self, x):
        # x.shape == (B, F, T)
        mean = self._ema(x) if self._affine else \
            self._ema(torch.mean(x, dim=1, keepdim=True))
        x = x - mean

        var = self._ema(torch.square(x)) if self._affine else \
            self._ema(torch.mean(torch.square(x), dim=1, keepdim=True))
        denom = torch.sqrt(var + self.eps)
        x = x / denom

        return (self.gamma(denom.detach()) * x + self.beta(mean.detach())) if self._adaptive else \
            (self.gamma * x + self.beta)

    def __repr__(self):
        return (f"TimeSeriesNorm({self._feature_dim}, momentum={self._ema.momentum:.2e}, "
                f"adaptive={self._adaptive}, affine={self._affine})")


class AdaptiveNormalization(nn.Module):
    def __init__(self, feature_dim: int, momentum: float, eps: float = 1e-6):
        nn.Module.__init__(self)
        assert eps > 0.
        assert 0. < momentum <= 1.

        self._feature_dim = feature_dim
        # shape = (1, feature_dim, 1)
        self.momentum = momentum
        self.eps = torch.tensor([eps])

        self.a2shift = nn.Conv1d(feature_dim, 1, 1, bias=False)
        # nn.init.eye_(self.a2shift.weight[:, :, 0])
        nn.init.constant_(self.a2shift.weight, 1 / feature_dim)
        self.b2scale = Conv1d(feature_dim, 1, 1, bias=False, weight_activation=torch.exp)
        # nn.init.eye_(self.b2scale.weight[:, :, 0])
        nn.init.constant_(self.b2scale.weight, math.log(1 / feature_dim))
        # self.c2gate = nn.Conv1d(feature_dim, feature_dim, 1)

        self.proj = nn.Conv1d(feature_dim, feature_dim, 1, groups=feature_dim)
        nn.init.ones_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        # print(x.shape)  # (B, F, T)

        # 1) Adaptive Shifting
        a = running_mean(x.detach(), self.momentum)
        x = x - self.a2shift(a)

        # 2) Adaptive Scaling
        # b = 1 / torch.sqrt(running_mean(torch.square(x.detach()), self.momentum) + self.eps)
        # x = x * self.b2scale(b)
        b = running_mean(torch.square(x.detach()), self.momentum)
        x = x / torch.sqrt(self.b2scale(b) + self.eps)

        # # 3) Adaptive Gating
        # c = running_mean(x, self.momentum)
        # gamma = f.sigmoid(self.c2gate(c))
        # x = x * gamma

        return self.proj(x)


class InstanceNorm1d(nn.Module):
    _USE_EMA: ClassVar[str] = None
    _EMA_INIT: ClassVar[str | None] = None

    @classmethod
    def set_mode(cls, use_ema: Literal['false', 'true', 'inference'],
                 ema_init: Literal['zero', 'unbiased', 'reflect'] = None):
        assert cls._USE_EMA is None
        cls._USE_EMA = use_ema
        cls._EMA_INIT = ema_init

    def __init__(self, num_features: int, eps: float = 1e-6, momentum: float = 0.01, affine: bool = True):
        nn.Module.__init__(self)
        assert self._USE_EMA in ('false', 'true', 'inference')

        self.features = num_features
        shape = (1, num_features, 1)
        self.gamma = nn.Parameter(torch.ones(shape), requires_grad=affine)
        self.beta = nn.Parameter(torch.zeros(shape), requires_grad=affine)

        if self._USE_EMA == 'true' or self._USE_EMA == 'inference':
            self._ema_mean = ExponentialMovingAverage(momentum, initialization=self._EMA_INIT)
            self._ema_var = ExponentialMovingAverage(momentum ** 2, initialization=self._EMA_INIT)
        else:
            self.register_parameter('_ema_mean', None)
            self.register_parameter('_ema_var', None)

        self.eps = torch.tensor(eps)

    def forward(self, x):
        assert len(x.shape) == 3, x.shape
        assert x.shape[1] == self.features, (x.shape, self.features)

        if self._USE_EMA == 'false' or (self._USE_EMA == 'inference' and self.training):
            # B, F, T
            var, mean = torch.var_mean(x, dim=2, keepdim=True, unbiased=False)  # B, F, 1
            x_new = (x - mean) * torch.rsqrt(var + self.eps)

            # mean_ = running_mean(x.detach(), self.momentum)
            # var_ = running_mean(torch.square(x.detach() - mean), self.momentum ** 2)
            #
            # fig, (ax3, ax1, ax2) = plt.subplots(3, 1, sharex='all')
            # ax3.plot(x[0, [0, 10, 100], :].detach().mT.numpy())
            # ax1.plot(mean[0, [0, 10, 100], :].expand((-1, x.shape[-1])).detach().mT.numpy())
            # ax1.plot(mean_[0, [0, 10, 100], :].detach().mT.numpy())
            # ax2.plot(var[0, [0, 10, 100], :].expand((-1, x.shape[-1])).detach().mT.numpy())
            # ax2.plot(var_[0, [0, 10, 100], :].detach().mT.numpy())
            # plt.show()
        elif self._USE_EMA == 'true' or (self._USE_EMA == 'inference' and not self.training):
            # assert self.ema_train or not x.requires_grad
            mean = torch.complex(self._ema_mean(x.real), self._ema_mean(x.imag)) if x.is_complex() else \
                self._ema_mean(x)
            x = x - mean
            var = self._ema_var((x * x.conj()).real) if x.is_complex() else \
                self._ema_var(torch.square(x))
            x_new = x * torch.rsqrt(var + self.eps)
        else:
            raise AssertionError(f"{self._USE_EMA, self.training}")

        return self.gamma * x_new + self.beta

    def __repr__(self):
        return f"InstanceNorm1d({self.features}, affine={self.gamma.requires_grad}, causal={self._USE_EMA}, ema={self._ema_mean})"


class FilterResponseNorm(nn.Module):
    """
    Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks
    """

    _USE_EMA: ClassVar[str] = None
    _EMA_INIT: ClassVar[str | None] = None

    @classmethod
    def set_mode(cls, use_ema: Literal['false', 'true', 'inference'],
                 ema_init: Literal['zero', 'unbiased', 'reflect'] = None):
        assert cls._USE_EMA is None
        cls._USE_EMA = use_ema
        cls._EMA_INIT = ema_init

    def __init__(self, num_features: int, affine: bool = True, momentum: float = .01, eps: float = 1e-6):
        assert self._USE_EMA in ('false', 'true', 'inference')
        assert 0 < eps, eps
        nn.Module.__init__(self)

        self.features = num_features

        shape = (1, num_features, 1)
        self.gamma = nn.Parameter(torch.ones(shape), requires_grad=affine)
        self.beta = nn.Parameter(torch.zeros(shape), requires_grad=affine)
        self.tau = nn.Parameter(torch.zeros(shape))

        if self._USE_EMA == 'true' or self._USE_EMA == 'inference':
            self._ema_var = ExponentialMovingAverage(momentum ** 2, initialization=self._EMA_INIT)
        else:
            self.register_parameter('_ema_var', None)

        self.eps = torch.tensor(eps)

    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 3, x.shape
        assert x.shape[1] == self.features, (x.shape, self.features)

        v = (x * x.conj()).real if x.is_complex() else torch.square(x)
        if self._USE_EMA == 'false' or (self._USE_EMA == 'inference' and self.training):
            v = torch.mean(v, dim=2, keepdim=True)
        elif self._USE_EMA == 'true' or (self._USE_EMA == 'inference' and not self.training):
            v = self._ema_var(v)
        else:
            raise AssertionError(f"{self._USE_EMA, self.training}")

        # var_ = running_mean(torch.square(x.detach() - mean), self.momentum ** 2)
        #
        # fig, (ax3, ax2) = plt.subplots(2, 1, sharex='all')
        # ax3.plot(x[0, [0, 10, 100], :].detach().mT.numpy())
        # ax2.plot(var[0, [0, 10, 100], :].expand((-1, x.shape[-1])).detach().mT.numpy())
        # ax2.plot(var_[0, [0, 10, 100], :].detach().mT.numpy())
        # plt.show()

        x = x * torch.rsqrt(v + self.eps)

        return torch.maximum(self.gamma * x + self.beta, self.tau)

    def __repr__(self):
        return f"FRN({self.features}, affine={self.gamma.requires_grad}, causal={self._USE_EMA}, ema={self._ema_var})"


def gated_running_mean(x: Tensor, gate: Tensor, unbiased: bool = False) -> Tensor:
    """
    Gated running mean -> momentum is time varying:
        y[t] = (1 - gate[t]) * y[t-1] + gate[t] * x[t]
    :param x: (..., T)
    :param gate: (..., T)
    :param unbiased: If true the unbiased EMA is used, else EMA is initialized with 0; default: False
    :return: (..., T)
    """
    assert is_probability(gate)

    y = torch.empty_like(x)
    biased_ema = torch.zeros(x.shape[:-1])
    correction = torch.ones(gate.shape[:-1])
    for t in range(x.shape[-1]):
        biased_ema = (1. - gate[..., t]) * biased_ema + gate[..., t] * x[..., t]
        if unbiased:
            correction *= 1. - gate[..., t].detach()
            y[..., t] = biased_ema / (1. - correction)
        else:
            y[..., t] = biased_ema
    return y


class GatedTSNorm(nn.Module):
    def __init__(self, feature_dim: int, momentum: float = .05, eps: float = 1e-6):
        assert 0 < momentum <= 1, momentum
        nn.Module.__init__(self)

        self.Wa = Conv1d(feature_dim, 1, 1, weight_activation=nn.Softmax(dim=1), bias=False)
        nn.init.zeros_(self.Wa.weight)
        self.Wb = Conv1d(feature_dim, 1, 1, weight_activation=nn.Softmax(dim=1), bias=False)
        nn.init.zeros_(self.Wb.weight)

        self.Wo = nn.Conv1d(feature_dim, feature_dim, kernel_size=1, groups=feature_dim)
        nn.init.ones_(self.Wo.weight)
        nn.init.zeros_(self.Wo.bias)

        self.momentum = torch.tensor(momentum)
        self.eps = torch.tensor(eps)

    def forward(self, x, g):
        assert is_probability(g)
        g = g * self.momentum

        mean = self.Wa(gated_running_mean(x.detach(), g))
        x = x - mean
        var = self.Wb(gated_running_mean(torch.square(x.detach()), g))
        x = x / torch.sqrt(var + self.eps)

        return self.Wo(x)


def skippy(layer: nn.Module, x: Tensor) -> Tensor:
    return x + layer(x)


class TDLayer(nn.Module):
    """
    Implements the Time Delay network
    """

    def __init__(self, in_size: int, kernel_size: int, in_channels: int = 1, out_channels: int = 1, strides: int = 1,
                 padding: int | str = 0, use_bias: bool = True, dtype=None):
        nn.Module.__init__(self)
        for param in (in_size, in_channels, out_channels, kernel_size, strides):
            assert isinstance(param, int) and param >= 1

        self.in_size, self.Cin, self.Cout = in_size, in_channels, out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding: int
        if isinstance(padding, int):
            self.padding = padding
        elif padding == 'valid':
            self.padding = 0
        elif padding == 'same':
            assert kernel_size % 2 == 1, f"padding=`same` only supported for odd kernel sizes"
            assert strides == 1, f"padding=`same` only supported for strides=1"
            self.padding = (kernel_size - 1) // 2
        elif padding == 'full':
            assert strides == 1, f"padding=`full` only supported for strides=1"
            self.padding = kernel_size - 1
        self.L_noStride = in_size + 2 * self.padding - (kernel_size - 1)
        assert (self.L_noStride - 1) % strides == 0, f"{in_size=}, {padding=}, {kernel_size=}, {strides=}"
        self.out_size = (self.L_noStride - 1) // strides + 1

        # He initialization
        self.weights = nn.Parameter(torch.randn(self.out_size, out_channels, in_channels * kernel_size, dtype=dtype) /
                                    torch.sqrt(torch.tensor(kernel_size * in_channels / 2)))
        self.bias = nn.Parameter(torch.zeros(self.out_size, out_channels, dtype=dtype)) if use_bias else None

    def __repr__(self):
        return (
            f"TDLayer({self.in_size}, in_channels={self.Cin}, out_channels={self.Cout}, kernel_size={self.kernel_size}, "
            f"strides={self.strides}, padding={self.padding}) -> {self.out_size}")

    def forward(self, x: Tensor):
        """
        :param x: (..., T, Cin)
        :return:
        """
        Lin, Cin = x.shape[-2:]
        assert Cin == self.Cin, (x.shape, Cin, self.Cin)
        assert Lin == self.in_size, (x.shape, Lin, self.in_size)

        # print(x.shape)
        x = f.pad(x, (0, 0, self.padding, self.padding))
        # print(x.shape)  # (..., Lin + 2*padding, Cin)

        x = x.unfold(size=self.kernel_size, step=self.strides, dimension=-2)
        # print(x.shape)  # (..., Lout, Cin, K)

        # (Lout, Cout, Cin*K) @ (Lout, Cin*K, 1) -> (Lout, Cout, 1)
        x = self.weights @ x.reshape(*x.shape[:-2], -1, 1)
        x = x.squeeze(-1)

        if self.bias is not None:
            x = x + self.bias

        return x


def squeezy(layer: Callable[[Tensor], Tensor], dim: int, x: Tensor) -> Tensor:
    return layer(x.unsqueeze(dim)).squeeze(dim)


def swappy(layer: Callable[[Tensor], Tensor], dim1: int, dim2: int, x: Tensor) -> Tensor:
    return layer(x.swapdims(dim1, dim2)).swapdims(dim1, dim2)


class TransposeTDLayer(nn.Module):
    """
    Implements the Transpose Time Delay network
    """

    def __init__(self, in_size: int, in_channels: int, out_channels: int, kernel_size: int, strides: int = 1,
                 use_bias: bool = True, dtype=None):
        nn.Module.__init__(self)

        self.Lin, self.Cin, self.Cout = in_size, in_channels, out_channels
        self.K = kernel_size
        self.strides = strides
        self.out_size = (in_size - 1) * strides + kernel_size

        self.weights = nn.Parameter(torch.randn(kernel_size, self.Lin, out_channels, in_channels, dtype=dtype))
        self.bias = nn.Parameter(torch.randn(self.out_size, out_channels, dtype=dtype)) if use_bias else None

    def __repr__(self):
        return (f"TransposeTDLayer({self.Lin}, in_channels={self.Cin}, out_channels={self.Cout}, kernel_size={self.K}, "
                f"strides={self.strides})")

    def forward(self, x):
        """
        :param x: (..., T, Cin)
        :return:
        """

        Lin, Cin = x.shape[-2:]
        assert Cin == self.Cin, (x.shape, self.Cin)
        assert Lin == self.Lin, (x.shape, self.Lin)
        x = torch.unsqueeze(x, -1)
        # print(x.shape)

        y = torch.zeros((*x.shape[:-3], self.out_size, self.Cout, 1))
        # print(f"{y.shape}")
        # print(y[..., 0:self.Lin * self.strides + 0:self.strides, :, :].shape)
        # print((self.weights[0] @ x).shape)
        y[..., 0:self.Lin * self.strides + 0:self.strides, :, :] = self.weights[0] @ x
        for dt in range(1, self.K):
            y[..., dt:self.Lin * self.strides + dt:self.strides, :, :] += self.weights[dt] @ x
        y = y.squeeze(-1)

        if self.bias is not None:
            # print(y.shape, self.bias.shape)
            y = y + self.bias

        return y


class PWTDConv1d(nn.Module):
    """
    Implements the Time Delay network
    """

    def __init__(self, in_channels: int, kernel_size: int, strides: int = 1,
                 padding: int | str = 0, use_bias: bool = True, dtype=None):
        nn.Module.__init__(self)
        for param in (in_channels, kernel_size, strides):
            assert isinstance(param, int) and param >= 1

        self.in_size = in_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding: int
        if isinstance(padding, int):
            self.padding = padding
        elif padding == 'valid':
            self.padding = 0
        elif padding == 'same':
            assert kernel_size % 2 == 1, f"padding=`same` only supported for odd kernel sizes"
            assert strides == 1, f"padding=`same` only supported for strides=1"
            self.padding = (kernel_size - 1) // 2
        elif padding == 'full':
            assert strides == 1, f"padding=`full` only supported for strides=1"
            self.padding = kernel_size - 1
        self.L_noStride = in_channels + 2 * self.padding - (kernel_size - 1)
        assert (self.L_noStride - 1) % strides == 0, f"{in_channels=}, {padding=}, {kernel_size=}, {strides=}"
        self.out_size = (self.L_noStride - 1) // strides + 1

        self.weights = nn.Parameter(torch.randn(self.out_size, 1, kernel_size, dtype=dtype) /
                                    torch.sqrt(torch.tensor(kernel_size)))
        self.bias = nn.Parameter(torch.zeros(self.out_size, 1, dtype=dtype)) if use_bias else None

    def __repr__(self):
        return (f"TDLayer({self.in_size}, kernel_size={self.kernel_size}, "
                f"strides={self.strides}, padding={self.padding}) -> {self.out_size}")

    def forward(self, x: Tensor):
        """
        :param x: (B, Cin, T)
        :return:
        """
        Cin = x.shape[1]
        assert Cin == self.in_size, x.shape
        x = x.swapdims(1, 2)  # B, T, Cin

        # print(x.shape)
        x = f.pad(x, (self.padding, self.padding))
        # print(x.shape)  # (..., T, Cin + 2*padding)

        x = x.unfold(size=self.kernel_size, step=self.strides, dimension=-2)
        # print(x.shape)  # (..., T, Cout, K)

        x = self.weights @ x.view(*x.shape[:-2], -1, 1)
        x = x.squeeze(-1)

        if self.bias is not None:
            x = x + self.bias

        return x.squeeze(-1)


class Norm1d(nn.Module):
    """
    This layer is similar to BatchNorm. The only difference is that Norm1d uses the population statistics in both
    training and inference. Population statistics are estimated during training using the Exponential Moving Average (EMA)
    """

    _momentum: ClassVar[float] = None
    _momentum_set: ClassVar[bool] = False

    @classmethod
    def set_momentum(cls, momentum: float):
        assert not cls._momentum_set
        assert momentum is None or 0. < momentum <= 1., momentum
        cls._momentum = momentum
        cls._momentum_set = True

    @staticmethod
    def bn(feature_dim: int, affine: bool = True):
        return Norm1d((1, feature_dim, 1) if affine else None, [0, 2])

    @staticmethod
    def ln(feature_dim: int, affine: bool = True):
        return Norm1d((1, feature_dim, 1) if affine else None, [0, 1])

    def __init__(self, shape: Sequence[int] | None, stat_dims: Sequence[int], eps: float = 1e-6):
        """
        :param shape: shape of weight and bias (has to be broadcastable to the input)
        :param eps: Regularization for the variance, default: 1e-6
        """
        raise Exception("You are an abomination!!!\n"
                        "Don't use Norm1d! It's only one of the very worst ideas I had!")
        assert self._momentum_set, f"Make sure to set the momentum before instantiating any Norm1d layer"
        assert eps > 0
        # assert 0 <= min(stat_dims) and max(stat_dims) <= len(shape)

        nn.Module.__init__(self)
        self.stat_dims = list(stat_dims)

        self.correction = 1.  # Bias correction for the EMA

        self.mean_biased, self.var_biased = 0, 0
        self.mean, self.var = None, None
        self.eps = nn.Parameter(torch.log(torch.tensor(eps)))

        if shape is None:
            self.gamma = self.beta = None
        else:
            self.gamma = nn.Parameter(torch.ones(shape))
            self.beta = nn.Parameter(torch.zeros(shape))

    def __repr__(self):
        if self.gamma is None:
            return f"Norm1d(stat_dims={self.stat_dims}, affine=False)"
        else:
            return f"Norm1d(shape={self.gamma.shape}, stat_dims={self.stat_dims})"

    def forward(self, x: Tensor):
        if self._momentum is None:
            self.var, self.mean = torch.var_mean(x, dim=self.stat_dims, unbiased=False, keepdim=True)
        else:
            if self.training:
                assert not x.requires_grad
                # Batch statistics
                var, mean = torch.var_mean(x, dim=self.stat_dims, unbiased=False, keepdim=True)
                # Add to population statistics using EMA
                self.mean_biased = (1 - self._momentum) * self.mean_biased + self._momentum * mean
                self.var_biased = (1 - self._momentum) * self.var_biased + self._momentum * var
                # Bias correction
                self.correction *= 1 - self._momentum
                self.mean = self.mean_biased / (1 - self.correction)
                self.var = self.var_biased / (1 - self.correction)
            else:
                assert self.mean is not None and self.var is not None, \
                    f"You are trying to evaluate a model, using PopulationNormalization before training the model!"

        x = (x - self.mean) / torch.sqrt(self.var + torch.exp(self.eps))
        return x if self.gamma is None else self.gamma * x + self.beta


class GroupedRNN(nn.Module):
    def __init__(self, rnn_base: type[nn.RNNBase], input_size: int, hidden_size: int = None, groups: int = 1,
                 bidirectional: bool = False,
                 num_layers: int = 1, bias: bool = True, dropout: float = 0.0, device=None, dtype=None):
        nn.Module.__init__(self)
        if hidden_size is None:
            hidden_size = input_size
        assert input_size % groups == 0 and hidden_size % groups == 0, (input_size, hidden_size, groups)

        self.input_size, hidden_size = input_size // groups, hidden_size // groups

        self.lstm = nn.ParameterList([
            rnn_base(input_size=self.input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias,
                     batch_first=True, dropout=dropout, device=device, dtype=dtype, bidirectional=bidirectional)
            for _ in range(groups)])

        self.groups = groups

    def forward(self, x: Tensor):
        assert len(x.shape) == 3, x.shape
        assert x.shape[-1] == self.input_size * self.groups, x.shape
        # B, T, F
        x = torch.split(x, self.input_size, dim=-1)
        return torch.cat([self.lstm[i](x[i])[0] for i in range(self.groups)], dim=-1)


# Layers from DenseNet:

ConvNd = {1: Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
channel_dim = {1: -2, 2: -3, 3: -4}
BatchNormNd = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}
AvgPoolNd = {1: nn.AvgPool1d, 2: nn.AvgPool2d, 3: nn.AvgPool3d}


class DenseBlock(nn.Module):
    # chin, chout, layer
    Layer = Callable[[int, int, int], nn.Module]

    def __init__(self, k0: int, k: int, L: int, layer: Layer, channel_dim: int = -1):
        nn.Module.__init__(self)

        self.k = k
        self.layers = nn.ParameterList(layer(k0 + k * l, k, l) for l in range(L))
        self.dim = channel_dim

        self.k1 = k0 + k * L

    def forward(self, x):
        for layer in self.layers:
            x = torch.cat([x, layer(x)], dim=self.dim)
        return x

    def __repr__(self):
        return f"DenseBlock(L={len(self.layers)}, k={self.k}, layer={repr(self.layers[0])})"

    @staticmethod
    def cnn_layer(bottleneck: bool, kernel_size: int, order: int) -> Layer:
        conv = ConvNd[order]
        norm = BatchNormNd[order]

        def fn(chin: int, chout: int, l: int):
            if bottleneck and chin > 4 * chout:
                return nn.Sequential(
                    norm(chin), nn.ReLU(), conv(in_channels=chin, out_channels=4 * chout, kernel_size=1),
                    norm(4 * chout), nn.ReLU(),
                    conv(in_channels=4 * chout, out_channels=chout, kernel_size=kernel_size)
                )
            else:
                return nn.Sequential(
                    norm(chin), nn.ReLU(), conv(in_channels=chin, out_channels=chout, kernel_size=kernel_size)
                )

        return fn


class TransitionLayer(nn.Module):
    def __init__(self, chin: int, theta: float, order: int = 2):
        nn.Module.__init__(self)
        assert order in (1, 2, 3)
        assert 0 < theta <= 1

        chout = math.floor(chin * theta)

        self.norm = BatchNormNd[order](num_features=chin)
        self.conv = ConvNd[order](in_channels=chin, out_channels=chout, kernel_size=1)
        self.pool = AvgPoolNd[order](kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(self.norm(x)))


class Squeeze(nn.Module):
    def __init__(self, dim: int):
        nn.Module.__init__(self)
        self.dim = dim

    def forward(self, x):
        return torch.squeeze(x, dim=self.dim)

    def __repr__(self):
        return f"Squeeze(dim={self.dim})"


class SwapDims(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        nn.Module.__init__(self)
        self.dim0, self.dim1 = dim0, dim1

    def forward(self, x):
        return torch.swapdims(x, dim0=self.dim0, dim1=self.dim1)

    def __repr__(self):
        return f"SwapDims({self.dim0}, {self.dim1})"


class Exp(nn.Module):
    def forward(self, x):
        return torch.exp(x)


class Median(nn.Module):
    def __init__(self, dim: int = 1):
        nn.Module.__init__(self)
        self._dim = dim

    def forward(self, x):
        return torch.median(x, dim=self._dim)[0]


def weighted_mean(x: Tensor, weights: Tensor) -> Tensor:
    return f.linear(x, f.softmax(weights, dim=-1))


class WeightedMean(nn.Module):
    def __init__(self, num_features: int):
        nn.Module.__init__(self)
        self.weights = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        return weighted_mean(x, self.weights)

    def __repr__(self):
        return f"WeightedMean({self.weights.shape[1]}, dim={self.dim})"


def complex_cardioid(x: Tensor) -> Tensor:
    assert x.dtype == torch.complex64
    return .5 * (1 + torch.cos(x.angle())) * x


class ModReLU(nn.Module):
    def __init__(self, feature_dim: Sequence[int]):
        nn.Module.__init__(self)

        self.b = nn.Parameter(torch.rand(1, *feature_dim))

    def forward(self, x: Tensor):
        return torch.polar(f.relu(x.abs() + self.b), x.angle())


def complex_relu(x: Tensor):
    return torch.complex(f.relu(x.real), f.relu(x.imag))


class Swish(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.beta = nn.Parameter(torch.ones(()))

    def forward(self, x: Tensor) -> Tensor:
        return x * (self.beta * x).sigmoid()


###############
# Funky stuff #
###############

if __name__ == '__main__':
    pass

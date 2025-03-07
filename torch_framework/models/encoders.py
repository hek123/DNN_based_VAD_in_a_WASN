import torch
from torch import Tensor
from torch.nn import Module, ParameterList
from torch.nn import Identity
from torch.nn import functional as f
from typing import Final, Literal, Sequence, Callable

from torch_framework.models.abstract_models import Encoder
from utils.utils import isPow2
from torch_framework.config import GlobalConfig
from torch_framework.models.custom_layers import Conv1d, InstanceNorm1d


class STFT(Encoder):
    def __init__(self, nfft: int = 2 * GlobalConfig.win_size):
        Module.__init__(self)
        assert isPow2(nfft)

        self._nfft: Final = nfft
        self._hop: Final = nfft // 2
        self._win: Final = torch.sqrt(torch.hann_window(nfft))
        self.out_features = self._nfft // 2 + 1

    def __repr__(self):
        return f"STFT(nfft={self._nfft})"

    def forward(self, x: Tensor) -> Tensor:
        # assert len(x.shape) == 2
        return torch.stft(x, n_fft=self._nfft, hop_length=self._hop, win_length=self._nfft, window=self._win,
                          center=True, pad_mode='constant', normalized=True, onesided=True, return_complex=True)[..., :-1]

    def istft(self, x: Tensor) -> Tensor:
        return torch.istft(x, n_fft=self._nfft, hop_length=self._hop, win_length=self._nfft, window=self._win,
                           center=True, length=x.shape[-1]*self._hop, normalized=True, onesided=True, return_complex=False)


class ConvEncoder1(Encoder, Conv1d):
    def __init__(self):
        Conv1d.__init__(self, in_channels=1, out_channels=GlobalConfig.win_size, kernel_size=GlobalConfig.win_size,
                        stride=GlobalConfig.win_size, bias=False)
        self.out_features = GlobalConfig.win_size

    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 2
        return Conv1d.forward(self, x.unsqueeze(1))


class ConvEncoder(Encoder):
    @staticmethod
    def default_encoder(num_layers: int, kernel_multiplier: float = 1., bias: bool = True,
                        act_fn: Callable[[Tensor], Tensor] | None = f.relu) -> "ConvEncoder":
        """
        Returns a convolutional encoder with `num_layers` layers with the strides set such that the number of parameters
        is minimal. The kernel size is set to stride * `overlap`.
        :param act_fn:
        :param num_layers: The number of convolutional layers in the Encoder
        :param kernel_multiplier: The ratio between the kernel size and stride, default: 1
        :param bias:
        :return: ConvEncoder
        """
        assert kernel_multiplier >= 1., kernel_multiplier

        strides = ConvEncoder.get_optimal_kernel_size(num_layers)
        assert sum(strides) == 9, strides
        strides = [2 ** l for l in strides]
        kernel_sizes = [round(kernel_multiplier * s) for s in strides]

        return ConvEncoder(kernel_sizes, strides, strides, act_fn=act_fn, bias=bias)

    def __init__(self, kernel_sizes: Sequence[int], strides: Sequence[int], output_channels: Sequence[int],
                 act_fn: Callable[[Tensor], Tensor], input_channels: int = 1, bias: bool = False):
        assert len(kernel_sizes) == len(strides) == len(output_channels) and len(kernel_sizes) > 0
        assert all(isPow2(s) for s in strides)
        assert all(0 < s <= k for s, k in zip(strides, kernel_sizes))
        Module.__init__(self)

        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.norm = ParameterList([])
        self.conv = ParameterList([])
        self.act_fn = Identity() if act_fn is None else act_fn

        s_tot = 1
        for k, s, chout in zip(kernel_sizes, strides, self.output_channels):
            # set momentum for a window of approx. 3 seconds
            self.norm.append(InstanceNorm1d(input_channels, momentum=s_tot / (3 * GlobalConfig.sample_rate)))
            # print(self.norm[-1]._ema_mean.momentum.item())
            self.conv.append(Conv1d(in_channels=input_channels, out_channels=input_channels*chout,
                                    kernel_size=k, stride=s, groups=input_channels,
                                    padding='causal', bias=bias))
            input_channels *= chout
            s_tot *= s
        self.out_features = input_channels
        assert s_tot == GlobalConfig.win_size, (s_tot, strides, kernel_sizes)

    def __repr__(self):
        return (f"ConvEncoder(kernel_sizes={self.kernel_sizes}, strides={self.strides}, act_fn={self.act_fn}, "
                f"bias={self.conv[0].bias is not None}) -> {self.out_features}")

    def forward(self, x: Tensor):
        if self.input_channels == 1 and len(x.shape) == 2:
            x = x.unsqueeze(1)
        assert len(x.shape) == 3, x.shape
        assert x.shape[1] == self.input_channels

        # x = self.norm(x)
        # shape_in = x.shape
        # numel_in = x.numel()
        for norm, conv in zip(self.norm, self.conv):
            x = self.act_fn(conv(norm(x)))
            # assert x.numel() == numel_in, f"{shape_in, x.shape}, Conv: {conv.kernel_size, conv.stride, conv.front_pad}"

        return x

    @staticmethod
    def _get_all_kernel_sizes(num_layers: int) -> list[tuple[int]]:
        assert num_layers > 0

        args = [()]
        for l in range(num_layers - 1):
            new_args = []
            for i in range(len(args)):
                new_args += [args[i] + (k,) for k in range(1, 10 - sum(args[i]))]
            args = new_args
        args = [arg + (9 - sum(arg),) for arg in args]
        return args

    @staticmethod
    def _get_num_params(kernel_sizes: tuple[int]) -> int:
        assert sum(kernel_sizes) == 9
        n_params = 0
        chin = 1
        for k in kernel_sizes:
            k = 2**k
            n_params += chin * k * (k + 1)
            chin *= k
        return n_params

    @staticmethod
    def get_optimal_kernel_size(num_layers: int):
        return min(ConvEncoder._get_all_kernel_sizes(num_layers), key=ConvEncoder._get_num_params)


class Block(Module):
    def forward(self, x: Tensor) -> Tensor:
        padding = GlobalConfig.win_size - ((x.shape[-1] - 1) % GlobalConfig.win_size + 1)
        x_ = f.pad(x, (0, padding))
        x_ = x_.view(*x.shape[:-1], -1, GlobalConfig.win_size).swapdims(-1, -2)
        # print(x_.shape)
        # assert torch.all(x_[..., 0] == x[..., :self.cfg.win_size]), torch.norm(x_[:, 0] - x[:self.cfg.win_size])
        return x_

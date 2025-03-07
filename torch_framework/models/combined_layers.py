from typing import Callable

import torch
from torch import Tensor
import torch.nn.functional as f
from torch.nn import Module, Sequential, ParameterList
from torch.nn import Identity, LSTM, Linear, RNNBase, ReLU
from torch_framework.models.custom_layers import Conv1d, TDLayer, TransposeTDLayer, Norm1d, PWTDConv1d, GroupedRNN

from collections import OrderedDict

from utils.utils import isPow2


def NormConv1d(
    in_channels: int, out_channels: int,
    kernel_size: int | tuple[int], stride: int | tuple[int] = 1, dilation: int | tuple[int] = 1,
    padding: str | int | tuple[int] = 0, padding_mode: str = 'zeros',
    bias: bool = True, act_fn: Module = Identity(),
    groups: int = 1,
    momentum: float = .01,
    device=None, dtype=None
):

    return Sequential(OrderedDict([
        ('norm', Norm1d(feature_dim=in_channels)),
        ('conv', Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                        dilation=dilation, padding=padding, padding_mode=padding_mode, bias=bias, groups=groups,
                        device=device, dtype=dtype)),
        ('act_fn', act_fn)
    ]))


class TDBlock(Module):
    def __init__(self, in_size: int, channels: tuple[int, int, int], kernel_size: int, strides: int = 1,
                 bias: bool = True, act_fn: Module | Callable[[Tensor], Tensor] = Identity()):
        Module.__init__(self)
        self.l1 = TDLayer(in_size=in_size, in_channels=channels[0], out_channels=channels[1], kernel_size=kernel_size,
                          strides=strides, use_bias=False)
        self.act_fn = act_fn
        self.l2 = TransposeTDLayer(in_size=self.l1.out_size, in_channels=channels[1], out_channels=channels[2],
                                   kernel_size=kernel_size, strides=strides, use_bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.l2(self.act_fn(self.l1(x)))

    def __repr__(self):
        return (f"TDBlock({self.l1.in_size}, channels: {self.l1.Cin}->{self.l1.Cout}->{self.l2.Cout}, "
                f"kernel_size={self.l1.kernel_size}, strides={self.l1.strides}, bias={self.l2.bias is not None}, "
                f"act_fn={self.act_fn.__class__.__name__})")


class SeparableTDConv1d(Module):
    def __init__(self, in_channels: int, channel_kernel: int, channel_strides: int,
                 kernel_size: int | tuple[int], stride: int | tuple[int] = 1, dilation: int | tuple[int] = 1,
                 padding: str | int | tuple[int] = 0, padding_mode: str = 'zeros',
                 bias: bool = True,
                 device=None, dtype=None):
        Module.__init__(self)

        self.dw_conv = Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                              stride=stride, dilation=dilation, padding=padding, padding_mode=padding_mode, bias=bias,
                              groups=in_channels, device=device, dtype=dtype)
        self.pw_td_conv = PWTDConv1d(in_channels=in_channels, kernel_size=channel_kernel, strides=channel_strides)

    def forward(self, x):
        return self.pw_td_conv(self.dw_conv(x))


class BLSTM(Module):
    def __init__(self, in_channels: int, hidden: int, kernel_size: int = 4, strides: int = 2):
        Module.__init__(self)

        self.norm1 = Norm1d.bn(in_channels, affine=False)
        self.reduce = TDLayer(in_size=in_channels, in_channels=1, out_channels=1,
                              kernel_size=kernel_size, strides=strides)

        self.norm3 = Norm1d.bn(self.reduce.out_size)
        self.lstm = LSTM(input_size=self.reduce.out_size, hidden_size=hidden, batch_first=True, num_layers=1)
        self.norm4 = Norm1d.bn(hidden, affine=False)
        self.proj = Linear(in_features=hidden, out_features=self.reduce.out_size)

        self.norm2 = Norm1d.bn(self.reduce.out_size, affine=False)
        self.expand = TransposeTDLayer(in_size=self.reduce.out_size, in_channels=1, out_channels=1,
                                       kernel_size=kernel_size, strides=strides)

    def forward(self, x: Tensor):
        # input shape: (B, C, T)
        x = self.norm1(x).swapdims(1, 2)  # B, T, C_in
        x_ = f.relu(self.reduce(x.unsqueeze(-1)).squeeze(-1))  # B, T, C_reduced

        x_ = self.norm3(x_.swapdims(1, 2)).swapdims(1, 2)
        x__, _ = self.lstm(x_)  # B, T, C_hidden
        x__ = self.norm4(x__.swapdims(1, 2)).swapdims(1, 2)
        x_ = x_ + self.proj(x__)  # B, T, C_reduced

        x_ = self.norm2(x_.swapdims(1, 2)).swapdims(1, 2)
        x = x + f.relu(self.expand(x_.unsqueeze(-1)).squeeze(-1))  # B, T, C_in
        return x.swapdims(1, 2)


class ResLSTMConv(Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int = None, kernel_size: int = 1):
        Module.__init__(self)

        self.conv = Conv1d(in_channels=in_channels, out_channels=hidden, kernel_size=kernel_size, bias=True,
                           padding='causal')
        self.lstm = LSTM(input_size=hidden, hidden_size=hidden, batch_first=True)

        out_channels = in_channels if out_channels is None else out_channels
        self.residual = (in_channels == out_channels)
        self.proj = Conv1d(in_channels=2*hidden, out_channels=out_channels, kernel_size=1, bias=False)

    def forward(self, x: Tensor):
        x1 = (self.conv(x))  # B, H, T
        x2, _ = self.lstm(x1.swapdims(1, 2))  # B, T, H
        x_ = self.proj(torch.cat([f.tanh(x1), x2.swapdims(1, 2)], dim=1))
        return x + x_ if self.residual else x_  # B, Cout, T


class TDRN(Module):
    """
    Time Delay Recurrent Network: Same as Convolutional Recurrent Network, but using time delay layer instead of
    convolution layer
    """

    @staticmethod
    def encoder_decoder(chin: int):
        assert isPow2(chin)

        encoder = Sequential(
            TDLayer(in_size=chin, in_channels=1, out_channels=1, kernel_size=3),
            ReLU(),
            TDLayer(in_size=chin - 2, in_channels=1, out_channels=1, kernel_size=4, strides=2),
            ReLU(),
            TransposeTDLayer(in_size=chin // 2 - 2, in_channels=1, out_channels=1, kernel_size=3)
        )
        decoder = Sequential(
            TDLayer(in_size=chin // 2, in_channels=1, out_channels=1, kernel_size=3),
            ReLU(),
            TransposeTDLayer(in_size=chin // 2 - 2, in_channels=1, out_channels=1, kernel_size=4, strides=2),
            ReLU(),
            TransposeTDLayer(in_size=chin - 2, in_channels=1, out_channels=1, kernel_size=3),
        )
        return encoder, decoder

    def __init__(self, channels: int = 512, hidden: int = None, groups: int = 1, rnn_base: type[RNNBase] = LSTM):
        assert hidden % groups == 0, (hidden, groups)
        assert (channels // 2) % groups == 0, (channels, channels // 2, groups)
        Module.__init__(self)

        self.enc_norm, self.dec_norm = Norm1d(channels), Norm1d(channels // 2)
        self.enc, self.dec = self.encoder_decoder(channels)

        self.rnn_norm = Norm1d(channels // 2)
        self.rnn = GroupedRNN(rnn_base, channels // 2, channels // 2 if hidden is None else hidden, groups,
                              out_proj=True)

    def forward(self, x: Tensor):
        # Encoder (downsampling)
        x1 = self.enc_norm(x).swapdims(1, 2)  # (B, C, T) -> (B, T, C)
        x1 = self.enc(x1.unsqueeze(-1)).squeeze(-1)  # (B, T, C//2)

        # Temporal modeling (RNN)
        x2 = self.rnn_norm(x1.swapdims(1, 2)).swapdims(1, 2)
        x2 = self.rnn(x2)  # B, T, Clstm

        # Decoder (upsampling)
        x1 = self.dec_norm((x1 + x2).swapdims(1, 2)).swapdims(1, 2)
        x1 = self.dec(x1.unsqueeze(-1)).squeeze(-1)  # B, T, C

        return (x.swapdims(1, 2) + x1).unsqueeze(-1)  # B, T, C, 1


class ConvTDBlock(Module):
    def __init__(self, h: int = 1, d: int = 1, C: int = 512, k_conv: int = 5, k_td: int = 5):
        Module.__init__(self)
        self.h, self.d = h, d

        self.conv = Conv1d(in_channels=C, out_channels=h * C, kernel_size=k_conv, groups=C, padding='causal')
        self.norm = Norm1d(C * h, affine=False)
        self.td = TDLayer(in_size=C, in_channels=h, out_channels=d, kernel_size=k_td, padding='same')

    def forward(self, x):
        B, C, T = x.shape
        x = f.relu(self.conv(x))  # B, h*C, T
        x = self.norm(x)
        x = x.swapdims(1, 2).view(B, T, C, self.h)
        x = self.td(x)  # B, T, C, d
        return x


class BottleNeck(Module):
    def __init__(self, in_features: int, hidden: int):
        Module.__init__(self)
        self.l1 = Linear(in_features, hidden, bias=True)
        self.l2 = Linear(hidden, in_features, bias=False)

    def forward(self, x: Tensor):
        return x + self.l2(self.l1(x))

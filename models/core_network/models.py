from typing import Sequence

import torch
from torch import Tensor
import torch.nn.functional as f
from torch.nn import Module, Sequential, ParameterList
from torch.nn import Linear, LSTM, MaxPool2d, MaxPool1d, Upsample, \
    LayerNorm, ReLU, Sigmoid
from models.custom_layers import TDLayer, Conv1d, \
    TimeSeriesNorm, DenseBlock, GatedTSNorm, InstanceNorm1d, \
    Squeeze, AdaptiveNormalization, swappy, squeezy, GroupedRNN
from models.combined_layers import TDBlock, BLSTM

from config import GlobalConfig
from models.main import CoreNetwork


#############################################
# Initial dimension reduction               #
#   input: shape=(B, F=513, T), dtype=cfloat#
#   output: shape=(B, C, T), dtype=float    #
#############################################


class DRCNN(Module):
    def __init__(self, chout: int, kernel_size: int = 3, causal: bool = True, complex_layer: bool = False,
                 complex2real: str = 'abs'):
        Module.__init__(self)
        C = GlobalConfig.win_size + 1

        self.dtype = torch.cfloat if complex_layer else torch.float
        match complex2real:
            case 'abs':
                self.c2r = torch.abs
            case 'pow':
                self.c2r = self.pow
            case 'as_real':
                self.c2r = self.as_real
            case 'real_relu':
                self.c2r = self.real_relu
            case _:
                raise AssertionError(complex2real)

        D1 = 2 if ((not complex_layer) and complex2real.count('real')) else 1
        D2 = 2 if complex2real.count('real') else 1
        self.l1 = Conv1d(in_channels=D1 * C, out_channels=D1 * C, kernel_size=kernel_size,
                         padding='causal' if causal else 'same', groups=C, bias=False, dtype=self.dtype)
        self.norm = AdaptiveNormalization(D2 * C, momentum=.01)
        self.l2 = Conv1d(in_channels=D2 * C, out_channels=chout, kernel_size=1)

    @staticmethod
    def as_real(x: Tensor):
        B, F, T = x.shape
        return torch.view_as_real(x.swapdims(1, 2)).view([B, T, 2 * F]).swapdims(1, 2)

    @staticmethod
    def real_relu(x: Tensor):
        B, F, T = x.shape
        return f.relu(torch.view_as_real(x.swapdims(1, 2)).view([B, T, 2 * F]).swapdims(1, 2))

    @staticmethod
    def pow(x: Tensor):
        return torch.square(x.abs())

    def forward(self, x):
        x = self.c2r(self.l1(x)) if self.dtype.is_complex else self.l1(self.c2r(x))
        x = self.norm(x)
        return self.l2(x)


class DRTDNN(Module):
    def __init__(self, kernel_size: Sequence[int], L: int = 1, channels: Sequence[int] = (),
                 strides: Sequence[int] = None, pooling: Sequence[int] = None, act_fn: type[Module] = ReLU):
        Module.__init__(self)
        assert L >= 1
        assert len(kernel_size) == L
        assert len(channels) == L - 1
        assert strides is None or len(strides) == L
        assert pooling is None or len(pooling) == L
        if strides is None:
            strides = [1] * L
        if pooling is None:
            pooling = [1] * L

        self._act_fn = act_fn.__name__

        self.layers = ParameterList()
        size = GlobalConfig.win_size + 1
        chin = 2
        channels = list(channels) + [1]
        self.sizes = [(size, chin)]
        for chout, k, s, p in zip(channels, kernel_size, strides, pooling):
            layer = TDLayer(in_size=size, in_channels=chin, out_channels=chout, kernel_size=k, strides=s)
            self.layers.append(Sequential(
                LayerNorm([size, chin]),
                # SwapDims(1, 3), TimeSeriesNorm(feature_dim=[size, chin]), SwapDims(1, 3),
                layer,
                act_fn(),
                MaxPool2d(kernel_size=(p, 1), stride=(p, 1))
            ))
            chin = chout
            assert layer.out_size % p == 0
            size = layer.out_size // p
            self.sizes.append((size, chin))

        self._out_size = size

    def __repr__(self):
        out = f"DRTDNN({self._act_fn}): {self.sizes[0]}"
        for size in self.sizes[1:]:
            out += f" -> {size}"
        return out

    @property
    def out_size(self):
        return self._out_size

    def forward(self, x):
        x = torch.view_as_real(x).swapdims(1, 2)
        # print(x.shape)  # B, T, F, 2
        for layer in self.layers:
            x = layer(x)

        return x.swapdims(1, 2).squeeze(-1)


def cnn1(kernel_size: int = 3, causal: bool = True, c2r: str = 'abs', complex: bool = False):
    return Sequential(
        DRCNN(chout=1, kernel_size=kernel_size, causal=causal, complex_layer=complex, complex2real=c2r),
        Squeeze(1),
        Sigmoid()
    )


class CNNBaseline(CoreNetwork):
    def __init__(self, chin: int):
        Module.__init__(self)
        self.out_features = chin

        self.norm1 = InstanceNorm1d(chin)
        self.conv1a = Conv1d(chin, chin, 5, padding='causal', groups=chin)
        self.conv1b = Conv1d(chin, chin, 1)

        self.norm2 = InstanceNorm1d(chin)
        self.conv2a = Conv1d(chin, chin, 5, padding='causal', groups=chin)
        self.conv2b = Conv1d(chin, chin,1)

    def forward(self, x):
        if torch.is_complex(x):
            x = x.abs()
        assert len(x.shape) == 3, x.shape

        # Layer 1
        x = self.norm1(x)
        x = f.relu(self.conv1a(x))
        x = f.relu(self.conv1b(x))

        # Layer 2
        x = self.norm2(x)
        x = f.relu(self.conv2a(x))
        x = f.relu(self.conv2b(x))

        return x


class LSTMNet(CoreNetwork):
    def __init__(self, chin: int, groups: int = 16):
        Module.__init__(self)
        self.out_features = chin

        self.pw_mix = TDLayer(chin, 5, padding='same')

        self.norm1 = InstanceNorm1d(chin)
        self.lstm1 = GroupedRNN(LSTM, chin, groups=groups)

        self.norm2 = InstanceNorm1d(chin)
        self.l2 = TDLayer(chin, 5, padding='same')

        self.norm3 = InstanceNorm1d(chin)
        self.l3 = TDLayer(chin, 5, padding='same')

        self.norm4 = InstanceNorm1d(chin)
        self.lstm4 = GroupedRNN(LSTM, chin, groups=groups)

        self.norm5 = InstanceNorm1d(chin)
        self.l5 = TDLayer(chin, 5, padding='same')

    def forward(self, x_in):
        # Layer 1: pw_mix
        x1 = f.relu(squeezy(self.pw_mix, -1, x_in.swapdims(1, 2)))

        x = swappy(self.norm1, 1, 2, x1)
        x = x + self.lstm1(x)

        x = swappy(self.norm2, 1, 2, x)
        x = x + f.relu(squeezy(self.l2, -1, x))

        x = swappy(self.norm3, 1, 2, x)
        x = x + f.relu(squeezy(self.l3, -1, x))

        x = swappy(self.norm4, 1, 2, x + x1)
        x = x + self.lstm4(x)

        x = swappy(self.norm5, 1, 2, x)
        x = x + f.relu(squeezy(self.l5, -1, x))

        return x.swapdims(1, 2)


class CNN2(Module):
    def __init__(self):
        Module.__init__(self)
        self.l0 = STFT()
        chin = self.l0.num_channels

        self.dim_reduction = Sequential(
            TDLayer(in_size=chin, in_channels=2, out_channels=8, kernel_size=3),
            ReLU(),
            LayerNorm(normalized_shape=[chin-2, 8]),

            TDLayer(in_size=chin-2, in_channels=8, out_channels=1, kernel_size=4),
            ReLU(),
            LayerNorm(normalized_shape=[(chin - 5), 1]),

            MaxPool2d((2, 1), (2, 1))
        )
        chin = (chin - 5) // 2

        self.cnn_stack = Sequential(
            Conv1d(in_channels=chin, out_channels=chin, kernel_size=5, groups=chin, padding='causal'),
            ReLU(),
            Conv1d(in_channels=chin, out_channels=64, kernel_size=1),
            ReLU(),
            InstanceNorm1d(num_features=64),

            Conv1d(in_channels=64, out_channels=16, kernel_size=5, padding='causal'),
            ReLU(),
            InstanceNorm1d(num_features=16)
        )

        self.l_end = Conv1d(in_channels=16, out_channels=1, kernel_size=1)

    def forward(self, x):
        x = torch.view_as_real(self.l0(x))
        # print(x.shape)

        # Block 1
        x = torch.swapdims(x, 1, 2)
        x = self.dim_reduction(x)
        # print(x.shape)

        # Block 2
        x = torch.swapdims(torch.squeeze(x, dim=-1), 1, 2)
        x = self.cnn_stack(x)
        # print(x.shape)

        x = f.sigmoid(self.l_end(x))
        # print(x.shape)

        return torch.squeeze(x, dim=1)


class DenseLSTM(Module):
    def __init__(self):
        Module.__init__(self)

        self.dense_rnn = DenseBlock(L=5, k=16, k0=self.l2.out_size, layer=self.RNNLayer)

    def forward(self, x):
        x = self.l0(x)
        # x = self.conv(x)
        x = torch.view_as_real(x.swapdims(1, 2))
        # print(x.shape)

        x = self.norm1(x)
        x = f.relu(self.l1(x))
        x = self.norm2(x)
        x = f.relu(self.l2(x))
        x = x.squeeze(-1)
        # print(x.shape)  # B, T, F

        x = self.dense_rnn(x)
        # print(x.shape)

        return x

    class RNNLayer(Module):
        def __init__(self, chin: int, chout: int, l: int):
            Module.__init__(self)

            self.norm = TimeSeriesNorm(chin)
            self.rnn = LSTM(input_size=chin, hidden_size=chout, batch_first=True)

        def forward(self, x):
            x = self.norm(x)
            x, _ = self.rnn(x)
            return x


class DenseRUNet(Module):
    def __init__(self, momentum: float):
        Module.__init__(self)

        self.encoder = ConvEncoder.build_encoder(3, 2)

        self.l1a = BLSTM(in_channels=512, hidden=16)
        self.l2a = BLSTM(in_channels=512,  hidden=16)
        self.l3ab = Sequential(
            BLSTM(in_channels=512, hidden=16),
            BLSTM(in_channels=512, hidden=16)
        )
        self.l2b = BLSTM(in_channels=512, hidden=16)
        self.l1b = BLSTM(in_channels=512, hidden=16)

        self.fc1 = Linear(512, 16)
        self.fc2 = Linear(16, 1)

    def forward(self, x, **kwargs):
        x = self.encoder(x)
        # print(x.shape)  # B, C, T

        x1 = self.l1a(x)
        x2 = self.l2a(f.max_pool1d(x1, 2, ceil_mode=True))
        x3 = self.l3ab(f.max_pool1d(x2, 2, ceil_mode=True))
        print(x3.shape)

        x3 = torch.repeat_interleave(x3, 2, dim=-1)[..., :x2.shape[-1]]
        x2 = self.l2b(x2 + x3)

        x2 = torch.repeat_interleave(x2, 2, dim=-1)[..., :x1.shape[-1]]
        x = self.l1b(x1 + x2)

        x = f.tanh(self.fc1(x.swapdims(1, 2)))
        return f.sigmoid(self.fc2(x)).squeeze(-1)

    class Down(Module):
        def __init__(self, chin, chout):
            Module.__init__(self)

            self.l1 = TimeSeriesNorm([chin], time_last=False)
            self.l2 = Linear(in_features=chin, out_features=chout)
            self.l3 = MaxPool1d(kernel_size=2, stride=2)

        def forward(self, x):
            x = self.l1(x)
            x = self.l2(x)
            x = self.l3(x.swapdims(1, 2)).swapdims(1, 2)
            return x

    class Up(Module):
        def __init__(self, chin, chout):
            Module.__init__(self)

            self.l1 = TimeSeriesNorm([chin], time_last=False)
            self.l2 = Linear(in_features=chin, out_features=chout)
            self.l3 = Upsample(scale_factor=2)

        def forward(self, x):
            x = self.l1(x)
            x = self.l2(x)
            x = self.l3(x.swapdims(1, 2)).swapdims(1, 2)
            return x


class CNN3(Module):
    class Layer(Module):
        C = GlobalConfig.win_size + 1

        def __init__(self, chin: int, chout: int, conv_kernel: int, td_kernel: int):
            Module.__init__(self)

            self.norm = LayerNorm([self.C, chin])

            self.dw_conv = Conv1d(in_channels=self.C * chin, out_channels=self.C * chin,
                                  kernel_size=conv_kernel, groups=self.C * chin, bias=False, padding='causal')
            self.td_pw_conv = TDBlock(in_size=self.C, channels=(chin, chout, chout), kernel_size=td_kernel)

        def forward(self, x: Tensor):
            B, T, F, C = x.shape
            x = self.norm(x)
            x = self.dw_conv(x.view(B, T, F * C).swapdims(1, 2))
            x = self.td_pw_conv(x.swapdims(1, 2).view(B, T, F, C))
            x = f.relu(x)
            return x

    def __init__(self):
        Module.__init__(self)

        self.l0 = STFT()

        self.l0_ = TDBlock(in_size=self.l0.num_channels, channels=(2, 4, 8), kernel_size=3)

        self.l1 = self.Layer(8, 8, 3, 3)
        self.l2 = self.Layer(16, 8, 3, 3)
        self.l3 = self.Layer(24, 8, 3, 3)

        self.fc1 = TDLayer(in_size=self.l0.num_channels, in_channels=32, out_channels=1, kernel_size=3)
        self.fc2 = Linear(in_features=self.l0.num_channels - 2, out_features=1)

    def forward(self, x):
        x = torch.view_as_real(self.l0(x).swapdims(1, 2))  # B, T, F, 2
        x = f.relu(self.l0_(x))  # B, T, F, 8

        x = torch.cat([x, self.l1(x)], dim=3)
        x = torch.cat([x, self.l2(x)], dim=3)
        x = torch.cat([x, self.l3(x)], dim=3)

        x = f.relu(self.fc1(x))
        x = f.sigmoid(self.fc2(x.squeeze(3)))
        return x.squeeze(2)


class DeepGatedNN(Module):
    def __init__(self):
        Module.__init__(self)

        self.gate = Conv1d(in_channels=513, out_channels=1, kernel_size=1)
        self.norm = GatedTSNorm(feature_dim=513)
        # self.norm = BatchNorm1d(513)
        self.fc = Linear(in_features=513, out_features=1)

    def forward(self, x):
        if torch.is_complex(x):
            x = x.abs()
        g = f.sigmoid(self.gate(x))
        x = self.norm(x, g)
        x = self.fc(x.swapdims(1, 2))
        return f.sigmoid(x).squeeze(2)

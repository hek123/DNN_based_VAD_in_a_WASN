import torch
import torch.nn.functional as f

from torch import Tensor
from torch.nn import Module, Sequential, ParameterList
from torch.nn import LSTM, Linear, Conv2d, LayerNorm, BatchNorm1d, Softmax

from models.abstract_models import CoreNetwork
from models.custom_layers import Conv1d, TimeSeriesNorm, InstanceNorm1d
from models.combined_layers import TDBlock


class CNN(CoreNetwork):
    def __init__(self, chin: int, kernel_size: int, chout: int = None, layers: int = 1):
        Module.__init__(self)
        if chout is None:
            chout = chin

        self.norms = ParameterList([TimeSeriesNorm(chin if l == 0 else chout) for l in range(layers)])
        self.layers = ParameterList([
            Conv1d(in_channels=chin if l == 0 else chout, out_channels=chout, kernel_size=kernel_size, padding='causal')
            for l in range(layers)
        ])
        self.out_features = chout

        # self.rnn = ExponentialMovingAverage([1., .5, .2, .1, .05, .02, .01, .001])
        # self.norm2 = Norm1d.bn(8, affine=False)
        # self.fc = Conv1d(in_channels=8, out_channels=1, kernel_size=1)

    def forward(self, x, **meta):
        assert len(x.shape) == 3, x.shape  # B, C, T

        for norm, layer in zip(self.norms, self.layers):
            x = f.relu(layer(norm(x)))  # B, C, T

        # x = self.rnn(x)  # B, c(8), T
        # x = f.sigmoid(self.fc(self.norm2(x)))  # .view(x.shape[0], 16*8, x.shape[-1]))))
        return x


#################################
# Networks from existing papers #
#################################


# NEURAL NETWORK BASED SPECTRAL MASK ESTIMATION FOR ACOUSTIC BEAMFORMING


class M1FF(Module):
    """
    From: NEURAL NETWORK BASED SPECTRAL MASK ESTIMATION FOR ACOUSTIC BEAMFORMING
    """
    def __init__(self, in_channels: int = 513):
        Module.__init__(self)

        self.l1 = Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.norm1 = BatchNorm1d(in_channels)
        self.l4 = Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.out_features = in_channels

    def forward(self, x):
        # print(x.shape)
        if torch.is_complex(x):
            x = x.abs()
        # print(x.shape)

        x = f.relu(f.dropout(self.l1(x), p=.5))
        x = self.norm1(x)
        x = f.sigmoid(self.l4(x))
        return x


class M2LSTM(CoreNetwork):
    """
    From: NEURAL NETWORK BASED SPECTRAL MASK ESTIMATION FOR ACOUSTIC BEAMFORMING
    """
    def __init__(self, in_features: int = 513, bidirectional: bool = True):
        Module.__init__(self)

        if in_features != 513:
            print(f"not the original input size !!!")

        norm = InstanceNorm1d
        # self.norm0 = norm(in_features)
        # self.l1 = GroupedRNN(LSTM, 512, 256 if bidirectional else 512, groups=16, bidirectional=bidirectional)
        self.l1 = LSTM(input_size=in_features, hidden_size=256 if bidirectional else 512,
                       bidirectional=bidirectional, batch_first=True)
        self.norm1 = norm(512)

        self.l2 = Conv1d(in_channels=512, out_channels=512, kernel_size=1)
        self.norm2 = norm(512)

        self.l3 = Conv1d(in_channels=512, out_channels=512, kernel_size=1)
        self.norm3 = norm(512)

        self.l4 = Conv1d(in_channels=512, out_channels=512, kernel_size=1)
        self.out_features = 512

    def forward(self, x: Tensor, *args, **kwargs):
        assert len(x.shape) == 3, x.shape  # M, F, T
        if torch.is_complex(x):
            x = x.abs()[:, :512, :]

        # x = self.norm0(x)
        x = self.l1(f.dropout(x.swapdims(1, 2), p=.5))[0].swapdims(1, 2)
        x = self.norm1(x)
        x = f.relu(f.dropout(self.l2(x), p=.5))
        x = self.norm2(x)
        x = f.relu(f.dropout(self.l3(x), p=.5))
        x = self.norm3(x)
        # print(torch.mean(x[:, :, 100], dim=1))
        x = f.sigmoid(self.l4(x))

        return x


# COMBINING DEEP NEURAL NETWORKS AND BEAMFORMING FOR REAL-TIME MULTI-CHANNEL SPEECH ENHANCEMENT USING A WIRELESS
# ACOUSTIC SENSOR NETWORK

class M1CNN(Module):
    class _Layer(Module):
        def __init__(self, l: int, channels: int):
            Module.__init__(self)
            self.conv = Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), dilation=2**l,
                               padding='same')
            self.norm = LayerNorm(normalized_shape=[513], elementwise_affine=False)

        def forward(self, x):
            return x + self.norm(f.relu(self.conv(x)))

    def __init__(self, S: int, L: int, causal: bool = False, channels: int = 16):
        Module.__init__(self)
        assert not causal

        self.l0 = STFT()
        self.conv1 = Conv2d(in_channels=2, out_channels=channels, kernel_size=(1, 1))
        self.norm1 = LayerNorm(normalized_shape=[513])

        def stack(s: int):
            return Sequential(*[M1CNN._Layer(l=l, channels=channels) for l in range(L)])

        self.layers = Sequential(*[stack(s) for s in range(S)])

        self.fc = Conv2d(in_channels=channels, out_channels=1, kernel_size=(1, 1))

        self.spp2vad = Conv1d(in_channels=513, out_channels=1, kernel_size=1, bias=False,
                              weight_activation=Softmax(dim=1))

    def forward(self, x):
        # print(x.shape)
        x = torch.view_as_real(self.l0(x))
        # print(x.shape)  # (B, F, N, 2)

        x = f.relu(self.conv1(x.swapdims(1, 3)))
        x = self.norm1(x)
        # print(x.shape)  # (B, 16, N, F)

        x = self.layers(x)

        x = f.sigmoid(self.fc(x).squeeze(1))
        x = self.spp2vad(x).squeeze(1).clip(0, 1)
        return x


# DUAL-PATH RNN: EFFICIENT LONG SEQUENCE MODELING FOR TIME-DOMAIN SINGLE-CHANNEL SPEECH SEPARATION

class DualPathRNN(Module):
    def __init__(self, K: int, P: int):
        assert 0 < P <= K
        Module.__init__(self)

        self.K = K
        self.P = P
        self.H = 16
        self.N = 4

        # self.intra_rnn = LSTM(input_size=1, hidden_size=H, batch_first=True, bidirectional=True)
        # self.intra_fc = Linear(in_features=K, out_features=K)
        self.intra_fc = TDBlock(in_size=K, channels=(1, 2, self.N), kernel_size=3)
        self.intra_norm = LayerNorm([K, self.N])

        self.inter_rnn = LSTM(input_size=self.N, hidden_size=self.H, batch_first=True)
        self.inter_fc = Linear(in_features=self.H, out_features=1)
        self.inter_norm = LayerNorm([K])

    def forward(self, x: Tensor):
        # 1) Create a sliding window view
        x = x.unfold(dimension=1, size=self.K, step=self.P)
        B, S, K = x.shape  # B, S, K

        # 2) Intra
        # x_ = self.intra_rnn(x.view(B * S, K, 1)).view(B, S, K, self.H)
        x = self.intra_norm(self.intra_fc(x.unsqueeze(-1)))  # B, S, K, N

        # 3) Inter
        x = x.swapdims(1, 2)
        x_, _ = self.inter_rnn(x.reshape(B * K, S, self.N))
        x = self.inter_norm(self.inter_fc(x_).view(B, K, S).swapdims(1, 2))

        return x.swapdims(1, 2)


if __name__ == '__main__':
    pass

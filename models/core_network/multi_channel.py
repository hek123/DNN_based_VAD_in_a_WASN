import random
from typing import Sequence, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module, Sequential
from torch.nn import Linear, LSTM, MaxPool1d, ReLU, MultiheadAttention
import torch.nn.functional as f

from models.abstract_models import CoreNetwork
import models.core_network.baseline as b
from models.custom_layers import TDLayer, TransposeTDLayer, Conv1d, Norm1d, \
    ExponentialMovingAverage, EMACell, GroupedRNN, \
    InstanceNorm1d, \
    swappy, squeezy
from models.combined_layers import TDBlock, ResLSTMConv


class AttTypeI(MultiheadAttention):
    def __init__(self, feature_dim: int, num_heads: int):
        MultiheadAttention.__init__(self, embed_dim=feature_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        # x: (M, F, T)
        x = x.swapdims(0, 1).swapdims(0, 2)
        x, _ = MultiheadAttention.forward(self, x, x, x, need_weights=False)
        # print(x)
        return x.swapdims(0, 2).swapdims(0, 1)


class ChannelAttention(Module):
    def __init__(self, momentum: float | None, trainable: bool = False, causal: bool = True, lag: int = 1):
        assert lag > 0, lag
        Module.__init__(self)
        self._causal = causal
        if causal:
            self.ema = ExponentialMovingAverage(momentum, trainable=trainable)
        self.lag = lag

    def __repr__(self):
        if self._causal:
            return f"{self.__class__.__name__}(lag={self.lag}, ema={self.ema})"
        else:
            return f"{self.__class__.__name__}(lag={self.lag}, non-causal)"

    def delay(self, x: Tensor):
        x_ext = [x]
        for l in range(1, self.lag):
            x_ = torch.roll(x, l, dims=1)
            x_[:, :l, :, :] = 0.
            x_ext.append(x_)
        return torch.concat(x_ext, dim=2)

    def get_single_channel_mask(self, att_shape: torch.Size):
        M = att_shape[-2]
        assert att_shape[-1] == self.lag * M, (att_shape, self.lag)

        mask = torch.empty(att_shape)
        for l in range(self.lag):
            mask[:, :, :, l*M:(l+1)*M] += torch.eye(M)

        return mask

    def forward(self, k: Tensor, q: Tensor, v: Tensor):
        """
        :param k: (F, T, Nk, dk)
        :param q: (F, T, Nq, dk)
        :param v: (F, T, Nk, dv)
        :return: shape: (F, T, Nq, dv)
        """

        assert len(k.shape) == len(q.shape) == len(v.shape) == 4 and \
            k.shape[0] == q.shape[0] == v.shape[0] and \
            (k.shape[1] == q.shape[1] == v.shape[1]) or (k.shape[1] == q.shape[1] and not self._causal) and \
            k.shape[2] == v.shape[2] and q.shape[2] and \
            k.shape[3] == q.shape[3], (k.shape, q.shape, v.shape)

        k, v = self.delay(k), self.delay(v)

        if self._causal:
            # (..., Nq, dk) @ (..., dk, Nk) -> (.., Nq, Nk)
            att = q @ k.swapdims(-1, -2)  # (B, T, Nq, Nk)
            # print(att.shape)
            att = swappy(self.ema, 1, 3, att)
            # raise AssertionError("Dumb")
        else:
            assert k.shape[3] == 1, k.shape
            k, q = k.squeeze(3), q.squeeze(3)
            # (..., T, Nq).mT @ (..., T, Nk) -> (..., Nq, Nk)
            att = (q.mT @ k).unsqueeze(1)  # (F, 1, Nq, Nk)
            # print(att.shape)

        # # Make single-channel by masking all cross-channel attention weights
        # att = att * self.get_single_channel_mask(att.shape)

        # Normalize: Layer Norm
        att = att / (torch.norm(att, dim=[0, 3], keepdim=True) + 1e-8)

        # (..., Nq, Nk) @ (..., Nk, dv) -> (..., Nq, dv)
        out = att @ v
        return out


class Type0(ChannelAttention):
    def forward(self, *kqv):
        # (M, T, F) -> (1, T, M, F)
        y = ChannelAttention.forward(self, *[x.swapdims(0, 1).unsqueeze(0) for x in kqv])
        return y.squeeze(0).swapdims(0, 1)


class TypeI(ChannelAttention):
    def forward(self, *kqv):
        # (M, T, F) -> (F, T, M, 1)
        y = ChannelAttention.forward(self, *[x.swapdims(0, 2).unsqueeze(-1) for x in kqv])
        return y.squeeze(-1).swapdims(0, 2)


class TypeII(ChannelAttention):
    def forward(self, k, q, v):
        M, T, F = k.shape
        # (M, T, C) -> (1, T, M*F, 1)
        y = ChannelAttention.forward(self, *[x.swapdims(0, 1).view(1, T, M*F, 1) for x in (k, q, v)])
        return y.view(T, M, F).swapdims(0, 1)


class RecurrentCA(Module):
    def __init__(self, momentum: float = .05):
        Module.__init__(self)

        self.ema = EMACell(momentum=momentum, unbiased=True)

    def forward(self, K, Q, V):
        # k, q, v: (C, T, M, d)

        C, T, Mq, dk = Q.shape
        M = K.shape[2]

        out = torch.empty_like(Q)
        att = torch.ones([C, Mq, M])
        self.ema.reset_state()
        for t in range(T):
            # 1) Filter
            k, q, v = K[:, t, :, :], Q[:, t, :, :], V[:, t, :, :]  # (C, M, d)
            out[:, t, :, :] = att @ v  # (C, M', dv)

            # 2) Update
            att = (q - out[:, t, :, :]) @ k.swapdims(1, 2)  # (C, M', M)
            # print(att.shape)
            att = self.ema(att)

            # Normalize: Layer Norm
            att = att / (torch.norm(att, dim=[0, 2], keepdim=True) + 1e-12)

        return out


class ChannelAttentionV1(CoreNetwork):
    class W(Module):
        def __init__(self, features: int):
            Module.__init__(self)
            self.conv1 = Conv1d(features, 3 * features, 5, groups=features, padding='causal')
            self.conv2 = Conv1d(3 * features, 3 * features, 1, groups=3)

        def forward(self, x):
            M, T, F = x.shape
            x = f.relu(self.conv1(x.swapdims(1, 2))).view(M, F, 3, T)
            x = x.swapdims(1, 2).reshape(M, F*3, T)
            return self.conv2(x).swapdims(1, 2)

    class ConvNet(Module):
        class SepConv(Module):
            def __init__(self, features: int, groups: int, kernel_size: int = 5):
                Module.__init__(self)
                self.norm = InstanceNorm1d(features * groups)
                self.conv1 = Conv1d(features * groups, features * groups, kernel_size, groups=features * groups, padding='causal')
                self.conv2 = Conv1d(features * groups, features * groups, 1, groups=groups)

            def forward(self, x):
                x_ = self.norm(x)
                x_ = f.relu(self.conv1(x_))
                x_ = self.conv2(x_)
                return x + x_

        def __init__(self, features: int):
            Module.__init__(self)
            self.norm1 = InstanceNorm1d(features)
            self.conv1a = Conv1d(features, 3 * features, 5, groups=features, padding='causal')
            self.conv1b = Conv1d(3 * features, 3 * features, 1, groups=3)

        def forward(self, x):
            M, T, F = x.shape
            x = self.norm1(x.swapdims(1, 2))
            x = f.relu(self.conv1a(x))
            # Restructure groups
            x = x.view(M, F, 3, T).swapdims(1, 2).reshape(M, F*3, T)
            x = self.conv1b(x)

            return torch.split(x.swapdims(1, 2), F, 2)

    class TransEnc(Module):
        def __init__(self, chin):
            Module.__init__(self)
            self.att = AttTypeI(chin, 8)
            self.norm1 = InstanceNorm1d(chin)
            self.ff = Sequential(
                Conv1d(chin, 2 * chin, 5, groups=chin, padding='causal'),
                ReLU(),
                Conv1d(2 * chin, chin, 1)
            )
            self.norm2 = InstanceNorm1d(chin)

        def forward(self, x):
            x = self.norm1(x + self.att(x))
            return self.norm2(x + self.ff(x))

    def __init__(self, chin: int):
        Module.__init__(self)
        self.out_features = chin

        # print((ca == Type0), ca)

        # self.att = ParameterList()
        # for _ in range(1):
        #     cat = TDLayer(chin, 1, in_channels=2, out_channels=1, use_bias=False, padding='same')
        #     init.constant_(cat.weights, .5)
        #     self.att.append(ParameterDict({
        #         'Wkqv': self.ConvNet(chin),
        #         'att': ca(causal=False, momentum=None, lag=3),
        #         'cat': cat
        #     }))
        #     # self.Wkqv = Linear(chin, 3 * chin)

        self.att_block = Sequential(
            self.TransEnc(chin), self.TransEnc(chin), self.TransEnc(chin)
        )

        # self.att2 = AttTypeI(chin, 8)
        # self.norm2a = InstanceNorm1d(chin)
        # self.ff2 = Sequential(
        #     Conv1d(chin, 2 * chin, 1),
        #     ReLU(),
        #     Conv1d(2 * chin, chin, 1)
        # )
        # self.norm2b = InstanceNorm1d(chin)

        # self.norm = InstanceNorm1d(chin)
        self.lstm = GroupedRNN(LSTM, chin, groups=8)
        # self.lstm = ResLSTM(chin)

    def forward(self, x: Tensor, *args, **kwargs):
        # print(x.shape)  # (M, F, T)
        # x = x.swapdims(1, 2)

        x = self.att_block(x)

        # assert len(self.att) == 2
        # # for params in self.att:
        # params = self.att[0]
        # kqv = params['Wkqv'](x)
        # x = params['att'](*kqv)
        # # x = torch.stack([x, params['att'](*kqv)], dim=-1)
        # # x = params['cat'](x).squeeze(-1)
        #
        # kqv = [skip + x for skip, x in zip(kqv, self.att[1]['Wkqv'](x))]
        # x = self.att[1]['att'](*kqv)
        # # x = torch.stack([x, self.att[1]['att'](*kqv)], dim=-1)
        # # x = self.att[1]['cat'](x).squeeze(-1)

        # x = self.norm(x)
        x = self.lstm(x.swapdims(1, 2))

        return x.swapdims(1, 2)


class ConvTDBlock(Module):
    def __init__(self, C: int = 512, k_conv: int = 5, k_td: int = 5):
        Module.__init__(self)

        self.conv = Conv1d(in_channels=C, out_channels=C, kernel_size=k_conv, groups=C, padding='causal')
        self.norm = Norm1d.bn(C)
        self.td = TDLayer(in_size=C, in_channels=1, out_channels=1, kernel_size=k_td, padding='same')

    def forward(self, x):
        # B, C, T = x.shape
        x = f.relu(self.conv(x))  # B, C, T
        x = self.norm(x)
        # (B, C, T) -> (B, T, C, 1)
        x = x.swapdims(1, 2).unsqueeze(-1)
        x = self.td(x)  # B, T, C, 1
        return x.swapdims(0, 2)  # C, T, N, 1


class ResLSTM(Module):
    norm = InstanceNorm1d

    def __init__(self, features: int, chout: int = 1, chin: int = 1):
        Module.__init__(self)
        self.m = chout
        self.chin = chin

        k = 8 if (features % 2 == 0) else 9
        self.reduce = TDLayer(in_size=features, in_channels=chin, out_channels=chin, kernel_size=k, strides=4)
        C = self.reduce.out_size
        self.norm1 = self.norm(C * chin)

        self.lstm = LSTM(input_size=C * chin, hidden_size=C * chout, batch_first=True)

        self.norm2 = self.norm(C * (chout + chin))
        self.expand = TransposeTDLayer(in_size=C, in_channels=chin + chout, out_channels=chout, kernel_size=k, strides=4)

        self.norm3 = self.norm(features * (chout + chin))
        self.cat = TDLayer(in_size=features, in_channels=chin + chout, out_channels=chout, kernel_size=5, padding='same')

    def forward(self, x: Tensor):
        # print(x.shape)  # B, T, F
        if self.chin == 1 and len(x.shape) == 3:
            x = x.unsqueeze(-1)
        B, T, _, _ = x.shape

        x1 = f.relu(self.reduce(x)).squeeze(-1)
        x1 = swappy(self.norm1, 1, 2, x1.view(B, T, self.reduce.out_size * self.chin))

        # print(x_.shape)
        x2, _ = self.lstm(x1.view(B, T, self.reduce.out_size * self.chin))  # B, T, M * F
        # print(x_.shape)
        x1 = torch.cat([x1, x2], dim=2)  # B, T, (1+M)*F
        x1 = swappy(self.norm2, 1, 2, x1)

        x1 = f.relu(self.expand(x1.view(B, T, self.reduce.out_size, self.m + self.chin)))

        x = torch.cat([x, x1], dim=-1)
        x = self.norm3(x.view(B, T, self.expand.out_size * (self.m + self.chin)).swapdims(1, 2)).swapdims(1, 2)
        x = self.cat(x.view(B, T, self.expand.out_size, self.m + self.chin)).squeeze(-1)

        # x1 = self.norm2(x1.view(B, T, self.expand.out_size * self.m).swapdims(1, 2)).swapdims(1, 2).view(B, T, self.expand.out_size, self.m)
        # x = self.cat(torch.cat([x.unsqueeze(-1), x1], dim=-1)).squeeze(-1)
        return x  # B, T, F, M


class RL(CoreNetwork, ResLSTM):
    def __init__(self, chin: int):
        CoreNetwork.__init__(self)
        ResLSTM.__init__(self, chin)
        self.out_features = chin

    def forward(self, x):
        return ResLSTM.forward(self, x.swapdims(1, 2)).swapdims(1, 2)


class CALinear(Module):
    def __init__(self, features: int = 512):
        Module.__init__(self)
        self.layer = Linear(features, features)

    def forward(self, x):
        x = self.layer(x)
        # (M, T, F) -> (F, T, M, 1)
        return x.swapdims(0, 2).unsqueeze(-1)


class CAConvV1(Module):
    def __init__(self, features: int = 512):
        Module.__init__(self)
        self.layer = Conv1d(features, features, groups=features, kernel_size=5, padding='causal')

    def forward(self, x):
        x = self.layer(x.swapdims(1, 2)).swapdims(1, 2)
        # (M, T, M) -> (F, T, M, 1)
        return x.swapdims(0, 2).unsqueeze(-1)


class CAConvV2(Module):
    def __init__(self, features: int = 512, kernel_size: int = 8, lag: int = 3):
        Module.__init__(self)
        self.k = kernel_size
        self.l = lag
        self.conv = Conv1d(features, kernel_size * features * lag, groups=512, kernel_size=3, padding='causal')
        # self.pw_mix = TDLayer(features, 3, kernel_size, kernel_size, padding='same')

    def forward(self, x):
        M, T, F = x.shape
        # print(M, T, F)
        x = self.conv(x.swapdims(1, 2)).swapdims(1, 2).view(M, T, F, self.k, self.l)
        x_new = torch.empty(self.l, M, T, F, self.k)
        for l in range(self.l):
            for i in range(self.k):
                x_new[l, :, :, :, i] = torch.roll(x[:, :, :, i, l], i+l, 2)
                x_new[l, :, :i+l, :, i] = 0.
        x_new = x_new.view(M * self.l, T, F, self.k)
        # print(x_new.shape)
        # x = self.pw_mix(x_new)
        # print(x.shape)
        return x_new.swapdims(0, 2)


class CALSTM(Module):
    def __init__(self, feature_dim: int):
        Module.__init__(self)
        self.layer = ResLSTM(feature_dim, 3)

    def forward(self, x):
        x = self.layer(x)  # M, T, F, 3
        k, q, v = torch.split(x.swapdims(0, 2), 1, dim=-1)
        return k, q, v


class ChannelAttentionV3(CoreNetwork):
    norm = InstanceNorm1d
    ResLSTM.norm = norm

    def __init__(self, chin: int, Wkqv: type[Module] = CALSTM):
        Module.__init__(self)

        self.norm1 = self.norm(chin)
        self.pw_mix = TDBlock(in_size=chin, channels=(1, 1, 1), kernel_size=5, strides=1, act_fn=ReLU())

        self.norm0 = self.norm(chin)
        self.lstm1 = ResLSTM(chin)
        assert chin == 512, chin

        self.att_norm = self.norm(512)
        self.Wkqv = Wkqv(512)

        # self.att = RecurrentCA(feature_dim=256, momentum=.05)
        self.att = ChannelAttention(.05)

        self.norm2 = self.norm(512)
        self.lstm2 = ResLSTM(512)

    def forward(self, x: Tensor, *args, **kwargs):
        # Pick a random reference microphone
        ref_idx = random.randrange(x.shape[0])

        # Some Layer
        x = self.norm1(x)  # M, C, T
        x = f.relu(squeezy(self.pw_mix, -1, x.swapdims(1, 2)))  # M, T, C

        x = swappy(self.norm0, 1, 2, x)
        x = self.lstm1(x)
        x = swappy(self.att_norm, 1, 2, x)  # M, T, C
        # y = x[None, ref_idx, ...]  # 1, T, C

        K, Q, V = self.Wkqv(x)  # F, T, M, d

        # Type I: (M, T, C) -> (C, T, M, 1)
        # K, Q, V = [x.swapdims(0, 2).unsqueeze(-1) for x in (K, Q, V)]

        # Type II: (M, T, C) -> (1, T, M*C, 1)

        # # Type V_I: (M, T, C) -> (C, T, [M@T, M@T-1], 1)
        # K1, K2, K3, Q, V1, V2, V3 = torch.split(KQV.swapdims(0, 2).unsqueeze(-1), 256)
        # K2, V2 = torch.roll(K2, 1, 1), torch.roll(V2, 1, 1)
        # K2[:, :1, :, :] = V2[:, :1, :, :] = 0
        # K3, V3 = torch.roll(K3, 2, 1), torch.roll(V3, 2, 1)
        # K3[:, :2, :, :] = V3[:, :2, :, :] = 0
        # K, V = torch.cat([K1, K2, K3], dim=2), torch.cat([V1, V2, V3], dim=2)

        o = self.att(K, Q, V)  # C, T, M', 1
        o = o.squeeze(-1).swapdims(0, 2)  # (M, T, C)
        # print(y_down.shape, o.shape)
        y = x + o

        y = swappy(self.norm2, 1, 2, y)
        y = self.lstm2(y)

        return y.swapdims(1, 2)


def realy(layer: Module |Sequence[Module], x: Tensor) -> Tensor | Sequence[Tensor]:
    if isinstance(layer, Sequence):
        x = torch.view_as_real(x)
        return [torch.view_as_complex(l(x)) for l in layer]
    return torch.view_as_complex(layer(torch.view_as_real(x)))


class ComplexChannelAttention(CoreNetwork):
    norm = InstanceNorm1d
    ResLSTM.norm = norm

    def __init__(self, chin: int):
        Module.__init__(self)
        assert chin == 513, chin
        self.out_features = 513

        self.norm1 = self.norm(chin)
        self.lstm1 = ResLSTM(chin, 2, 2)

        self.att_norm = self.norm(513)
        self.Wk, self.Wq, self.Wv = ResLSTM(chin, 2, 2), ResLSTM(chin, 2, 2), ResLSTM(chin, 2, 2)

        # self.att = RecurrentCA(feature_dim=256, momentum=.05)
        self.att = ChannelAttention(.05, causal=False)

        self.norm2 = self.norm(513)
        self.lstm2 = ResLSTM(513, 1, 2)

    def forward(self, x: Tensor, *args, **kwargs):
        assert x.is_complex(), x.dtype

        # Some Layer
        x = self.norm1(x)  # M, F, T
        x = realy(self.lstm1, x.swapdims(1, 2))  # M, T, F

        x = swappy(self.att_norm, 1, 2, x)  # M, T, F
        # y = x[None, ref_idx, ...]  # 1, T, C

        K, Q, V = realy([self.Wk, self.Wq, self.Wv], x)  # F, T, M, 1
        K, Q, V = [kqv.swapdims(0, 2).unsqueeze(-1) for kqv in [K, Q, V]]

        o = self.att(K, Q, V)  # C, T, M', 1
        o = o.squeeze(-1).swapdims(0, 2)  # (M, T, C)
        # print(y_down.shape, o.shape)
        y = x + o

        y = swappy(self.norm2, 1, 2, y)
        y = self.lstm2(torch.view_as_real(y))

        return y.swapdims(1, 2)


class BigBoy(Module):
    class AttentionBlock(Module):
        def __init__(self, feature_dim: int):
            Module.__init__(self)
            self.normx, self.normy = Norm1d([1, 1, feature_dim], [0, 1]), Norm1d([1, 1, feature_dim], [0, 1])
            self.Wk, self.Wq, self.Wv = [LSTM(feature_dim, feature_dim, batch_first=True) for _ in range(3)]
            self.att = ChannelAttention(momentum=.05, trainable=False)

        def forward(self, x, y):
            x, y = self.normx(x), self.normy(y)
            K, Q, V = self.Wk(x)[0], self.Wq(y)[0], self.Wv(x)[0]  # M, T2, C2

            # Type I: (M, T, C) -> (C, T, M, 1)
            # Type II: (M, T, C) -> (1, T, M*C, 1)
            K, Q, V = [x.swapdims(0, 2).unsqueeze(-1) for x in (K, Q, V)]
            o = self.att(K, Q, V)  # C, T, M', 1
            o = o.squeeze(-1).swapdims(0, 2)  # (M, T, C)

            return y + o

    def __init__(self):
        Module.__init__(self)
        self.encoder = b.ConvEncoder.build_encoder(3, 2, act_fn=ReLU)

        self.norm1 = Norm1d.bn(512)
        self.pw_mix = single_channel(TDBlock(in_size=512, channels=(1, 1, 1), kernel_size=5, act_fn=ReLU()))
        self.down1 = MaxPool1d(2)

        self.att_KV = self.AttentionBlock(256)
        self.att_Q = self.AttentionBlock(256)

        self.att = self.AttentionBlock(256)

        self.norm2 = Norm1d([1, 1, 256], [0, 1])
        self.lstm = LSTM(256, 64, batch_first=True)

        self.norm3 = Norm1d([1, 1, 64], [0, 1])
        self.fc = Linear(64, 1)

    def forward(self, x: Tensor, *args, **kwargs):
        # Pick a random reference microphone
        ref_idx = random.randrange(x.shape[0])

        assert x.shape[0] <= 5, x.shape
        x = self.encoder(x.unsqueeze(1))  # M, C, T = x.shape

        # Some Layer
        x = self.norm1(x)  # M, C, T
        x = f.relu(self.pw_mix(x.swapdims(1, 2)))  # M, T, C

        x = self.down1(x)  # M, T/2, C/2
        y = x[None, ref_idx, ...]  # 1, T/2, C/2

        x = self.att_KV(x, x)
        y = self.att_Q(x, y)

        y = self.att(x, y)

        y = self.norm2(y)
        y, _ = self.lstm(y)

        y = self.norm3(y)
        y = f.sigmoid(self.fc(y))

        return y.squeeze(-1)


class RecurrentChannelAttention(Module):
    class CA(Module):
        def __init__(self, momentum: float = .05):
            Module.__init__(self)

            self.Wk, self.Wv = ConvTDBlock(), ConvTDBlock()
            self.Wq = Sequential(
                Linear(512, 16),
                ReLU(),
                Linear(16, 512)
            )
            # self.Wv = Conv1d(in_channels=512, out_channels=512, kernel_size=5, groups=512, padding='causal')
            self.ema = EMACell(momentum=momentum, unbiased=True)

        def forward(self, x, y):
            # x: M, C, T
            # y: M', C, T

            K, V = self.Wk(x), self.Wv(x)
            # print(k.shape, q.shape, v.shape)  # (M, T, C, dk), (M, T, C, dv)
            K, V = K.swapdims(0, 2), V.swapdims(0, 2)  # (C, T, M, d)

            Mq, C, T = y.shape

            out = torch.empty_like(y)
            y_ = torch.zeros(Mq, C)
            self.ema.reset_state()
            for t in range(T):
                k, v = K[:, t, :, :], V[:, t, :, :]  # (C, M, d)
                q = self.Wq(y_ + y[:, :, t])  # (M', C)
                q = q.swapdims(0, 1).unsqueeze(-1)  # (C, M', 1)

                att = q @ k.swapdims(1, 2)  # (C, M', M)
                # print(att.shape)
                att = self.ema(att)

                # Normalize: Layer Norm
                att = att / (torch.norm(att, dim=[0, 2], keepdim=True) + 1e-12)

                y_ = (att @ v).squeeze(-1).swapdims(0, 1)  # (M', C)
                out[:, :, t] = y_

            return out

    def __init__(self):
        Module.__init__(self)
        self.encoder = b.ConvEncoder.build_encoder(3, 2)

        self.norm1 = Norm1d(512)
        self.l1 = ResLSTMConv(512, 16)

        self.att_norm = Norm1d(512)
        self.att1 = self.CA(momentum=.1)  # T = .32s; Tau = 10
        self.att2 = self.CA(momentum=.05)  # T = .64s; Tau = 20
        self.att5 = self.CA(momentum=.02)  # T = 1.6s; Tau = 50
        self.att3 = self.CA(momentum=.01)  # T = 3.2s; Tau = 100
        # self.att4 = self.CA(momentum=.005)  # T = 6.4s
        self.att_linear = TDBlock(in_size=512, channels=(4, 2, 1), kernel_size=6, strides=2, act_fn=f.relu)

        self.normfc1 = Norm1d(512)
        self.fc1 = ResLSTMConv(in_channels=512, hidden=16)
        self.normfc2 = Norm1d(512)
        self.fc2 = ResLSTMConv(in_channels=512, hidden=16, out_channels=1)

    def forward(self, x: Tensor, *args, **kwargs):
        # print(x.shape)
        assert x.shape[0] <= 5, x.shape
        x = self.encoder(x)  # M, C, T = x.shape

        # Some Layer
        x = self.norm1(x)
        x = self.l1(x)  # M, C, T

        x = self.att_norm(x)  # M, C, T
        # Pick a random reference microphone
        y = x[random.randrange(x.shape[0])].unsqueeze(0)

        # print(y.shape)  # 1, C, T
        y1 = self.att1(x, y)
        y2 = self.att2(x, y)
        y3 = self.att3(x, y)
        # y4 = self.att4(x, y)
        y5 = self.att5(x, y)

        y = torch.cat([y1, y2, y3, y5]).swapdims(0, 2)  # T, C, 3
        y = self.att_linear(y).swapdims(0, 2)  # 1, C, T

        # x = self.normfc1(y)
        # x = self.fc1(x)
        x = self.normfc2(y)
        x = f.sigmoid(self.fc2(x))

        return x.squeeze(1).squeeze(0)


class RecurrentChannelAttentionV2(Module):
    class CA(Module):
        def __init__(self, momentum: float = .05):
            Module.__init__(self)

            self.Wk, self.Wv, self.Wd = ConvTDBlock(), ConvTDBlock(), ConvTDBlock()
            # self.Wq = Sequential(
            #     Linear(512, 16),
            #     ReLU(),
            #     Linear(16, 512)
            # )
            # self.Wv = Conv1d(in_channels=512, out_channels=512, kernel_size=5, groups=512, padding='causal')
            self.ema = EMACell(momentum=momentum, unbiased=True)

        def forward(self, x, y):
            # x: M, C, T
            # y: M', C, T

            K, V, D = self.Wk(x), self.Wv(x), self.Wd(y)
            # print(k.shape, q.shape, v.shape)  # (M, T, C, dk), (M, T, C, dv)
            K, V, D = K.swapdims(0, 2), V.swapdims(0, 2), D.swapdims(0, 2)  # (C, T, M, d)

            Mq, C, T = y.shape
            M = x.shape[0]

            out = torch.empty_like(y)
            self.ema.reset_state()
            att = self.ema(torch.zeros(C, Mq, M))
            for t in range(T):
                k, v, d = K[:, t, :, :], V[:, t, :, :], D[:, t, :, :]  # (C, M, d)

                # Filter
                y = att @ v  # C, M', 1
                out[:, :, t] = y.squeeze(-1).swapdims(0, 1)

                e = d + y

                # Update
                att = self.ema(e @ k.swapdims(1, 2))  # (C, M', M)
                # Normalize: Layer Norm
                att = att / (torch.norm(att, dim=[0, 2], keepdim=True) + 1e-12)

            return out

    def __init__(self):
        Module.__init__(self)
        self.encoder = b.ConvEncoder.build_encoder(3, 2)

        self.norm1 = Norm1d(512)
        self.l1 = ResLSTMConv(512, 16)

        self.att_norm = Norm1d(512)
        # self.att1 = self.CA(momentum=.1)  # T = .32s; Tau = 10
        self.att2 = self.CA(momentum=.05)  # T = .64s; Tau = 20
        # self.att5 = self.CA(momentum=.02)  # T = 1.6s; Tau = 50
        # self.att3 = self.CA(momentum=.01)  # T = 3.2s; Tau = 100
        # self.att4 = self.CA(momentum=.005)  # T = 6.4s
        # self.att_linear = TDBlock(in_size=512, channels=(4, 2, 1), kernel_size=6, strides=2, act_fn=f.relu)

        # self.normfc1 = Norm1d(512)
        # self.fc1 = ResLSTMConv(in_channels=512, hidden=16)
        self.normfc2 = Norm1d(512)
        self.fc2 = ResLSTMConv(in_channels=512, hidden=16, out_channels=1)

    def forward(self, x: Tensor, *args, **kwargs):
        # print(x.shape)
        assert x.shape[0] <= 5, x.shape
        x = self.encoder(x)  # M, C, T = x.shape

        # Some Layer
        x = self.norm1(x)
        x = self.l1(x)  # M, C, T

        x = self.att_norm(x)  # M, C, T
        # Pick a random reference microphone
        y = x[random.randrange(x.shape[0])].unsqueeze(0)

        # print(y.shape)  # 1, C, T
        # y1 = self.att1(x, y)
        y2 = self.att2(x, y)
        # y3 = self.att3(x, y)
        # y4 = self.att4(x, y)
        # y5 = self.att5(x, y)

        # y = torch.cat([y1, y2, y3, y5]).swapdims(0, 2)  # T, C, 3
        # y = self.att_linear(y).swapdims(0, 2)  # 1, C, T

        # x = self.normfc1(y)
        # x = self.fc1(x)
        x = self.normfc2(y2)
        x = f.sigmoid(self.fc2(x))

        return x.squeeze(1).squeeze(0)


class RecurrentChannelAttentionV3(Module):
    class CA(Module):
        def __init__(self, feature_dim: int, momentum: float = .05):
            Module.__init__(self)

            self.Wk = LSTM(feature_dim, feature_dim, batch_first=True)
            self.Wq = LSTM(feature_dim, feature_dim, batch_first=True)
            self.Wv = LSTM(feature_dim, feature_dim, batch_first=True)

            self.We = Linear(2 * feature_dim, feature_dim)

            self.ema = EMACell(momentum=momentum, unbiased=True)

        def forward(self, x, y):
            # x: M, T, C
            # y: M', T, C

            K, Q, V = self.Wk(x), self.Wv(x), self.Wq(y)
            K, V = K.swapdims(0, 2), V.swapdims(0, 2)

            Mq, T, C = y.shape
            M = x.shape[0]

            out = torch.empty_like(y)
            att = torch.ones([C, Mq, M])
            # y_ = torch.zeros(Mq, C)
            self.ema.reset_state()
            for t in range(T):
                # 1) Filter
                # k,v: (C, M, d);  q: (1, C)
                k, q, v = K[:, t, :, None], Q[:, t, :], V[:, t, :, None]  # (C, M, d)
                out[:, :, t] = (att @ v).squeeze(-1).swapdims(0, 1)  # (M', C)

                # 2) Update
                q = self.We(torch.cat([q, out[:, :, t]], dim=1))  # (M', C)
                q = q.swapdims(0, 1).unsqueeze(-1)  # (C, M', 1)

                att = q @ k.swapdims(1, 2)  # (C, M', M)
                # print(att.shape)
                att = self.ema(att)

                # Normalize: Layer Norm
                att = att / (torch.norm(att, dim=[0, 2], keepdim=True) + 1e-12)

            return out

    def __init__(self):
        Module.__init__(self)
        self.encoder = b.ConvEncoder.build_encoder(3, 2)

        self.norm1 = Norm1d(512)
        self.l1 = ResLSTMConv(512, 16)

        self.att_norm = Norm1d(512)
        self.att1 = self.CA(momentum=.1)  # T = .32s; Tau = 10
        self.att2 = self.CA(momentum=.05)  # T = .64s; Tau = 20
        self.att5 = self.CA(momentum=.02)  # T = 1.6s; Tau = 50
        self.att3 = self.CA(momentum=.01)  # T = 3.2s; Tau = 100
        # self.att4 = self.CA(momentum=.005)  # T = 6.4s
        self.att_linear = TDBlock(in_size=512, channels=(4, 2, 1), kernel_size=6, strides=2, act_fn=f.relu)

        self.normfc1 = Norm1d(512)
        self.fc1 = ResLSTMConv(in_channels=512, hidden=16)
        self.normfc2 = Norm1d(512)
        self.fc2 = ResLSTMConv(in_channels=512, hidden=16, out_channels=1)

    def forward(self, x: Tensor, *args, **kwargs):
        # print(x.shape)
        assert x.shape[0] <= 5, x.shape
        x = self.encoder(x)  # M, C, T = x.shape

        # Some Layer
        x = self.norm1(x)
        x = self.l1(x)  # M, C, T

        x = self.att_norm(x)  # M, C, T
        # Pick a random reference microphone
        y = x[random.randrange(x.shape[0])].unsqueeze(0)

        # print(y.shape)  # 1, C, T
        y1 = self.att1(x, y)
        y2 = self.att2(x, y)
        y3 = self.att3(x, y)
        # y4 = self.att4(x, y)
        y5 = self.att5(x, y)

        y = torch.cat([y1, y2, y3, y5]).swapdims(0, 2)  # T, C, 3
        y = self.att_linear(y).swapdims(0, 2)  # 1, C, T

        # x = self.normfc1(y)
        # x = self.fc1(x)
        x = self.normfc2(y)
        x = f.sigmoid(self.fc2(x))

        return x.squeeze(1).squeeze(0)


class MWFBasedCA(Module):
    def __init__(self, causal: bool = False):
        Module.__init__(self)

        self.causal = causal

        self.enc = b.ConvEncoder((2**8, 2**6, 2**4), (2**4, 2**3, 2**2), output_channels=(2**4, 2**3, 2**3))

        self.norm1 = Norm1d.bn(2*512)

        self.x2x = LSTM(input_size=2*512, hidden_size=512, batch_first=True)
        self.x2n = LSTM(input_size=2*512, hidden_size=512, batch_first=True)
        self.x2y = LSTM(input_size=2*512, hidden_size=512, batch_first=True)

        self.ema = ExponentialMovingAverage(.05)

        self.norm3 = Norm1d((1, 1, 512), (0, 1))
        self.lstm = LSTM(512, 256, batch_first=True)

        self.norm2 = Norm1d(None, (0, 1))
        self.fc = Linear(in_features=256, out_features=1)

    def forward(self, x, vad=None, **kwargs):
        M = x.shape[0]
        ref_mic = random.randrange(M)
        # print(f"{M=}")
        x = self.enc(x.unsqueeze(1))

        x = self.norm1(x).swapdims(1, 2)  # M, T, F
        x_ref = x[None, ref_mic, ...]
        x, d, y = self.x2x(x)[0], self.x2n(x_ref)[0], self.x2y(x)[0]
        # print(x.shape)

        if self.causal:
            x, d = x.swapdims(0, 2).unsqueeze(-1), d.swapdims(0, 2).unsqueeze(-1)  # F, T, M, 1
            Rxx, Rxd = x @ x.mT, x @ d.mT  # (F, T, M, M), (F, T, M, 1)
            # print(Rxx.shape)
            Rxx, Rxd = self.ema(Rxx.swapdims(1, -1)).swapdims(1, -1), self.ema(Rxd.swapdims(1, -1)).swapdims(1, -1)
        else:
            x, d = x.swapdims(0, 2), d.swapdims(0, 2)  # F, T, M
            T = x.shape[1]
            Rxx, Rxd = (x.mT @ x) / T, (x.mT @ d) / T  # F, M, M
            Rxx, Rxd = Rxx.unsqueeze(1), Rxd.unsqueeze(1)  # F, 1, M, M
        # print(Rxx.shape)

        # MWF - or not hahaha
        w = torch.linalg.solve(Rxx, Rxd).squeeze(-1)
        w = w / torch.linalg.norm(w, dim=[0, 2], keepdim=True)
        # print(w.shape)  # F, T, M

        # w: (F, T, 1, M) @ y: (F, T, M, 1) -> (F, T, 1, 1)
        y = (w.unsqueeze(2) @ y.swapdims(0, 2).unsqueeze(-1)).squeeze(-1)
        # print(y.shape)  # F, T,

        y = self.norm3(y.swapdims(0, 2))
        y = self.lstm(y)[0]

        y = self.norm2(y)  # 1, T, F
        y = f.sigmoid(self.fc(y))
        return y.squeeze(-1)


############################################
# Mixing DNN with linear signal processing #
############################################
from Acoustics.signal_processing import frequency_domain_mwf, get_correlation_matrices, NoStatisticsException


class TimeDomainMWF(Module):
    def __init__(self, time_domain_vad: Module):
        Module.__init__(self)

        # Single channel VAD
        self.vad = time_domain_vad

    def forward(self, x, **kwargs):
        B, T_ = x.shape

        vad = self.vad(x.unsqueeze(1))
        # print(x.shape, vad.shape)

        # Get correlation matrices
        Rxx, Rnn = get_correlation_matrices(x, vad, 512)

        # plt.figure()
        # plt.title("Spatio-Temporal Noise Covariance Matrix")
        # plt.imshow(Rnn.detach().numpy(), interpolation='none')
        # plt.show()

        # Compute optimal filter - non causal
        w = (torch.eye(Rxx.shape[0]) - torch.linalg.solve(Rxx, Rnn))[:, 0].view(1, B, 512)
        x = x.unsqueeze(0)
        # print(x.shape, w.shape)

        x = f.conv1d(x, w, padding=511)[:, :, :-511]
        # print(x.shape)

        return self.vad(x)


class ChickenAndEgg(Module):
    def __init__(self):
        Module.__init__(self)

        self.enc = b.STFT()
        # self.vad1 = b.FC(513)
        self.vad2 = b.M2LSTM(bidirectional=True)

    def forward(self, x, vad=None, ann=None, **kwargs):
        x = self.enc(x)
        # if vad is None:
        #     vad = self.vad1(x.abs())
        assert vad is not None

        ref_mic = random.randrange(x.shape[0])

        # rirs = ann['RIRs']
        # microphones, sources = len(rirs), len(rirs[0])
        # print(f"{sources=}, {microphones=}")
        # A = torch.empty((self.enc.num_channels, microphones, sources), dtype=torch.cfloat)
        # direct_delay = torch.inf
        # for row in rirs:
        #     for i in range(len(row)):
        #         row[i] = torch.tensor(row[i], dtype=torch.float)
        #         direct_delay = min(direct_delay, torch.min(torch.argwhere(row[i] != 0.)))
        # print(f"{direct_delay=}")
        # for s in range(sources):
        #     for m in range(microphones):
        #         A[:, m, s] = torch.fft.rfft(rirs[m][s][direct_delay:direct_delay + self.enc._nfft], self.enc._nfft)
        #
        # # Don't fully dereverb, target = source signal as in reference microphone
        # A /= A[:, ref_mic, None, :]
        #
        # b = torch.zeros((sources, 1), dtype=torch.cfloat)
        # b[0] = 1.

        try:
            # x_filt = LCMV_beamformer(x, vad, A, b)
            x_filt = frequency_domain_mwf(x, vad, ref_mic)
        except NoStatisticsException:
            x_filt = x[ref_mic].unsqueeze(0)

        # N = round(GlobalConfig.frame_rate * 5)
        # fig, (ax1, ax2) = plt.subplots(2, 1)
        # x_win, (offset, num_samples) = utils.utils.random_window(x[ref_mic].detach(), N, dim=-1)
        # t_frame = (offset + torch.arange(num_samples)) / GlobalConfig.frame_rate
        # t_sample = (GlobalConfig.win_size * offset + torch.arange(num_samples * GlobalConfig.win_size)) / GlobalConfig.sample_rate
        # ax1.plot(t_sample[::2], self.enc.istft(x_win)[::2])
        # x_win = utils.utils.window(x_filt[0].detach(), offset, num_samples, dim=-1)
        # ax2.plot(t_sample[:-2:2], self.enc.istft(x_win)[:-2:2])
        # ax2.plot(t_frame, .1 * utils.utils.window(vad[0], offset, num_samples))
        # plt.show()

        return self.vad2(x_filt.abs())

from torch.nn import Module

from config import GlobalConfig


class Encoder(Module):
    out_features: int = GlobalConfig.win_size


class CoreNetwork(Module):
    out_features: int = GlobalConfig.win_size


class FinalLayer(Module):
    pass

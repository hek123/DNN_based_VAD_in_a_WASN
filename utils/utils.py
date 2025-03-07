import random
from typing import Callable

import torch
from torch import Tensor
import matplotlib.pyplot as plt
import torchaudio.transforms as tt
import torchaudio.functional as tf
import numpy as np

from torch_framework.config import GlobalConfig


def ceildiv(a, b):
    return -(a // -b)


def isPow2(x: int):
    assert isinstance(x, int), type(x)
    assert x > 0, x
    return (x & (x - 1)) == 0


def sequential(*functions: Callable) -> Callable:
    def wrapper(x):
        for f in functions:
            x = f(x)
        return x
    return wrapper


def random_window(x: torch.Tensor, num_samples: int, dim: int = 0):
    num_samples = min(num_samples, x.shape[dim])
    start_idx = random.randint(0, x.shape[dim] - num_samples)

    return torch.narrow(x, start=start_idx, length=num_samples, dim=dim), (start_idx, num_samples)


def window(x: torch.Tensor, offset: int, num_samples: int, dim: int = 0):
    assert num_samples <= x.shape[dim] - offset, f"x: {x.shape[dim]}, {num_samples=}, {offset=}"
    return torch.narrow(x, start=offset, length=num_samples, dim=dim)


def _vad_window(audio: torch.Tensor, vad: torch.Tensor, duration: float):
    num_frames = round(duration * GlobalConfig.frame_rate)
    vad, (offset, num_frames) = random_window(vad, num_frames)
    audio = window(audio, GlobalConfig.win_size * num_frames, GlobalConfig.win_size * offset)
    return audio, vad


def plot_spectrum(x: torch.Tensor, nfft: int, ax: plt.Axes = None):
    if ax is None:
        _, ax = plt.subplots()

    X = tf.spectrogram(x, pad=0, window=torch.hann_window(nfft), n_fft=nfft, hop_length=nfft // 2, win_length=nfft,
                       power=None, normalized=False, center=True, pad_mode="reflect", onesided=True)
    S = torch.abs(X)
    # print(S.shape)
    ax.imshow(20 * torch.log10(S + 1e-8), aspect='auto', interpolation='none')

    return ax


def plot_vad(audio: torch.Tensor, vad: torch.Tensor, ax: plt.Axes = None):
    assert len(audio.shape) == 1
    assert len(vad.shape) == 1
    assert audio.shape[0] == GlobalConfig.win_size * vad.shape[0]

    if ax is None:
        _, ax = plt.subplots()

    assert audio.shape[0] == vad.shape[0] * GlobalConfig.win_size

    ax.plot(torch.arange(0, audio.shape[0]) / GlobalConfig.sample_rate, audio)
    ax.plot((torch.arange(0, vad.shape[0]) + .5) / GlobalConfig.frame_rate, 0.2 * vad)

    return ax


def select(audio: Tensor, vad: Tensor) -> tuple[Tensor, Tensor]:
    assert len(audio.shape) == len(vad.shape)
    assert audio.shape[-1] == GlobalConfig.win_size * vad.shape[-1]
    assert is_binary(vad)

    x = audio.view(*audio.shape[:-1], -1, GlobalConfig.win_size)
    # assert torch.all(x[0, :] == audio[:Config.win_size])

    return x[..., vad == 1, :], x[..., vad == 0, :]


def is_binary(x: torch.Tensor) -> bool:
    return bool(torch.all(torch.logical_or(x == 0., x == 1.)))


def is_probability(x: torch.Tensor) -> bool:
    return bool(torch.all(torch.logical_and(0 <= x, x <= 1)))


class Format:
    @staticmethod
    def dB(value: float):
        return f"{value: .2f}dB"

    @staticmethod
    def percent(value: float):
        return f"{100 * value: 2.2f}%"


def shape_psd(get_shape: Callable[[np.ndarray], np.ndarray]):
    def noise_generator(size: int) -> torch.Tensor:
        X_white = np.fft.rfft(np.random.randn(size))
        S = get_shape(np.fft.rfftfreq(size, 1 / GlobalConfig.sample_rate))
        X_shaped = X_white * S / np.sqrt(np.mean(S ** 2))
        return torch.tensor(np.fft.irfft(X_shaped))
    return noise_generator


@shape_psd
def pink_noise(f):
    return 1/np.where(f == 0, float('inf'), np.sqrt(f))

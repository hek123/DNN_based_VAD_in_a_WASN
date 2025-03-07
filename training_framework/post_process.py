import math
import random
from abc import ABC, abstractmethod
from typing import Callable, Sequence

import torch
from torch import Tensor
import matplotlib.pyplot as plt
from neptune import Run

import utils.utils as utils
from config import GlobalConfig
from Acoustics.signal_processing import snr

Sample = tuple[Tensor, Tensor, dict]


class PostProcess(ABC):
    @abstractmethod
    def __call__(self, audio: Tensor, vad_spp: Tensor, annotations: dict, log: Run) -> tuple[Tensor, Tensor]:
        pass


class ComputeSNR(PostProcess):
    def __init__(self, name: str, eps: float = 1e-12):
        self.name = name
        self.eps = eps

    def __call__(self, audio: Tensor, vad: Tensor, annotations: dict, log: Run):
        assert annotations['label_type'] == 'vad'
        S, N = snr(audio, vad)

        # print(f"{num1=}, {num0=}")
        # print(f"{S=}, {N=}, T={torch.var(x)}")
        # print(torch.mean(x_[y==1, :]), torch.mean(x_[y==0, :]))

        assert self.name not in annotations
        annotations[self.name] = {'S': S, 'N': N, 'SNR': 10 * math.log10(S / (N + 1e-8))}

        if log is not None:
            log[f'data/{self.name}'].append(annotations[self.name]['SNR'])

        return audio, vad


class AddNoise(PostProcess):
    def __init__(self, noise_generator: "NoiseGenerator", name: str, add_to_current_noise: bool = False,
                 log: Run = None):

        self.noise_generator = noise_generator
        self.name = name
        if log is not None:
            log[f'config/data/noise/{name}'] = {'snr': str((noise_generator.snr_min, noise_generator.snr_max))}
        self.ignore_noise = not add_to_current_noise

        # print(f"{noise_type}: {snr: 2.2f}dB")

    def __call__(self, x: Tensor, y: Tensor, annotations: dict, log: Run):
        assert len(x.shape) == 1
        assert 'SNR_clean' in annotations, f"Clean SNR not found, make sure to proceed Noise operation by ComputeSNR"
        S, N = annotations['SNR_clean']['S'], annotations['SNR_clean']['N']

        noise = self.noise_generator(x.shape)
        noise, snr = self.noise_generator.scale(noise, S, None if self.ignore_noise else N)
        x += noise

        if log is not None:
            log[f'data/noise/{self.name}'].append(snr)
        return x, y


class NoiseGenerator(ABC):
    def __init__(self, snr: float | Sequence[float] | Callable[[], float]):
        if isinstance(snr, float) or isinstance(snr, int):
            self.snr = lambda: snr
        elif isinstance(snr, Sequence):
            if len(snr) == 2:
                self.snr = lambda: random.uniform(*snr)
            elif len(snr) == 3:
                # self.snr = lambda: random.triangular(*snr)
                a, b, mode = snr
                self.snr = lambda: a + (b - a) * random.betavariate(1 + 4 * (mode - a) / (b - a), 1 + 4 * (b - mode) / (b - a))
            else:
                raise AssertionError(f"{snr=}")
        elif isinstance(snr, Callable):
            self.snr = snr
        else:
            raise AssertionError(f"{snr=}, {type(snr)=}")

    @abstractmethod
    def __call__(self, size: Sequence[int]) -> Tensor:
        pass

    def scale(self, x: Tensor, S: float, N: float = None, eps=1e-8) -> tuple[Tensor, float]:
        assert S > 0, S
        assert N is None or N >= 0, N

        snr = self.snr()
        snr_linear = 10 ** (snr / 10)
        scale = max(S / snr_linear if N is None else S / snr_linear - N, eps)
        # print(snr, scale)
        return math.sqrt(scale) * x, snr


class SpectralNoise(NoiseGenerator):
    def __init__(self, spectrum: Tensor, snr: float | tuple[float, float], nfft: int):
        assert len(spectrum.shape) == 1
        assert utils.isPow2(nfft)
        assert spectrum.shape[0] == nfft // 2 + 1

        NoiseGenerator.__init__(self, snr)

        self.S = torch.sqrt(spectrum[:, None])
        self.S /= torch.norm(self.S)

        # plt.figure()
        # plt.plot(self.S)
        # plt.show()

        self.nfft = nfft
        self._win = torch.sqrt(torch.hann_window(self.nfft))

    def __call__(self, size: Sequence[int]) -> Tensor:
        assert len(size) == 1 or len(size) == 2

        X = torch.stft(torch.randn(size), n_fft=self.nfft, hop_length=self.nfft // 2, win_length=self.nfft,
                       window=self._win, center=True, pad_mode='constant', normalized=True, onesided=True,
                       return_complex=True)

        # print(X.shape, self.S.shape, (X*self.S).shape)
        # print(X.dtype, self.S.dtype, (X*self.S).dtype)

        x = torch.istft(X * self.S, n_fft=self.nfft, hop_length=self.nfft // 2, win_length=self.nfft,
                        window=self._win, center=True, normalized=True, onesided=True, length=size[-1],
                        return_complex=False)

        # plt.figure()
        # x_window, _ = utils.random_window(x, Config.sample_rate * 10)
        # plt.plot(x_window)
        # plt.show()

        return x


class ColoredNoise(NoiseGenerator):
    def __init__(self, color: str | float, snr: float | tuple[float, float]):
        NoiseGenerator.__init__(self, snr)
        if isinstance(color, float) or isinstance(color, int):
            assert 0 <= color <= 2
            self.beta = color
        elif isinstance(color, str):
            self.beta = {'white': 0, 'pink': 1, 'brown': 2, 'blue': -1}[color]
        else:
            raise AssertionError(f"{color=}, {type(color)=}")

    def spectral_noise(self, size: Sequence[int]) -> Tensor:
        assert len(size) == 1 or len(size) == 2
        X = torch.fft.rfft(torch.randn(size))
        f = torch.fft.rfftfreq(size[-1], d=1 / GlobalConfig.sample_rate)
        S = torch.pow(f, -self.beta / 2)
        S[f == 0] = 0
        S /= torch.norm(S)
        # plt.figure()
        # plt.plot(S)
        # plt.show()
        return S * X

    def __call__(self, size: Sequence[int]) -> Tensor:
        if self.beta == 0:
            return torch.randn(size)
        else:
            return torch.fft.irfft(self.spectral_noise(size))


class ClassBalance(PostProcess):
    def __call__(self, audio: Tensor, vad: Tensor, annotations: dict, log: Run):
        assert annotations['label_type'] == 'vad'

        annotations['class_balance'] = float(torch.sum(vad == 1) / vad.shape[0])

        return audio, vad


class AddSilence(PostProcess):
    def __init__(self, p_add: float):
        self.p_add = p_add

    def __call__(self, audio: Tensor, vad_spp: Tensor, annotations: dict, log: Run):
        assert 'offset' in annotations
        assert 'durations' in annotations

        split_indices = torch.cumsum(annotations['durations'], dim=0) - annotations['offset']
        split_indices = list(split_indices[(0 < split_indices) * (split_indices < vad_spp.shape[0])])
        # split_indices += [0, vad_spp.shape[0]]  # add start and end for possible split

        print(len(split_indices))
        split_indices = [x for x in split_indices if random.random() < self.p_add]
        print(len(split_indices))
        durations = [round(GlobalConfig.frame_rate * math.exp(.5 * torch.randn(1))) for _ in range(len(split_indices))]
        print(durations)

        split_audio = torch.tensor_split(audio.view(-1, GlobalConfig.win_size), split_indices, dim=0)
        split_vad_spp = torch.tensor_split(vad_spp, split_indices)
        audio_durations = [x.shape[0] for x in split_vad_spp]

        new_duration = sum(durations) + vad_spp.shape[0]
        audio, vad_spp = torch.empty(new_duration, GlobalConfig.win_size), torch.empty(new_duration)

        idx = 0
        audio[:audio_durations[0]] = split_audio[0]
        vad_spp[:audio_durations[0]] = split_vad_spp[0]
        idx += split_indices[0]
        for k, (d_silence, d_audio) in enumerate(zip(durations, audio_durations[1:])):
            audio[idx:idx+d_silence] = 0
            vad_spp[idx:idx+d_silence] = 0
            idx += d_silence

            audio[idx:idx+d_audio] = split_audio[k]
            vad_spp[idx:idx+d_audio] = split_vad_spp[k]
            idx += d_audio

        audio = audio.view(-1)

        assert len(audio.shape) == 1
        assert audio.shape[0] == vad_spp.shape[0] * GlobalConfig.win_size

        del annotations['durations']

        return audio, vad_spp


class PlotSample(PostProcess):
    def __init__(self, duration: float):
        self.num_frames = round(duration * GlobalConfig.frame_rate)

    def __call__(self, audio, vad_spp, ann, log: Run):
        vad_spp_window, (offset, num_frames) = utils.random_window(vad_spp, self.num_frames)
        audio_window = utils.window(audio, offset=GlobalConfig.win_size * offset, num_samples=GlobalConfig.win_size * num_frames)

        utils.plot_vad(audio_window, vad_spp_window)
        plt.show()

        return audio, vad_spp

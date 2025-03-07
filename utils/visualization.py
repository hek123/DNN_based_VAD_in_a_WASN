import random
from typing import Sequence, Callable
import math

from matplotlib import pyplot as plt
from torch import Tensor, arange, narrow
import torch

from config import GlobalConfig
# import utils.utils as utils


def _plot_prediction(audio: Tensor, label: Tensor, prediction: Tensor, ax1: plt.Axes, ax2: plt.Axes,
                     num_samples: int, start_idx: int):
    assert len(audio.shape) == len(label.shape) == len(prediction.shape) == 1

    audio = narrow(audio, start=start_idx, length=num_samples, dim=0)
    scale = torch.max(audio)
    t_sample = (start_idx + arange(0, num_samples)) / GlobalConfig.sample_rate
    ax1.plot(t_sample, audio, 'k', label='audio', linewidth=.5, alpha=.6)

    # print("Vad")
    num_samples //= GlobalConfig.win_size
    start_idx //= GlobalConfig.win_size
    t_frame = (start_idx + arange(0, num_samples) + .5) / GlobalConfig.frame_rate
    label = narrow(label, start=start_idx, length=num_samples, dim=0)
    prediction = narrow(prediction, start=start_idx, length=num_samples, dim=0)
    # for name in y:
    #     y[name] = torch.narrow(y[name], start=start_idx, length=num_samples, dim=0)
    scale = 1.
    ax2.plot(t_frame, scale * label, label='ground truth')  # ax1 ...
    ax2.plot(t_frame, scale * prediction, label='prediction')
    # for name, yi in y.items():
    #     ax1.plot(t_frame, scale * yi, label=name)
    ax2.legend()

    # # Visualize prediction with heuristic
    # ax2.plot(t_frame, 1.1 * label, label='ground truth')
    # ax2.plot(t_frame, torch.round(prediction), label='threshold')
    # for name, yi in y.items():
    #     ax2.plot(t_frame, torch.round(yi), label=name)
    # # ax2.plot(t_frame, spp2vad(speech_pad_ms=0)(prediction), label='heuristic')
    # ax2.legend()


def visualize_prediction(audio: Tensor, label: Tensor, prediction: Tensor, T: int = 10,
                         **heuristic: Callable[[Tensor], Tensor]):
    assert len(audio.shape) == len(label.shape) == len(prediction.shape) == 1
    assert audio.shape[0] == GlobalConfig.win_size * label.shape[0] == GlobalConfig.win_size * prediction.shape[0] or \
           audio.shape[0] == label.shape[0] == prediction.shape[0], (audio.shape, label.shape, prediction.shape)

    y = {}
    for name, fn in heuristic.items():
        y[name] = fn(prediction).squeeze()
        assert y[name].shape == prediction.shape, (y[name].shape, prediction.shape, fn)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col')

    num_samples = min(round(T * GlobalConfig.frame_rate), label.shape[0]) * GlobalConfig.win_size
    start_idx = random.randint(0, audio.shape[0] - num_samples)

    aud = narrow(audio, start=start_idx, length=num_samples, dim=0)
    scale = torch.max(aud)
    t_sample = (start_idx + arange(0, num_samples)) / GlobalConfig.sample_rate
    ax1.plot(t_sample, aud, 'k', label='audio', linewidth=.5, alpha=.6)

    # print(audio.shape, label.shape, prediction.shape)
    if label.shape[0] == audio.shape[0]:
        # Visualize prediction
        ax2.plot(t_sample, narrow(label, start=start_idx, length=num_samples, dim=0), label='ground truth', alpha=.5)
        ax2.plot(t_sample, narrow(prediction, start=start_idx, length=num_samples, dim=0), label='prediction', alpha=.5)
        ax2.legend()
    else:
        # print("Vad")
        num_samples //= GlobalConfig.win_size
        start_idx //= GlobalConfig.win_size
        t_frame = (start_idx + arange(0, num_samples) + .5) / GlobalConfig.frame_rate
        label = narrow(label, start=start_idx, length=num_samples, dim=0)
        prediction = narrow(prediction, start=start_idx, length=num_samples, dim=0)
        for name in y:
            y[name] = torch.narrow(y[name], start=start_idx, length=num_samples, dim=0)
        scale = 1.
        ax2.plot(t_frame, scale * label, label='ground truth')  # ax1 ...
        ax2.plot(t_frame, scale * prediction, label='prediction')
        # for name, yi in y.items():
        #     ax1.plot(t_frame, scale * yi, label=name)
        ax2.legend()

        # # Visualize prediction with heuristic
        # ax2.plot(t_frame, 1.1 * label, label='ground truth')
        # ax2.plot(t_frame, torch.round(prediction), label='threshold')
        # for name, yi in y.items():
        #     ax2.plot(t_frame, torch.round(yi), label=name)
        # # ax2.plot(t_frame, spp2vad(speech_pad_ms=0)(prediction), label='heuristic')
        # ax2.legend()

    return fig


def err_distribution(train, val):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle("Violin plot")
    fig.tight_layout()

    ax1.violinplot(100 * train.numpy(), vert=False, showextrema=False, positions=[0])
    ax1.set_xlim([0, 100])
    ax1.set_title("Train Error")

    ax2.violinplot(100 * val.numpy(), vert=False, showextrema=False, positions=[0])
    ax2.set_xlim([0, 100])
    ax2.set_title("Validation Error")

    return fig


def _get_minmax_snr(audio: Tensor, label: Tensor) -> tuple[int, int]:
    x = audio.view(audio.shape[0], -1, GlobalConfig.win_size)
    N0, N1 = torch.count_nonzero(label == 0.), torch.count_nonzero(label == 1.)

    if not (N0 > 2 and N1 > 2):
        return random.choices(range(audio.shape[0]), k=2)

    snr = torch.empty(audio.shape[0])
    for m in range(audio.shape[0]):
        S = torch.var(x[m, label[m] == 1])
        N = torch.var(x[m, label[m] == 0])
        snr[m] = 10 * math.log(S / (N + 1e-8))
    # print(snr)
    min_snr, max_snr = torch.argmin(snr), torch.argmax(snr)
    # print(min_snr, max_snr)
    return min_snr, max_snr


def viz_pred_distributed(audio: Tensor, label: Tensor, pred: Tensor, T: int = 10):
    assert len(audio.shape) == len(label.shape) == len(pred.shape) == 2
    assert audio.shape[0] == pred.shape[0] == label.shape[0], (audio.shape, label.shape, pred.shape)
    assert audio.shape[1] == GlobalConfig.win_size * label.shape[1] == GlobalConfig.win_size * pred.shape[1] or \
           audio.shape[1] == label.shape[1] == pred.shape[1], (audio.shape, label.shape, pred.shape)

    # min_snr, max_snr = _get_minmax_snr(audio, label)
    min_snr, max_snr = 0, -1

    fig: plt.Figure
    axes: list[list[plt.Axes]]
    fig, axes = plt.subplots(2, 2, sharex='col')

    num_samples = min(round(T * GlobalConfig.frame_rate), label.shape[1]) * GlobalConfig.win_size
    start_idx = random.randint(0, audio.shape[1] - num_samples)

    _plot_prediction(audio[min_snr], label[min_snr], pred[min_snr], axes[0][0], axes[1][0], num_samples, start_idx)
    axes[0][0].set_title("low SNR node")
    _plot_prediction(audio[max_snr], label[max_snr], pred[max_snr], axes[0][1], axes[1][1], num_samples, start_idx)
    axes[0][1].set_title("high SNR node")

    return fig


def plot_rirs(rirs):
    mics, sources = len(rirs), len(rirs[0])
    axes: list[list[plt.Axes]]
    fig, axes = plt.subplots(mics, sources)
    for m in range(mics):
        for s in range(sources):
            axes[m][s].plot(rirs[m][s])

    plt.show()


def plot_vad_on_audo(audio: Tensor, vad: Sequence[Tensor], T: float = math.inf, ax: plt.Axes = None,
                     vad_names: Sequence[str] = None):
    assert len(vad) > 0, vad
    assert len(audio.shape) == len(vad[0].shape) == 1, (audio.shape, vad[0].shape)
    assert all(y.shape == vad[0].shape for y in vad), [y.shape for y in vad]

    if ax is None:
        _, ax = plt.subplots()

    num_frames = round(min(T * GlobalConfig.frame_rate, vad[0].shape[0]))
    start_frame = random.randint(0, vad[0].shape[0] - num_frames)
    num_samples, start_idx = num_frames * GlobalConfig.win_size, start_frame * GlobalConfig.win_size

    audio = narrow(audio, start=start_idx, length=num_samples, dim=0)
    scale = torch.max(audio)
    t_sample = (start_idx + arange(0, num_samples)) / GlobalConfig.sample_rate
    ax.plot(t_sample[::2], audio[::2], label='audio', linewidth=.5, alpha=.6)

    t_frame = (start_frame + arange(0, num_frames) + .5) / GlobalConfig.frame_rate
    a = torch.linspace(.95, 1.05, len(vad))
    for i in range(len(vad)):
        y = narrow(vad[i], start=start_frame, length=num_frames, dim=0)
        ax.plot(t_frame, a[i] * scale * y, label='VAD' if vad_names is None else vad_names[i])  # ax1 ...
        ax.legend()


if __name__ == '__main__':
    pass

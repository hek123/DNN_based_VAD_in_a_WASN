import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
from torch import Tensor
import torchaudio
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy import signal

from tqdm import tqdm
from silero_vad.utils_vad import timestamps_to_full_vad, prob_to_vad, timestamps2vad
from config import GlobalConfig, default_config


@dataclass
class DatasetPaths:
    root = "C:\\Users\\hekto\\PycharmProjects\\MyThesis\\code\\data"
    libri_speech = os.path.join(root, "train-clean-100", "LibriSpeech", "train-clean-100")
    ls_concat = os.path.join(root, "LibriSpeechConcat")
    vad_spp = os.path.join(root, "preprocessed")


def default_path():
    return DatasetPaths()


class LibriSpeech(Dataset):
    def __init__(self, labels: "GetLabels", size: int = None, raw_ls=False, paths: DatasetPaths = default_path(),
                 config: GlobalConfig = default_config()):

        self.ls_folder = paths.libri_speech if raw_ls else paths.ls_concat

        # self.vad_folder = None if vad_folder is None else os.path.join(paths.vad_spp, vad_folder)
        self.get_label = labels

        self.paths = paths

        self.fs = config.sample_rate
        self.cfg = config
        # self.preprocess = [] if preprocess is None else preprocess

        self.files = []
        for s in os.listdir(self.ls_folder):
            self.files += [os.path.join(s, t) for t in os.listdir(os.path.join(self.ls_folder, s))]
        self.size = len(self.files) if size is None else min(len(self.files), size)
        print(f"size = {len(self)}, full dataset: {len(self.files)}")

    def __getitem__(self, idx):
        assert 0 <= idx <= len(self)
        if idx == len(self):
            raise StopIteration
        audio_folder = os.path.join(self.ls_folder, self.files[idx])
        file = self.files[idx].replace('\\', '-')
        audio_file = "speech-" + file + ".wav"
        annotation_file = "annotation-" + file + ".npz"
        # vad_file = file + ".npy"
        data, fs = torchaudio.load(os.path.join(audio_folder, audio_file))
        num_channels = data.shape[0]
        assert num_channels == 1, f"multiple channels not supported, num_ch = {num_channels}"
        assert fs == self.fs, f"invalid sample frequency: {fs}, expected {self.fs}"
        data = torch.squeeze(data, dim=0)

        # load annotations (split indices)
        annotations = dict(np.load(os.path.join(audio_folder, annotation_file)))
        assert "idx" not in annotations
        annotations["idx"] = idx
        annotations["filename"] = file

        # if self.vad_folder is None:
        #     return data, annotations
        # else:
        #     spp = np.load(os.path.join(self.vad_folder, vad_file)).astype(np.float32)
        #     for fn in self.preprocess:
        #         data, spp, annotations = fn(data, spp, annotations)
        #     return data, spp, annotations

        # get labels
        y = self.get_label(file, data)

        return data, y, annotations

    def __len__(self) -> int:
        return self.size

    @staticmethod
    def default_collate_fn(batch: list[tuple[Tensor, Tensor, dict]]) -> tuple[Tensor, Tensor, list[dict]]:
        B = len(batch)
        min_len = min(audio.shape[0] for audio, _, _ in batch)
        n_frames = min_len // 512
        min_len = 512 * n_frames

        x, y = torch.empty((B, min_len)), torch.empty((B, n_frames))
        annotations = []

        # crop sequences to have equal length
        for b, (audio, spp, ann) in enumerate(batch):
            # audio, spp, annotations <= b
            assert audio.shape[0] // 512 == spp.shape[0]
            x[b] = audio[:min_len]
            y[b] = spp[:n_frames]
            annotations.append(ann)

        return x, y, annotations


class GetLabels(ABC):
    @abstractmethod
    def __call__(self, file: str, audio: torch.Tensor) -> torch.Tensor:
        pass


class LoadSPP(GetLabels):
    def __init__(self, spp_folder: str, paths: DatasetPaths = default_path()):
        self.spp_folder = os.path.join(paths.vad_spp, spp_folder)

    def __call__(self, file: str, audio):
        spp = np.load(os.path.join(self.spp_folder, file + '.npy'))
        return torch.tensor(spp, dtype=audio.dtype)


class LoadVADFromSPP(GetLabels):
    def __init__(self, spp_folder: str, paths: DatasetPaths = default_path(), cfg: GlobalConfig = default_config()):
        self.spp_folder = os.path.join(paths.vad_spp, spp_folder)
        self.cfg = cfg

    def __call__(self, file: str, audio):
        assert len(audio.shape) == 1
        spp = np.load(os.path.join(self.spp_folder, file + '.npy'))
        timestamps = prob_to_vad(spp, audio.shape[-1], sampling_rate=self.cfg.sample_rate, window_size_samples=self.cfg.win_size)
        vad = timestamps2vad(timestamps, audio.shape[0], self.cfg.win_size)
        return vad


class LoadVADFromTimestamps(GetLabels):
    def __init__(self, vad_folder: str, cfg: GlobalConfig = default_config(), paths: DatasetPaths = default_path()):
        self.vad_folder = os.path.join(paths.vad_spp, vad_folder)
        self.win_size = cfg.win_size

    def __call__(self, file: str, audio: Tensor):
        timestamps = np.load(os.path.join(self.vad_folder, file + '.npy'))
        assert len(audio.shape) == 1
        vad = torch.zeros((audio.shape[0] // self.win_size,))
        for i in range(timestamps.shape[0]):
            vad[timestamps[i, 0]:timestamps[i, 1]] = 1.
        return vad


def _speed_test(dataset: Dataset):
    from silero_vad.utils_vad import prob_to_vad
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for audio, spp, ann, idx in tqdm(dataloader):
        _ = prob_to_vad(spp, audio.shape[-1])


def _plot_spectrum(x: Tensor | np.ndarray, vad: Tensor | np.ndarray | None, nfft: int, fs: int):
    if isinstance(x, Tensor):
        x = x.numpy()
    if isinstance(vad, Tensor):
        vad = vad.numpy()
    snr = 10 * np.log10(np.linalg.norm(x[vad == 1]) / np.linalg.norm(x[vad == 0]))
    plt.figure()
    plt.plot(x[:10 * fs])
    if vad is not None:
        plt.plot(0.2 * vad[:10 * fs])
    plt.title("[WARNING]: detected low SNR(=%.2f) in clean signal - %i" % (snr, k))

    plt.figure()
    S, f, t, ax = plt.specgram(x, NFFT=nfft, Fs=fs)
    print(S.shape)

    std, mean = np.std(S, axis=1), np.mean(S, axis=1)
    pks, _ = signal.find_peaks(mean, distance=40)
    # pks2 = signal.find_peaks(np.fft.rfft(x))

    plt.figure()
    plt.subplot(311)
    plt.plot(f, std)
    plt.xlabel("frequency bin")
    plt.ylabel("variance")
    plt.subplot(312)
    plt.plot(f, mean)
    plt.vlines(f[pks], 0, np.max(mean), colors='r')
    plt.xlabel("frequency bin")
    plt.ylabel("mean")
    plt.subplot(313)
    plt.plot(f, std / mean)
    plt.xlabel("frequency bin")
    plt.ylabel("std / mean")


def notch_filter(fs: int, f_stop: int, Q=30, plot=False):
    # bandstop filter at 60Hz
    b, a = signal.iirnotch(f_stop, Q, fs=fs)

    if plot:
        plot_tf(b, a, fs, "Notch filter", round(1.5*f_stop))

    def fn(x):
        return signal.filtfilt(b, a, x)

    return fn


def plot_tf(b, a, fs: int, title: str, f_max: int):
    # Frequency response
    freq, h = signal.freqz(b, a, fs=fs, worN=np.linspace(0, f_max, 1024))
    # Plot
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax[0].plot(freq, 20 * np.log10(abs(h)), color='blue')
    ax[0].set_title(title)
    ax[0].set_ylabel("Amplitude (dB)", color='blue')
    ax[0].set_xlim([0, f_max])
    ax[0].set_ylim([-25, 10])
    ax[0].grid(True)
    ax[1].plot(freq, np.unwrap(np.angle(h)) * 180 / np.pi, color='green')
    ax[1].set_ylabel("Angle (degrees)", color='green')
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_xlim([0, f_max])
    ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
    ax[1].set_ylim([-90, 90])
    ax[1].grid(True)


def hpf(fs: int, cutoff: int = 20, plot=False):
    b, a = signal.butter(2, cutoff, btype='high', fs=fs)

    if plot:
        plot_tf(b, a, fs, "High-Pass Filter", cutoff*5)

    def fn(x):
        return signal.filtfilt(b, a, x)

    return fn


def lpf(fs: int, cutoff: int = 4_000, plot=False):
    b, a = signal.butter(2, cutoff, btype='low', fs=fs)

    if plot:
        plot_tf(b, a, fs, "Low-Pass Filter", cutoff*5)

    def fn(x):
        return signal.filtfilt(b, a, x)

    return fn


def preprocess(fs: int, N_harm=4):
    notch_filters = [notch_filter(fs, 60 * (i + 1), Q=30) for i in range(N_harm)]
    lowpass = hpf(fs, 20)

    def fn(x):
        # remove 60Hz pollution from the mains (US)
        for i in range(N_harm):
            x = notch_filters[i](x)
        # remove low frequencies -> center around the origin
        x = lowpass(x)
        return x

    return fn


def compute_snr(dataset):
    snr = np.empty(len(dataset))
    for i, d in enumerate(tqdm(dataset)):
        x, spp = d[0], d[1]
        x = x.numpy()
        vad = timestamps_to_full_vad(prob_to_vad(spp, x.shape[-1]), x.shape[-1])
        snr[i] = 10 * np.log10(np.linalg.norm(x[vad == 1]) / np.linalg.norm(x[vad == 0]))

    return snr


def _plot_audio_size(dataset: LibriSpeech):
    sizes = np.empty(len(dataset), dtype=np.int64)
    for i, (audio, _, _) in tqdm(enumerate(dataset), desc="iterating over dataset"):
        assert len(audio.shape) == 1
        sizes[i] = audio.shape[0]

    min, mean, max = np.min(sizes), np.mean(sizes), np.max(sizes)
    print(f"{min=}, {mean=}")
    print(f"min: {min / dataset.fs}, mean: {mean / dataset.fs}, max: {max / dataset.fs}")

    plt.figure()
    plt.hist(sizes)
    plt.show()


def _class_unbalance():
    ls_data = LibriSpeech(LoadVADFromTimestamps("silero_vad_512_timestamp"))

    H1, tot = 0,  0
    for audio, vad, annotations in tqdm(ls_data, desc="calculating dataset unbalance"):
        H1 += torch.sum(vad == 1)
        tot += vad.shape[-1]
    print("fraction H1 = %.2f" % (100 * H1 / tot) + "%")


def _save_silero_vad_timestamps():
    cfg = default_config()
    ls_data = LibriSpeech(labels=LoadSPP("silero_vad_512_preproc"))

    out_folder = os.path.join(ls_data.paths.vad_spp, "silero_vad_512_timestamp")
    assert not os.path.exists(out_folder)
    os.mkdir(out_folder)

    for audio, spp, annotations in tqdm(ls_data, desc="computing VAD"):
        timestamps = prob_to_vad(spp, audio.shape[-1], sampling_rate=cfg.sample_rate, window_size_samples=cfg.win_size)
        arr = np.empty((len(timestamps), 2), dtype=np.uint32)
        for t, timestamp in enumerate(timestamps):
            arr[t, 0] = timestamp['start']
            arr[t, 1] = timestamp['end']

        file = os.path.join(out_folder, annotations["filename"])
        np.save(file, arr)


if __name__ == '__main__':
    # paths, cfg = default_path(), default_config()
    # timestamps = os.path.join(paths.vad_spp, "silero_vad_512_timestamps")
    # labels = LoadVADFromTimestamps(timestamps, cfg)
    # ls_data = LibriSpeech(labels=labels, config=default_config())
    # ls_data = ConvertLSDataset(ls_folder="../data/train-clean-100/LibriSpeech/train-clean-100",
    #                            new_folder=None, fs=16_000)
    # train_data = DataLoader(ls_data, batch_size=1, shuffle=True, num_workers=2)

    # _speed_test(ls_data)
    _class_unbalance()
    # _plot_audio_size(ls_data)
    # _save_silero_vad_timestamps()

    # k = 38
    # x, _, _ = ls_data[k]
    # # vad = timestamps_to_full_vad(prob_to_vad(spp, x.shape[-1]), x.shape[-1])
    # vad = None
    # _plot_spectrum(x, vad, nfft=2**14, fs=ls_data.fs)
    # # x = lpf(ls_data.fs)(x)
    # # x = preprocess(ls_data.fs)(x)
    # # _plot_spectrum(x, vad, nfft=2 ** 14, fs=ls_data.fs)
    #
    # # snr = compute_snr(ls_data)
    # # np.save("SNR_preproc_silero_old", snr)
    #
    # # snr = np.load("SNR_raw_sileroVAD.npy")
    # #
    # # plt.figure()
    # # plt.hist(snr, bins=30)
    # # plt.title("SNR distribution for the LibriSpeech dataset")
    #
    # plt.show()

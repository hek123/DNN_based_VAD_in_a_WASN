import math
import random
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from torch import Tensor
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm
import torchaudio

from config import GlobalConfig
import utils.utils as utils
from utils.utils import snr_from_vad, is_binary, Format
import data.dataset as data


class Inspection(ABC):
    @abstractmethod
    def __call__(self, audio: Tensor, spp: Tensor, annotation: dict) -> None:
        pass

    @abstractmethod
    def finish(self) -> None:
        pass


class FileSizes(Inspection):
    def __init__(self, dataset_size: int):
        self.sizes = np.empty(dataset_size, dtype=np.int64)

    def __call__(self, audio: Tensor, spp: Tensor, annotation: dict):
        assert len(audio.shape) == 1
        self.sizes[annotation['idx']] = audio.shape[0]

    def finish(self):
        print(f"--- Distribution of file sizes (number of samples) in dataset ---")
        min, mean, max = np.min(self.sizes), np.mean(self.sizes), np.max(self.sizes)
        print(f" - #samples: {min=}, {mean=}")
        print(f" - seconds: min: {min / GlobalConfig.sample_rate}, mean: {mean / GlobalConfig.sample_rate}, max: {max / GlobalConfig.sample_rate}")
        print(f" - total duration of dataset: {np.sum(self.sizes) / GlobalConfig.sample_rate / 3600: .2f} hours")

        plt.figure()
        plt.hist(self.sizes / GlobalConfig.sample_rate)
        plt.title("Duration of audio files (seconds)")
        plt.show()


class ClassBalance(Inspection):
    def __init__(self, dataset_size: int):
        self.H1, self.tot = 0,  0
        self.all1, self.all0 = [], []
        self.size = dataset_size

    def __call__(self, audio, spp, ann):
        # assert is_binary(vad)
        vad = spp > .5
        if torch.all(vad == 1):
            self.all1.append(ann['idx'])
        elif torch.all(vad == 0):
            self.all0.append(ann['idx'])
        else:
            self.H1 += torch.sum(vad > .5)
            self.tot += vad.shape[-1]

    def finish(self):
        print(f"--- Class Unbalance ---")
        print(" - Fraction H1 = %.2f" % (100 * self.H1 / self.tot) + "%")
        print(f" - Full Silence({len(self.all0) / self.size * 100: .2f}%): {self.all0}")
        print(f" - Full Speech({len(self.all1) / self.size * 100: .2f}%): {self.all1}")


class SNR(Inspection):
    def __init__(self, dataset_size: int, min_snr: float = -math.inf, plot_sample: bool = False):
        self.snr = np.empty(dataset_size)
        self.min_snr = min_snr
        self.low_snr = dict()
        self.size = dataset_size
        self.plot = plot_sample

    def __call__(self, audio, spp, ann):
        vad: Tensor = spp > .5
        num1 = torch.count_nonzero(vad)
        num0 = vad.shape[0] - num1
        if num0 > 2 and num1 > 2:
            snr = snr_from_vad(audio, vad)
            self.snr[ann['idx']] = snr
            if snr < self.min_snr:
                self.low_snr[ann['filename']] = snr

                if self.plot:
                    T = 20
                    spp, (offset, num_frames) = utils.random_window(spp, round(T * GlobalConfig.frame_rate))
                    audio = utils.window(audio, offset=GlobalConfig.win_size * offset, num_samples=GlobalConfig.win_size * num_frames)
                    ax = utils.plot_vad(audio, vad)
                    ax.set_title(f"SNR = {snr}")
                    plt.show()

    def finish(self):
        print(f"--- SNR ---")
        print(f" - Num NaN: {np.count_nonzero(np.isnan(self.snr))}, {Format.percent(np.count_nonzero(np.isnan(self.snr)) / self.size)}")
        min, mean, max = np.nanmin(self.snr), np.nanmean(self.snr), np.nanmax(self.snr)
        print(f" - {min=: .2f}, {mean=: .2f}, {max=: .2f}")
        if len(self.low_snr):
            print(f" - Low SNR(SNR<{self.min_snr}), #{len(self.low_snr)}, {Format.percent(len(self.low_snr) / self.size)}: "
                  f"{dict([(f, Format.dB(v)) for f, v in self.low_snr.items()])}")

        plt.figure()
        plt.hist(self.snr)
        plt.title("SNR of the data")
        plt.show()


class Switches(Inspection):
    def __init__(self, dataset_size: int):
        self.switches = np.empty(dataset_size, dtype=np.int_)
        self.samples = np.empty(dataset_size, dtype=np.int_)
        self.var = np.empty(dataset_size)

    def __call__(self, audio, vad, ann):
        assert is_binary(vad)
        k = ann['idx']
        self.switches[k] = int(torch.sum(torch.abs(torch.diff(vad))))
        self.samples[k] = vad.shape[0]
        self.var[k] = float(torch.var(vad))

    def finish(self) -> None:
        print(f"--- VAD switches ---")
        few_switch = np.argwhere((self.switches / self.samples * GlobalConfig.frame_rate) < (1 / 10))
        print(list(few_switch.flat))

        print(f" - Number of samples with VAR < 0.09 (unbalance > 90%): {np.sum(self.var < 0.09)}")
        print(f" - VAD switches:\n"
              f" -- 0 switches: {np.sum(self.switches == 0)}\n"
              f" -- Less than 1 in 10 seconds: {few_switch.shape[0]}\n"
              f" -- Dataset size: {len(self.switches)}")

        plt.figure()
        plt.subplot(211)
        plt.hist(self.switches / self.samples * GlobalConfig.frame_rate)
        plt.title("Switching frequency of the VAD")
        plt.subplot(212)
        plt.hist(self.var)
        plt.title("Variance of the VAD")
        plt.show()


class DataStatistics(Inspection):
    def __init__(self, dataset_size: int, out_file: Path = None):
        self.mean, self.var = 0, 0
        self.N = dataset_size

        from models.encoders import STFT
        self.stft = STFT()
        self.spectrum = torch.zeros(self.stft.num_channels)

        self._file = out_file

    def __call__(self, x, y, ann):
        self.mean += torch.mean(x)
        self.var += torch.var(x)

        S = self.stft(x)
        # print(S.shape)
        self.spectrum += torch.mean(torch.square(S.abs()), dim=1)

    def finish(self) -> None:
        self.mean /= self.N
        self.var /= self.N
        self.spectrum /= self.N

        print(f"--- DataSet statistics ---")
        print(f" - Mean: {self.mean}, Std: {math.sqrt(self.var)}")
        print(f" - Spectrum: {self.spectrum}")

        if self._file is not None:
            torch.save({'mean': self.mean, 'std': math.sqrt(self.var), 'spectrum': self.spectrum}, self._file)

        plt.figure()
        plt.plot(torch.abs(self.spectrum))
        plt.show()


def conformance():
    print("--- Conformance ---")
    dataset = LibriSpeech()
    silero = LoadLabels("silero_vad_512_timestamp", out='vad')
    hmm = LoadLabels("VarHMM_512", out='vad')

    conf = np.empty(len(dataset))
    for k, (audio, _, annotations) in enumerate(tqdm(dataset, desc=" - Computing")):
        vad_silero = silero.load(annotations['filename'], annotations['offset'], audio.shape[0] // GlobalConfig.win_size)
        vad_hmm = hmm.load(annotations['filename'], annotations['offset'], audio.shape[0] // GlobalConfig.win_size)
        conf[k] = float(torch.sum(vad_silero == vad_hmm) / vad_silero.shape[0])

    min, median, max = np.min(conf), np.median(conf), np.max(conf)
    print(f" - min = {min}, median = {median}, max = {max}")

    # This is worse than chance (where both VADs are completely random, with unbalance of 85%)
    disagree = np.argwhere(conf < .75)
    print(f" - disagree = {list(disagree.flat)}")
    print(f" -- with margin: {list(np.argwhere(conf < .75+.02).flat)}")

    plt.figure()
    plt.hist(conf)
    plt.figure()
    plt.boxplot(conf)
    plt.show()


def bad_files_some_numbers():
    excludes = data.LibriSpeechConcat.excludes
    silero_vad, silero_snr10, hmm, disagree = excludes['silero_vad'], excludes['silero_snr10'], excludes['hmm'], excludes['disagree']
    silero_both = silero_vad | silero_snr10
    print(f"silero: {len(silero_vad)}, hmm: {len(hmm)}\n"
          f"disagree: {len(disagree)}\n")
    print(f"Union[silero, hmm]: {len(silero_vad | hmm)};\t"
          f"Union[silero+lowSNR, hmm]: {len(silero_both | hmm)}\n"
          f"Intersect[silero+lowSNR, hmm]: {len(hmm.intersection(silero_both))}\n")
    print(f"Intersect[silero, dis]: {len(silero_vad.intersection(disagree))};\t"
          f"Intersect[silero+lowSNR, dis]: {len(disagree.intersection(silero_both))}\n"
          f"Intersect[hmm, dis]: {len(hmm.intersection(disagree))}\n"
          f"Intersect[hmm+silero, dis]: {len(disagree.intersection(hmm | silero_both))}\n")
    print(f"Intersect[silero+lowSNR, hmm, dist]: {len(hmm.intersection(silero_both, disagree))}\n"
          f"Union[Intersect[silero, hmm], dist]: {len(hmm.intersection(silero_both | disagree))}")


def run_inspection(dataset: Dataset, *inspections: Inspection, preprocess: list = None):
    if preprocess is None:
        preprocess = []

    for audio, spp, ann in tqdm(dataset, desc="Iterating over dataset"):
        for f in preprocess:
            audio = f(audio)
        for f in inspections:
            f(audio, spp, ann)

    for f in inspections:
        f.finish()


def vctk_mic12_duration():
    finfo = data.VCTK(
        labels="", label_type='none',
        exclude=('p280', 'p315')
    )
    for subject, utterances in tqdm(finfo.utterances.items()):
        for utterance in utterances['mic1']:
            _, mic1_path, _ = finfo.get_paths(subject, utterance, 'mic1')
            _, mic2_path, _ = finfo.get_paths(subject, utterance, 'mic2')
            mic1_info: torchaudio.AudioMetaData = torchaudio.info(str(mic1_path))
            mic2_info: torchaudio.AudioMetaData = torchaudio.info(str(mic2_path))
            assert mic1_info.num_frames == mic2_info.num_frames, (mic1_info.num_frames, mic2_info.num_frames)

    print("All audio files have exactly the same length!")


def compare_mic12():
    finfo = data.VCTK(
        labels="", label_type='none',
        exclude=('p280', 'p315')
    )

    resample = torchaudio.transforms.Resample(finfo.fs, GlobalConfig.sample_rate)
    from data.preprocess import hpf
    filt = hpf(GlobalConfig.sample_rate, 50)

    from vad_labeling.silero import SileroVAD
    vad = SileroVAD()

    from vad_labeling.labeling import spp2vad
    spp = spp2vad(speech_pad_ms=30)

    while len(finfo.subjects):
        idx = random.randrange(0, len(finfo.subjects))
        subject = finfo.subjects.pop(idx)

        fig, axes = plt.subplots(2, 3, sharex='col')
        for k in range(3):
            idx = random.randrange(0, len(finfo.utterances[subject]['mic1']))
            utterance = finfo.utterances[subject]['mic1'].pop(idx)

            _, mic1_path, _ = finfo.get_paths(subject, utterance, 'mic1')
            _, mic2_path, _ = finfo.get_paths(subject, utterance, 'mic2')
            mic1_audio, _ = torchaudio.load(str(mic1_path))
            mic2_audio, _ = torchaudio.load(str(mic2_path))

            # print(mic1_audio.shape, mic2_audio.shape)
            mic1_audio = filt(resample(mic1_audio)[0])
            mic2_audio = filt(resample(mic2_audio)[0])
            # print(mic1_audio.shape, mic2_audio.shape)

            vad1 = vad(mic1_audio)
            vad2 = vad(mic2_audio)

            mean = .5*(vad1+vad2)
            harm_mean = torch.sqrt(vad1*vad2)

            spp_mean = spp(mean)

            t1 = np.arange(mic1_audio.shape[0]) / GlobalConfig.sample_rate
            t2 = (0.5 + np.arange(vad1.shape[0])) / GlobalConfig.frame_rate
            ax1, ax2 = axes[0][k], axes[1][k]
            ax1.plot(t1, mic1_audio, label='speech')
            ax1.plot(t2, .5*vad1, label='spp')
            ax1.plot(t2, .5 * mean, label='mean')
            ax1.plot(t2, .5 * harm_mean, label='harmonic mean')
            ax1.set_title("mic1")
            # ax1.legend()
            ax2.plot(t1, mic2_audio)
            ax2.plot(t2, .5*vad2)
            ax2.set_title("mic2")
            ax1.plot(t2, .51 * spp_mean, label='spp_mean')
        plt.show()


if __name__ == '__main__':
    # finfo = data.LibriSpeechConcat(
    #     labels='silero_vad_512_timestamp', label_type='vad'
    # )
    dataset = data.VCTKPreProcessed(random_offset=False, duration=None, mic='all')

    # --- Dataset inspection ---
    from data.preprocess import hpf

    run_inspection(dataset, DataStatistics(len(dataset)))
                   # Path("C:\\Users\\hekto\\PycharmProjects\\MyThesis\\code\\data\\VCTK-preprocessed\\stats.pt")))

    # check_VAD_switches(ls_dataset)
    # conformance()
    # bad_files_some_numbers()

    # --- Sample inspection ---
    # low_snr = {38, 39, 273}  # Due to harmonic pollution, removed in preprocessing
    # non_zero_mean = {35}  # Fixed in ls_concat
    #
    # indices = low_snr
    # for k in indices:
    #     print(f"[{k}]: ", end='')
    #     audio, vad, annotations = dataset[k]
    #     print(f"VAR[vad] = {torch.var(vad)}")
    #     # plot_spectrum(audio, vad, nfft=1024)
    #     plot_vad(audio, vad)

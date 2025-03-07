from pathlib import Path

import numpy as np
import torchaudio
from scipy import signal
import matplotlib.pyplot as plt

import torch
import torchaudio.transforms as tt
import torchaudio.functional as tf

from torch_framework.config import GlobalConfig, Paths
import data.dataset as data


def notch_filter(fs: int, f_stop: int, Q=30, plot=False):
    # bandstop filter at 60Hz
    b, a = signal.iirnotch(f_stop, Q, fs=fs)

    if plot:
        plot_tf(b, a, fs, "Notch filter")

    def fn(x):
        return signal.filtfilt(b, a, x)

    return fn


def plot_tf(b, a, fs: int, title: str, f_range: tuple[int, int] = (10, GlobalConfig.sample_rate // 2)):
    # Frequency response
    freq, h = signal.freqz(b, a, fs=fs, worN=np.linspace(f_range[0], f_range[1], 1024))
    # Plot
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex='all')
    ax[0].semilogx(freq, 20 * np.log10(abs(h)), color='blue')
    ax[0].set_title(title)
    ax[0].set_ylabel("Amplitude (dB)", color='blue')
    ax[0].set_xlim(f_range)
    ax[0].set_ylim([-25, 10])
    ax[0].grid(True)
    ax[1].semilogx(freq, np.unwrap(np.angle(h)) * 180 / np.pi, color='green')
    ax[1].set_ylabel("Angle (degrees)", color='green')
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_xlim(f_range)
    ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
    ax[1].set_ylim([-90, 90])
    ax[1].grid(True)


def hpf(fs: int, cutoff: int = 20, plot=False) -> data.Transform:
    b, a = signal.butter(2, cutoff, btype='high', fs=fs)

    if plot:
        plot_tf(b, a, fs, "High-Pass Filter")

    b, a = torch.tensor(b, dtype=torch.float), torch.tensor(a, dtype=torch.float)

    def fn(x: torch.Tensor):
        return tf.filtfilt(x, a, b, clamp=False)

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
    highpass = hpf(fs, 20)

    def fn(x):
        # remove 60Hz pollution from the mains
        for i in range(N_harm):
            x = notch_filters[i](x)
        # remove low frequencies -> center around the origin
        x = highpass(x)
        return x

    return fn


def process_VCTK(target_folder: str, test: bool = False):
    from vad_labeling.silero import SileroVAD
    from vad_labeling.labeling import spp2timestamps, timestamps_as_array, timestamps2vad
    import utils.utils as utils
    from tqdm import tqdm

    target_path = Path(Paths.data, target_folder)

    if not test:
        target_path.mkdir(exist_ok=False)

    finfo = data.VCTK(labels='', label_type='none')

    audio_loader = data.AudioLoader(sample_rate=finfo.fs, duration=None, random_offset=False, last_frame='pad')
    b, a = signal.butter(2, 50, btype='highpass', fs=GlobalConfig.sample_rate, output='ba')
    a, b = torch.tensor(a, dtype=torch.float), torch.tensor(b, dtype=torch.float)

    dataset_info = {
        'subjects': finfo.subjects,
        'utterances': dict((s, u['mic1']) for s, u in finfo.utterances.items()),
        'sample_rate': GlobalConfig.sample_rate, 'win_size': GlobalConfig.win_size,
        'labels': 'silero',
        'spp2vad': {'threshold': 0.5, 'min_speech_duration_ms': 250, 'max_speech_duration_s': float('inf'),
                    'min_silence_duration_ms': 100, 'speech_pad_ms': 30},
        'preprocess': {'high_pass': {'ord': 2, 'fn': 50}}
    }

    oracle = SileroVAD()
    to_timestamps = spp2timestamps(**dataset_info['spp2vad'])

    for subject in tqdm(finfo.subjects):
        # 1) Load audio
        audio = []
        durations = []
        for utterance in finfo.utterances[subject]['mic1']:
            _, path1, _ = finfo.get_paths(subject, utterance, 'mic1')
            _, path2, _ = finfo.get_paths(subject, utterance, 'mic2')
            audio1, _ = audio_loader(str(path1))
            audio2, _ = audio_loader(str(path2))

            assert audio1.shape == audio2.shape
            assert audio1.shape[0] % GlobalConfig.win_size == 0

            audio.append(torch.stack([audio1, audio2], dim=0))
            durations.append(audio1.shape[0] // GlobalConfig.win_size)

        # 2) Concatenate
        audio = torch.cat(audio, dim=1)
        # print(f"{audio.shape=}")
        durations = torch.tensor(durations, dtype=torch.int32)

        # 3) Preprocess
        audio = tf.filtfilt(audio, a, b)

        # 4) Label
        spp = oracle(audio[0])
        timestamps = to_timestamps(spp)
        timestamps = timestamps_as_array(timestamps)

        if test:
            T = 20
            vad = timestamps2vad('array')(timestamps, spp.shape[0])
            print(f"{vad.shape=}, {spp.shape=}, {audio.shape=}")
            print(f"num_frames={round(GlobalConfig.frame_rate * T)}, num_samples={GlobalConfig.win_size * round(GlobalConfig.frame_rate * T)}")
            spp, (offset, num_samples) = utils.random_window(spp, round(GlobalConfig.frame_rate * T))
            vad = utils.window(vad, offset=offset, num_samples=num_samples)
            print(f"{vad.shape=}, {spp.shape=}")
            audio = utils.window(audio, offset=GlobalConfig.win_size * offset, num_samples=GlobalConfig.win_size * num_samples, dim=1)
            print(f"{audio.shape=}")

            fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col')
            t_start = offset / GlobalConfig.frame_rate
            t_sample = t_start + torch.arange(audio.shape[1]) / GlobalConfig.sample_rate
            t_frame = t_start + (0.5 + torch.arange(spp.shape[0])) / GlobalConfig.frame_rate
            ax1.plot(t_sample, audio[0])
            ax1.plot(t_frame, spp)
            ax1.plot(t_frame, vad)
            ax2.plot(t_sample, audio[1])
            plt.show()
        else:
            # 5) Save
            # create new folder
            subject_folder = Path(target_path, subject)
            subject_folder.mkdir()

            torchaudio.save(
                uri=str(Path(subject_folder, f"{subject}-speech.flac")),
                src=audio, sample_rate=GlobalConfig.sample_rate
            )
            torch.save(durations, Path(subject_folder, f"{subject}-durations.pt"))
            torch.save(timestamps, Path(subject_folder, f"{subject}-vad_silero.pt"))


if __name__ == '__main__':
    process_VCTK("VCTK-preprocessed", False)

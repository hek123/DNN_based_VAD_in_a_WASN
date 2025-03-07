import math
from typing import Literal, Final
from pathlib import Path

import numpy as np
import torch
from neptune import Run
from torch import Tensor
from torch.utils.data import Dataset
import random
import torchaudio

import pyroomacoustics as pra

from data.dataset import MyDataset, Sample
from torch_framework.post_process import ColoredNoise, NoiseGenerator
import Acoustics.signal_processing as sp
from torch_framework.config import GlobalConfig, DataConfig
import utils.utils as utils
from utils.visualization import plot_rirs


class MultiChannelData(Dataset):
    MIN_DISTANCE = .5

    def __init__(self, single_channel_dataset: MyDataset, config: DataConfig, log: Run = None):
        assert len(single_channel_dataset.post_process) == 0
        self._dataset = single_channel_dataset

        self._iid_noise: list[NoiseGenerator] = []
        self._interfering_noise: list[tuple[float, NoiseGenerator]] = []

        self.log = log
        self.label = config.ground_truth
        self.num_mic = config.num_mic
        self.num_devices = config.num_devices

        self.reverb = config.reverb

    def add_iid_noise(self, noise: NoiseGenerator):
        self._iid_noise.append(noise)

    def add_localized_noise(self, noise: NoiseGenerator, p: float = 1.):
        self._interfering_noise.append((p, noise))

    @staticmethod
    def _uniform_open(a: float, b: float):
        # x = random.uniform(a, b)
        # while x == a or x == b:
        #     x = random.uniform(a, b)
        # return x
        return a + random.betavariate(1.5, 1.5) * (b - a)

    @staticmethod
    def get_random_pos(room: pra.Room, margin: float = 0.):
        assert margin == .0, "TODO"

        dims = room.get_bbox()
        for _ in range(3):  # Limit to 3 tries
            p = MultiChannelData._uniform_open(*dims[0]), MultiChannelData._uniform_open(*dims[1])
            if MultiChannelData.check_pos(p, room):
                return p

        print("[WARNING]: Placing source or microphone on illegal position")
        return p

    @classmethod
    def check_pos(cls, pos: tuple[float, float], room: pra.Room) -> bool:
        pos = np.array(pos)
        for i in range(room.n_mics):
            mic_pos = room.mic_array.R[:, i]
            if np.linalg.norm(mic_pos - pos) < cls.MIN_DISTANCE:
                return False
        for i in range(room.n_sources):
            source_pos = room.sources[i].position
            if np.linalg.norm(source_pos - pos) < cls.MIN_DISTANCE:
                return False
        return True

    def place_microphones(self, room: pra.Room):
        d = self.num_devices()
        # print(d)
        nmics = 0
        for _ in range(d):
            LMA = {'M': self.num_mic(),
                   'center': self.get_random_pos(room),
                   'phi': math.tau * random.random(),
                   'd': random.uniform(1e-2, 20e-2)}
                    # 'd': random.uniform(10e-2, 1.)}
            # print(LMA['M'])
            nmics += LMA['M']
            if nmics > 10:
                break
            room.add_microphone_array(pra.bf.linear_2D_array(**LMA))

    def place_source(self, room: pra.Room, signal: np.ndarray):
        p = self.get_random_pos(room)
        room.add_source(p, signal=signal)

    # @staticmethod
    # def get_short_time_energy(x):
    #     assert len(x.shape) == 1
    #     return torch.sqrt(torch.mean(torch.square(x.view(-1, GlobalConfig.win_size)), dim=1))

    def __getitem__(self, idx: int) -> Sample:
        clean_audio, label, ann = self._dataset[idx]
        assert utils.is_binary(label), label
        assert torch.all(torch.isfinite(clean_audio)), clean_audio
        assert len(clean_audio.shape) == len(label.shape) == 1

        # labels = {
        #     "speech": clean_audio.unsqueeze(0),
        #     "vad": label.unsqueeze(0)
        # }

        # 1) Compute the initial SNR
        if "SNR_clean" in ann:
            S, N = ann["SNR_clean"]["S"], ann["SNR_clean"]["N"]
        else:
            S, N = sp.snr(clean_audio, label)
            S += 1e-8
            N += 1e-8

        # labels['energy'] = self.get_short_time_energy(clean_audio).unsqueeze(0)

        # 2) Create a room
        room_dims = (random.uniform(4, 20), random.uniform(4, 20))  # (width, lenght(, height))
        # height = random.uniform(2, 4)
        if self.reverb:
            room = pra.ShoeBox(room_dims, fs=GlobalConfig.sample_rate, use_rand_ism=True)
        else:
            room = pra.AnechoicRoom(room_dims, fs=GlobalConfig.sample_rate)

        # 3) Add microphones
        self.place_microphones(room)

        # 4) Add the target speaker
        self.place_source(room, clean_audio.numpy())
        # speaker = {'d': random.uniform(.4, 5), 'theta': math.tau * random.random()}
        # ann['speaker'] = speaker
        # # Add speaker source; distance from .4m to 5m
        # # delay to be compensated to ensure alignment with labeling
        # delay = round(speaker['d'] / pra.constants.get('c') * GlobalConfig.sample_rate)
        # attenuation = 1 / (speaker['d']**2)
        # room.add_source((speaker['d'] * math.cos(speaker['theta']), speaker['d'] * math.sin(speaker['theta'])),
        #                 signal=clean_audio.numpy())

        # 5) Interfering (localized) noise
        for p, if_noise in self._interfering_noise:
            if random.random() < p:
                # noise = if_noise(clean_audio.shape)
                # IN = {'d': random.uniform(speaker['d'], 10), 'theta': math.tau * random.random()}
                # noise, snr = if_noise.scale(noise, S=S)
                # assert torch.all(torch.isfinite(noise)), (if_noise, noise)
                # # offset = random.randint(0, audio.shape[0] - noise.shape[0])
                # room.add_source((IN['d'] * math.cos(IN['theta']), IN['d'] * math.sin(IN['theta'])),
                #                 signal=noise)
                # # TODO: verify formula
                # IN['SIR'] = snr * IN['d']**2 * attenuation
                # sigmaI = S / IN['SIR']
                # if sigmaI > .5:
                #     print("[WARNING]!")
                # ann['interfering_noise'] = IN
                noise = if_noise(clean_audio.shape)
                noise, snr = if_noise.scale(noise, S)
                # ann[f'SIR/{if_noise.__class__.__name__}'] = snr *
                self.place_source(room, noise.numpy())

        ann['mic_pos'] = room.mic_array.R
        ann['source_pos'] = np.c_[*[s.position for s in room.sources]]
        # print(ann['mic_pos'])
        # print(ann['source_pos'])
        # print()

        delay_matrix = np.empty((room.n_sources, room.n_mics), dtype=np.int_)
        att_matrix = np.empty((room.n_sources, room.n_mics), dtype=np.float_)
        for i in range(room.n_sources):
            ps = room.sources[i].position
            for j in range(room.n_mics):
                pm = room.mic_array.R[:, j]
                distance = np.linalg.norm(ps - pm)
                delay_matrix[i, j] = np.round(distance / pra.constants.get('c') * GlobalConfig.sample_rate)
                att_matrix[i, j] = 1 / (distance ** 2)

        # print(delay_matrix)

        # room.plot()
        # plt.show()

        # 6) Simulate
        room.simulate()
        # print(np.sum(room.rir[0][0]), math.sqrt(attenuation))  # mic, source
        ann['RIRs'] = room.rir
        assert all(all(np.all(np.isfinite(rir)) for rir in rirs) for rirs in room.rir)
        # print(len(room.rir), room.n_mics, room.n_sources)
        # plot_rirs(room.rir)

        # delay = delay_matrix[0, 0]
        # audio = room.mic_array.signals[:, delay:delay+clean_audio.shape[0]]
        audio = room.mic_array.signals
        # assert audio.shape[1] == clean_audio.shape[0], (audio.shape, clean_audio.shape)
        # assert np.all(np.isfinite(audio)), audio
        audio = torch.tensor(audio, dtype=torch.float)
        assert torch.all(torch.isfinite(audio)), audio
        # print(audio.shape)

        # 7) Add iid (non localized) noise
        for iid_noise in self._iid_noise:
            attenuation = att_matrix[0, 0]
            iid_noise, snr = iid_noise.scale(iid_noise(audio.shape), S=attenuation*S, N=attenuation*N)
            audio += iid_noise
            ann['SNR'] = snr

        if self.log is not None:
            self.log['data'].append(ann)

        assert torch.all(torch.isfinite(audio)), audio

        # return torch.clip(audio, -1, 1), labels[self.label], ann
        return audio, label.unsqueeze(0), ann

    def __len__(self):
        return len(self._dataset)


def align_labels(audio: Tensor, y_pred: Tensor, y_true: Tensor, source_pos, mic_pos, calibration: int):
    assert len(audio.shape) == len(y_pred.shape) == len(y_true.shape) == 2, (audio.shape, y_pred.shape, y_true.shape)
    assert y_true.shape[0] == 1
    assert mic_pos.shape[1] == y_pred.shape[0] == audio.shape[0], (audio.shape, mic_pos.shape, y_pred.shape)

    sample_delay, frame_delay = [], []
    for i in range(mic_pos.shape[1]):
        distance = np.linalg.norm(source_pos - mic_pos[:, i])
        frame_delay.append(round(distance / pra.constants.get('c') * GlobalConfig.frame_rate))
        sample_delay.append(round(distance / pra.constants.get('c') * GlobalConfig.sample_rate))

    new_pred = torch.empty(y_pred.shape[0], min(y_true.shape[1], y_pred.shape[1] - max(frame_delay)))
    new_audio = torch.empty(audio.shape[0], GlobalConfig.win_size * new_pred.shape[1])

    for i in range(mic_pos.shape[1]):
        new_audio[i] = audio[i, sample_delay[i]:sample_delay[i] + new_audio.shape[1]]
        new_pred[i] = y_pred[i, frame_delay[i]:frame_delay[i] + new_pred.shape[1]]
    y_true = y_true[:, :new_pred.shape[1]].expand(new_pred.shape)

    # # assert y_true.shape[1] <= y_pred.shape[1] - max_delay, (y_true.shape, y_pred.shape, max_delay)
    # out_size = min(y_true.shape[1], y_pred.shape[1] - max_delay)
    # y_pred = torch.stack([y[:out_size] for y in y_pred_out])

    return new_audio[:, calibration*GlobalConfig.win_size:], new_pred[:, calibration:], y_true[:, calibration:]


def multi_channel_batch(batch: list[Sample]) -> tuple[Tensor, Tensor, dict]:
    assert len(batch[0][0].shape) == len(batch[0][1].shape) == 2
    x = torch.cat([sample[0] for sample in batch])
    y = torch.cat([sample[1] for sample in batch])
    ann = {'sample_ann': [sample[2] for sample in batch],
           'num_channels': [sample[0].shape[0] for sample in batch]}
    return x, y, ann


class SavedDataset(Dataset):
    ROOT: Final = Path("C:\\Users\\hekto\\PycharmProjects\\MyThesis\\code\\data\\simulated_dataset")

    def __init__(self, folder: str, d: int = 4):
        self.path = self.ROOT.joinpath(folder)
        assert self.path.exists(), f"folder: {self.path} does not exist"

        self.format = f"0{d}d"

        names = [f.name for f in self.path.iterdir() if f.suffix == '.wav']
        self.size = max(int(f.split('_')[0]) for f in names) + 1
        print(self.size)

    def __getitem__(self, i):
        fname = f"{i:{self.format}}"
        audio, _ = torchaudio.load(str(self.path.joinpath(f"{fname}_speech.wav")))
        label = torch.load(self.path.joinpath(f"{fname}_VAD.pt"))
        ann = torch.load(self.path.joinpath(f"{fname}_ann.pt"))
        return audio, label, ann

    def __len__(self):
        return self.size


# create and save multi-channel dataset
if __name__ == '__main__':
    from data.dataset import VCTKPreProcessed, MS_SNSD_Noise
    from torch_framework import config
    from torch_framework.post_process import ColoredNoise
    from tqdm import tqdm

    # SavedDataset("mc-sd")

    new_folder = "mc-sd"
    new_folder = SavedDataset.ROOT.joinpath(new_folder)
    assert not new_folder.exists(), f"folder: {new_folder} already exists!"
    new_folder.mkdir()
    vctk_dataset = VCTKPreProcessed(duration=30, add_full_silence=2, random_offset=True, mic='all')

    data_config = config.DataConfig(multi_channel=True, clean=False, ground_truth='vad',
                                    num_devices=lambda: random.randint(1, 1))

    dataset = MultiChannelData(vctk_dataset, config=data_config)
    if not data_config.clean:
        dataset.add_iid_noise(ColoredNoise('white', (10, 60)))
        # default: (-5, 15, 0)
        dataset.add_localized_noise(MS_SNSD_Noise('single source', (0, 30, 10)), p=.95)

    i = 0
    for _ in range(5):
        for audio, label, ann in tqdm(dataset):
            fname = f"{i:04d}"
            torchaudio.save(str(new_folder.joinpath(f"{fname}_speech.wav")), audio, GlobalConfig.sample_rate)
            torch.save(label, new_folder.joinpath(f"{fname}_VAD.pt"))
            torch.save(ann, new_folder.joinpath(f"{fname}_ann.pt"))
            i += 1

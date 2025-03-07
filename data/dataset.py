import math
import os
from time import sleep
import random
from abc import ABC, abstractmethod
from collections.abc import Sequence, Iterable
from dataclasses import dataclass, field, InitVar
from pathlib import Path
from typing import Callable, Union, Optional, Final

import torch
from neptune import Run
from torch import Tensor
import torchaudio
from torch.utils.data import Dataset
import numpy as np

from tqdm import tqdm

import utils.utils as utils
from config import GlobalConfig, Paths
from training_framework.post_process import PostProcess, NoiseGenerator, snr


Sample = tuple[Tensor, Tensor, dict]
Transform = Callable[[Tensor], Tensor]


class MyDataset(Dataset, Sequence, ABC):
    def __init__(self, label_type: str, post_process: list[PostProcess] = None, log: Run = None):
        assert label_type in ('spp', 'vad')
        self.label_type = label_type
        self.post_process = [] if post_process is None else post_process
        self.log = log

    @abstractmethod
    def get_sample(self, idx: int) -> Sample:
        pass

    def __getitem__(self, idx: int) -> Sample:
        audio, label, ann = self.get_sample(idx)
        self._check_sample(audio, label, ann)

        ann['idx'] = idx
        ann['label_type'] = self.label_type

        for p in self.post_process:
            audio, label = p(audio, label, ann, self.log)
            # self._check_sample(audio, label, ann), repr(p)

        self._check_sample(audio, label, ann)
        return audio, label, ann

    def _check_sample(self, audio: Tensor, label, ann):
        assert audio.dtype == label.dtype == torch.float
        assert len(audio.shape) == len(label.shape) == 1
        assert audio.shape[0] % GlobalConfig.win_size == 0 and audio.shape[0] == label.shape[0] * GlobalConfig.win_size
        assert torch.isfinite(audio).all() and \
               (utils.is_binary(label) if self.label_type == 'vad' else utils.is_probability(label))


@dataclass
class LibriSpeechConcat:
    excludes = {
        'silero_vad': {0, 1, 21, 24, 29, 41, 53, 58, 60, 62, 73, 75, 90, 93, 96, 105, 118, 119, 128, 135, 136, 138, 150,
                       158, 196, 205, 208, 215, 217, 231, 235, 236, 238, 239, 240, 247, 272, 273, 284, 285, 286, 287,
                       300, 301, 312, 313, 314, 316, 319, 326, 327, 335, 336, 337, 348, 368, 385, 391, 471, 534, 547,
                       577, 581, 582},
        'silero_snr10': {38, 39, 410},
        'hmm': {20, 21, 38, 39, 50, 51, 52, 58, 63, 73, 74, 84, 85, 90, 91, 100, 110, 111, 148, 167, 169, 170, 179, 217,
                235, 236, 244, 248, 256, 273, 277, 284, 285, 287, 306, 314, 327, 339, 355, 356, 384, 385, 389, 398, 426,
                427, 428, 471, 481, 482, 518, 531, 533, 536},
        'disagree': {20, 21, 90, 167, 217, 235, 236, 285, 306, 312, 313, 314, 327, 385, 389, 518}
    }

    root: Path = Path(Paths.data, "LibriSpeechConcat")
    labels_folder: Path = Paths.labels

    labels: InitVar[str] = "silero_512_spp"
    label_type: str = "spp"

    fs: int = field(default=16_000, init=False)

    train_path: Path = field(init=False)
    label_path: Path = field(init=False)

    subjects: list[str] = field(init=False)
    utterances: dict[str, dict[str, list[str]]] = field(init=False)

    exclude: InitVar[set[int]] = None

    def __post_init__(self, labels, exclude):
        self.files = []
        for s in os.listdir(self.root):
            self.files += [(s, t) for t in os.listdir(Path(self.root, s))]

        if exclude is not None:
            exclude = sorted(exclude, reverse=True)
            for k in exclude:
                del self.files[k]

        self.label_path = Path(self.labels_folder, labels)

    def get_file_paths(self, subject: str, track: str) -> tuple[dict, Path, Path]:
        audio_folder = Path(self.root, subject, track)
        filename = f"{subject}-{track}"
        audio_file = f"speech-{filename}.wav"
        annotation_file = f"annotation-{filename}.npz"

        annotations = dict(np.load(os.path.join(audio_folder, annotation_file)))
        annotations["filename"] = filename

        return annotations, Path(audio_folder, audio_file), Path(self.label_path, f"{filename}.npy")

    def all_file_paths(self):
        for s, t in self.files:
            yield self.get_file_paths(s, t)

    def __len__(self) -> int:
        return len(self.files)


@dataclass
class Stimuli:
    root: Path = Path(Paths.data, "stimuli")
    files: list[tuple[str, Path, Optional[Path]]] = field(init=False)

    fs: int = field(default=48_000, init=False)

    def __post_init__(self):
        self.files = [(file.removesuffix('.wav'), Path(self.root, file), None) for file in os.listdir(self.root)
                      if file.endswith('.wav')]


@dataclass
class VCTK:
    labels: InitVar[str]
    label_type: str

    root: Path = Path(Paths.data, "VCTK-Corpus-0.92")
    labels_folder: Path = Path(Paths.labels, "VCTK")

    train_path: Path = field(init=False)
    label_path: Path = field(init=False)

    fs: int = field(default=48_000, init=False)

    subjects: list[str] = field(init=False)
    utterances: dict[str, dict[str, list[str]]] = field(init=False)

    # Subjects 280 and 315 do not contain audio for mic2 and are hence not considered
    exclude: InitVar[Iterable[str]] = ('p280', 'p315')

    def __post_init__(self, labels, exclude):
        self.train_path = Path(self.root, "wav48_silence_trimmed")
        assert self.train_path.exists(), f"Path not found: {self.train_path}"

        if len(labels):
            self.label_path = Path(self.labels_folder, labels)
            assert self.label_path.exists(), f"Path not found: {self.label_path}"
        else:
            if self.label_type != 'none':
                print(f"No label path provided, setting label_type to 'none'")
                self.label_type = 'none'

        self.subjects = [file for file in os.listdir(self.train_path) if
                         (file.startswith('p') and (file not in exclude))]

        self.utterances = dict([(s, {'mic1': [], 'mic2': []}) for s in self.subjects])
        for subject in self.subjects:
            files = os.listdir(Path(self.train_path, subject))
            for file in files:
                s, t, mic = file.removesuffix('.flac').split('_')
                assert s == subject
                assert mic == 'mic1' or mic == 'mic2', mic
                self.utterances[subject][mic].append(t)

            if len(self.utterances[subject]['mic2']) == 0:
                print(f"[WARNING]: subject {subject} is missing mic2 data")
            else:
                assert self.utterances[subject]['mic1'] == self.utterances[subject]['mic2'], \
                    f"{subject=}\n{self.utterances[subject]['mic1']=}\n{self.utterances[subject]['mic2']=})"
            self.utterances[subject]['mic1'].sort()
            self.utterances[subject]['mic2'].sort()

    def get_paths(self, subject: str, utterance: str, mic: str) -> tuple[dict, Path, Path | None]:
        filename = f"{subject}_{utterance}_{mic}"
        audio = Path(self.train_path, subject, filename + '.flac')
        if self.label_type == 'none':
            label = None
        else:
            label = Path(self.label_path, filename + '.npy')
        return {'filename': filename}, audio, label

    def all_file_paths(self, mic: str = 'mic1'):
        for subject, utterances in self.utterances.items():
            for utterance in utterances[mic]:
                yield self.get_paths(subject, utterance, mic)


class VCTKPreProcessed(MyDataset):
    def __init__(self, duration: float = None, random_offset: bool = True, post_processing: list["PostProcess"] = None,
                 mic: str = 'mic1', add_full_silence: int = 0, add_short_silences: bool = True,
                 log: Run = None, root: Path = None):
        assert mic in ('mic1', 'mic2', 'all'), mic

        MyDataset.__init__(self, label_type='vad', post_process=post_processing, log=log)

        self.root: Path = Path(Paths.data, "VCTK-preprocessed") if root is None else root
        assert self.root.exists(), f"Path not found: {self.root}"

        self.subjects = [file for file in os.listdir(self.root) if file.startswith('p')]
        self.subjects += [None] * add_full_silence

        self.audio_loader = AudioLoader(sample_rate=16_000, duration=duration, random_offset=random_offset,
                                        last_frame='none')
        self.label_loader = load_vad(pad_front=-1, pad_end=-1)
        self._mic = mic

        self._silence = add_short_silences

        self.meta = torch.load(Path(self.root, "stats.pt"))

        print(f"--- Preprocessed VCTK-0.92 dataset ---\n"
              f" - mic: {mic}\n"
              f" - duration: {'full' if duration is None else f'{duration}s'}\n"
              f" - dataset size: {len(self.subjects) - add_full_silence} subjects, {add_full_silence} silence tracks")

    def get_sample(self, idx: int) -> Sample:
        subject = self.subjects[idx // 2] if self._mic == 'all' else self.subjects[idx]
        if subject is None:
            return (torch.zeros(self.audio_loader.max_frames * GlobalConfig.win_size),
                    torch.zeros(self.audio_loader.max_frames),
                    {'filename': 'silence', 'durations': torch.tensor([]),
                     'SNR_clean': {'S': self.meta['std']**2, 'N': 0.}})

        audio_path = Path(self.root, subject, f"{subject}-speech.flac")
        label_path = Path(self.root, subject, f"{subject}-vad_silero.pt")
        ann_path = Path(self.root, subject, f"{subject}-durations.pt")

        channel = {'mic1': 0, 'mic2': 1, 'all': idx % 2}[self._mic]
        audio, (offset, duration) = self.audio_loader(str(audio_path), channel=channel)
        label = self.label_loader(label_path, offset, duration)
        utterance_durations = torch.load(ann_path)

        if self._silence:
            audio, label = self.add_silences(audio, label, offset, duration, utterance_durations)

        ann = {'filename': subject, 'durations': utterance_durations}

        return audio, label, ann

    def add_silences(self, audio: Tensor, label: Tensor, offset: int, duration: int, durations: Tensor):
        start_idx = torch.cumsum(durations, 0)
        start_idx = start_idx[torch.logical_and(offset <= start_idx, start_idx < offset + duration)]
        start_idx = start_idx - offset
        # print(duration, start_idx)

        start_idx = [idx for idx in start_idx if random.random() < .3]
        silence = [round(GlobalConfig.frame_rate * random.triangular(.2, 3., 1.)) for _ in start_idx]
        # print(silence)

        audio = list(torch.tensor_split(audio, [idx * GlobalConfig.win_size for idx in start_idx]))
        label = list(torch.tensor_split(label, start_idx))
        # print(len(label), len(silence), len(start_idx))
        for i in range(len(start_idx)):
            i = len(start_idx) - i - 1
            # print('i =', i)
            audio.insert(i + 1, torch.zeros((GlobalConfig.win_size * silence[i])))
            label.insert(i + 1, torch.zeros((silence[i])))
        audio, label = torch.cat(audio), torch.cat(label)

        # fig, (ax1, ax2) = plt.subplots(2, 1)
        # ax1.plot(audio)
        # ax2.plot(label)
        # plt.show()

        return audio, label

    def __len__(self):
        return 2*len(self.subjects) if self._mic == 'all' else len(self.subjects)

    def normalize(self) -> PostProcess:
        def fn(audio, label, ann, log):
            return audio / self.meta['std'], label

        return fn


class MyAudioDataset(MyDataset):
    def __init__(self, dataset: Sequence[tuple[dict, Path, Path]], fs: int, label_type: str, duration: float = None,
                 random_offset: bool = True, post_processing: list[PostProcess] = None, last_frame: str = 'drop',
                 pad_front: int = 0):
        MyDataset.__init__(self, label_type, post_processing)
        assert duration is None or duration > 0

        self.paths = dataset

        self.audio_loader = AudioLoader(sample_rate=fs, duration=duration, random_offset=random_offset,
                                        last_frame=last_frame)
        self.label_loader = get_label_loader(label_type, pad_front)

    def get_sample(self, idx):
        assert 0 <= idx <= len(self), "Index out of range"
        if idx == len(self):
            raise StopIteration()

        annotations, audio_path, label_path = self.paths[idx]

        data, (offset, frames) = self.audio_loader(str(audio_path))

        # Get labels
        y = None if self.label_loader is None else self.label_loader(label_path, offset, frames)

        # Annotations
        annotations['offset'] = offset

        return data, y, annotations

    def __len__(self) -> int:
        return len(self.paths)


class VCTKDataset(Dataset):
    def __init__(self, vctk: VCTK, duration: float = None, post_processing: list["PostProcess"] = None):
        assert duration is None or duration > 0

        self.vctk = vctk

        self.audio_loader = AudioLoader(sample_rate=vctk.fs, duration=None, random_offset=False, last_frame='pad')
        self.label_loader = get_label_loader(vctk.label_type)
        self.post_process: list[PostProcess] = [] if post_processing is None else post_processing

        self.num_frames = None if duration is None else round(duration * GlobalConfig.frame_rate)

        self.log = None

    def load_sample(self, subject: str, utterance: str, mic: str = 'mic1'):
        annotations, audio_path, label_path = self.vctk.get_paths(subject, utterance, mic)

        # Load audio
        x, (offset, frames) = self.audio_loader(str(audio_path))

        # Get labels
        y = self.label_loader(label_path, offset, frames)

        return x, y, annotations

    def __getitem__(self, idx):
        assert 0 <= idx <= len(self), "Index out of range"
        if idx == len(self):
            raise StopIteration()

        subject = self.vctk.subjects[idx]
        x, y = self.load_subject_data(subject, p_silence=.5)

        annotations = {'subject': subject}

        # post processing
        for p in self.post_process:
            x, y, annotations = p(x, y, annotations, self.log)

        return x, y, annotations

    def __len__(self) -> int:
        return len(self.vctk.subjects)

    def load_subject_data(self, subject: str, shuffle: bool = True, p_silence=.5) -> tuple[Tensor, Tensor]:
        """
        Create one long continuous audio signal from the given utterances with added silences

        :param subject: the subject from which to load speech
        :param shuffle: Shuffle the utterances to a random order, default: True
        :param p_silence: probability to add silence between consecutive utterances
        :return: speech, labels
        """
        assert 0. <= p_silence < 1., f"{p_silence=}"

        utterances = self.vctk.utterances[subject]['mic1'].copy()
        if shuffle:
            random.shuffle(utterances)

        # target_num_frames = round(duration * Config.frame_rate)
        num_frames = 0
        x, y = [], []
        for utterance in utterances:
            # load speech and label
            audio, label, _ = self.load_sample(subject, utterance)
            x.append(audio)
            y.append(label)
            num_frames += label.shape[0]

            if self.num_frames is not None and num_frames > self.num_frames:
                break

            # add silence
            if random.random() < p_silence:
                silence_duration = random.lognormvariate(math.log(1.), .5)
                # print(f"Add {silence_duration: .2f} seconds of silence")
                silence_duration = round(silence_duration * GlobalConfig.frame_rate)
                x.append(torch.zeros((GlobalConfig.win_size * silence_duration,)))
                y.append(torch.zeros((silence_duration,)))
                num_frames += silence_duration

            if self.num_frames is not None and num_frames > self.num_frames:
                break

        x = torch.cat(x)
        y = torch.cat(y)
        assert x.shape[0] == GlobalConfig.win_size * y.shape[0], f"x: {x.shape[0]}, y: {y.shape[0]}"

        if self.num_frames is not None:
            if y.shape[0] < self.num_frames:
                print("[WARNING]: target duration not reached")

            y, (offset, num_frames) = utils.random_window(y, self.num_frames)
            assert num_frames == self.num_frames
            x = utils.window(x, num_samples=GlobalConfig.win_size * num_frames, offset=GlobalConfig.win_size * offset)

        return x, y


def load_spp(path: str | Path, offset: int, duration: int):
    return torch.tensor(np.load(path)[offset:offset + duration])


def load_vad(pad_front: int = 0, pad_end: int = 0):
    def fn(path: str | Path, frame_offset: int, frame_duration: int):
        # timestamps = np.load(path).astype(np.int_) - frame_offset
        timestamps = torch.load(path) - frame_offset
        timestamps[:, 0] -= pad_front
        timestamps[:, 1] += pad_end
        vad = torch.zeros((frame_duration,))
        for timestamp in timestamps:
            if timestamp[0] > frame_duration:
                break
            if timestamp[1] > frame_duration:
                vad[timestamp[0]:] = 1.
                break
            if timestamp[1] < 0:
                continue
            if timestamp[0] < 0:
                vad[:timestamp[1]] = 1.
                continue
            vad[timestamp[0]:timestamp[1]] = 1.
        return vad
    return fn


def get_label_loader(label_type: str, pad_front: int = 0) -> Union[Callable[[str | Path, int, int], Tensor], None]:
    if label_type == 'vad':
        return load_vad(pad_front=pad_front)
    elif label_type == 'spp':
        return load_spp
    elif label_type == 'none':
        return None
    else:
        raise AssertionError(label_type)


def default_collate_fn(batch: list[Sample]) -> tuple[Tensor, Tensor, list[dict]]:
    x, y, _ = batch[0]
    if len(x.shape) == 2:
        assert x.shape[0] == y.shape[0] == 1, (x.shape, y.shape)
        make_batched_tensor = torch.cat
    elif len(x.shape) == 1:
        assert len(y.shape) == 1, (x.shape, y.shape)
        make_batched_tensor = torch.stack
    else:
        raise AssertionError((x.shape, y.shape))

    x = make_batched_tensor([sample[0] for sample in batch])
    y = make_batched_tensor([sample[1] for sample in batch])
    ann = [sample[2] for sample in batch]

    return x, y, ann


def unbatched(batch: list[Sample]) -> Sample:
    assert len(batch) == 1
    return batch[0]


class AudioLoader:
    """
    Helper class for loading audio
    Resamples the audio to the frequency defined by Config.sample_rate
    """
    def __init__(self, sample_rate: int, duration: float = None, random_offset: bool = True, last_frame: str = 'drop'):
        """
        :param sample_rate: The original sample rate of the audio to be loaded, the sample rate of the returned audio is
            always Config.sample_rate
        :param duration: If specified, limits the duration of the loaded audio (in seconds)
        :param random_offset: If True and duration is specified, the loader will take a random window from the audio
         file. If false the loader will always load starting from the beginning of the file.
        :param last_frame: Specify this parameter to align the audio with the VAD/SPP labeling. The audio is typically
            labeled in windows defined by Config.win_size. Can be one of:
            - 'drop': drops the last audio frame
            - 'pad': the last frame is padded with zeros
            - 'none': load the audio as is, the audio is not a multiple of Config.win_size
        """
        assert sample_rate % GlobalConfig.sample_rate == 0
        assert duration is None or duration > 0
        assert last_frame in {'drop', 'pad', 'none'}
        if duration is None:
            if random_offset:
                print("[WARNING]: setting <random_offset=True> has no effect when <duration=None>")
            random_offset = False

        self.fs: Final = sample_rate
        self.scale: Final = (sample_rate // GlobalConfig.sample_rate) * GlobalConfig.win_size
        self.max_frames: Final = None if duration is None else round(duration * GlobalConfig.frame_rate)
        self.random_offset: Final = random_offset
        self.last_frame: Final = last_frame

        self._num_frames: Callable[[int, int], int] = int.__floordiv__ if self.last_frame == 'drop' else utils.ceildiv

        self._resample = torchaudio.transforms.Resample(sample_rate, GlobalConfig.sample_rate)

    def __call__(self, path: str, channel: int = 0) -> tuple[Tensor, tuple[int, int]]:
        assert os.path.exists(path), path
        info: torchaudio.AudioMetaData = torchaudio.info(path)
        assert info.sample_rate == self.fs, f"Invalid sample frequency: {info.sample_rate}Hz, expected {self.fs}Hz"

        max_frames = self._num_frames(info.num_frames, self.scale)
        num_frames = max_frames if self.max_frames is None else min(self.max_frames, max_frames)
        frame_offset = random.randint(0, max_frames - num_frames) if self.random_offset else 0

        fails = []
        for _ in range(3):
            try:
                data, fs = torchaudio.load(path, frame_offset=frame_offset*self.scale, num_frames=num_frames*self.scale,
                                           normalize=True, channels_first=True)
                continue
            except Exception as e:
                print(f"Failed to load audio file: {path}\n"
                      f"\tTrying again")
                fails.append(e)
                sleep(.01)
        if len(fails):
            raise RuntimeError(*fails)

        data = data[channel, :]
        data = self._resample(data)
        if self.last_frame == 'pad':
            data = torch.cat([data, torch.zeros(GlobalConfig.win_size * num_frames - data.shape[1])])

        # if self.last_frame == 'pad' or self.last_frame == 'drop':
        #     assert data.shape[0] % Config.win_size == 0

        # assert data.dtype == torch.float
        return data, (frame_offset, num_frames)

    def num_frames(self, path: str | Path):
        assert os.path.exists(path), path
        return self._num_frames(torchaudio.info(path).num_frames, self.scale)


def load_audio(path: str, sample_rate: int, channel: int = 0, duration: float = None, random_offset: bool = True,
               last_frame: str = 'drop') -> tuple[Tensor, tuple[int, int]]:
    """
    :param sample_rate: The original sample rate of the audio to be loaded, the sample rate of the returned audio is
        always Config.sample_rate
    :param duration: If specified, limits the duration of the loaded audio (in seconds)
    :param random_offset: If True and duration is specified, the loader will take a random window from the audio
     file. If false the loader will always load starting from the beginning of the file.
    :param last_frame: Specify this parameter to align the audio with the VAD/SPP labeling. The audio is typically
        labeled in windows defined by Config.win_size. Can be one of:
        - 'drop': drops the last audio frame
        - 'pad': the last frame is padded with zeros
        - 'none': load the audio as is, the audio is not a multiple of Config.win_size
    """
    assert os.path.exists(path), path
    assert sample_rate % GlobalConfig.sample_rate == 0
    assert duration is None or duration > 0
    assert last_frame in {'drop', 'pad', 'none'}
    if duration is None:
        if random_offset:
            print("[WARNING]: setting <random_offset=True> has no effect when <duration=None>")
        random_offset = False

    scale = (sample_rate // GlobalConfig.sample_rate) * GlobalConfig.win_size

    info: torchaudio.AudioMetaData = torchaudio.info(path)
    assert info.sample_rate == sample_rate, \
        f"Invalid sample frequency: {info.sample_rate}Hz, expected {sample_rate}Hz"

    max_frames = info.num_frames // scale if last_frame == 'drop' else utils.ceildiv(info.num_frames, scale)
    num_frames = max_frames if duration is None else min(max_frames, round(duration * GlobalConfig.frame_rate))
    frame_offset = random.randint(0, max_frames - num_frames) if random_offset else 0

    data, fs = torchaudio.load(path, frame_offset=frame_offset*scale, num_frames=num_frames*scale,
                               normalize=True, channels_first=True)
    data = data[channel, :]
    data = torchaudio.functional.resample(data, fs, GlobalConfig.sample_rate)
    if last_frame == 'pad':
        data = torch.cat([data, torch.zeros(GlobalConfig.win_size * num_frames - data.shape[0])])

    if last_frame == 'pad' or last_frame == 'drop':
        assert data.shape[0] % GlobalConfig.win_size == 0

    # assert data.dtype == torch.float
    return data, (frame_offset, num_frames)


class NoiseFromDataset(NoiseGenerator):
    def __init__(self, dataset: Dataset, snr: Sequence[float] | Callable[[], float]):
        NoiseGenerator.__init__(self, snr)
        self.data = dataset

    def __call__(self, size: Sequence[int]) -> Tensor:
        idx = random.randrange(len(self.data))
        audio, vad, _ = self.data[idx]
        S, _ = snr(audio, vad, allow_nan=True)
        if not math.isnan(S):
            audio /= math.sqrt(S + 1e-8)
        # else:  # Basically noise (on clean data) only -> no signal ...
        #     print("[WARING]: no statistics available")
        assert audio.shape == size, (audio, size)

        return audio


class MS_SNSD_Noise(NoiseGenerator):
    def __init__(self, noise_type: str | list[str], snr: Sequence[float] | Callable[[], float], root: Path = None):
        NoiseGenerator.__init__(self, snr)

        self.root: Path = Path(Paths.data, "noise", "MS-SNSD") if root is None else root
        assert self.root.exists()
        self.train_path = Path(self.root, "noise_train")
        self.test_path = Path(self.root, "noise_test")

        self.noises: dict[str, int] = dict()

        for file in self.train_path.iterdir():
            if file.suffix == '.wav':
                name = file.name.split('_')[0]
                if name in self.noises:
                    self.noises[name] += 1
                else:
                    self.noises[name] = 1
        # print(self.noises)

        if isinstance(noise_type, list):
            assert all(n in self.noises for n in noise_type)
            self.selected_noises = noise_type
        else:
            self.selected_noises = self._get_list(noise_type)

    def _get_list(self, noise_type: str):
        match noise_type:
            case 'background': return [
                'AirConditioner', 'Babble', 'AirportAnnouncements', 'Bus', 'CafeTeria', 'Cafe', 'Car', 'Field',
                'Hallway', 'Kitchen', 'LivingRoom', 'Metro', 'NeighborSpeaking', 'Office', 'Restaurant', 'Park',
                'Station', 'Traffic', 'Square', 'VacuumCleaner', 'WasherDryer']
            case 'event': return ['CopyMachine', 'SqueakyChair', 'Typing', 'ShuttingDoor']
            case 'single source': return [
                'AirConditioner', 'CopyMachine', 'NeighborSpeaking', 'ShuttingDoor', 'Typing', 'VacuumCleaner',
                'WasherDryer', 'Washing']
            case 'all': return [
                'AirConditioner', 'Babble', 'AirportAnnouncements', 'Bus', 'CafeTeria', 'Cafe', 'Car', 'Field',
                'Hallway', 'Kitchen', 'LivingRoom', 'Metro', 'NeighborSpeaking', 'Office', 'Restaurant', 'Park',
                'Station', 'Traffic', 'Square', 'VacuumCleaner', 'WasherDryer', 'CopyMachine', 'SqueakyChair', 'Typing',
                'ShuttingDoor', 'Munching']
            case _: raise AssertionError(noise_type)

    def get_random_noise(self, noises: list[str]):
        noise_type = random.choice(noises)
        idx = random.randrange(self.noises[noise_type])
        filename = f"{noise_type}_{idx + 1}.wav"
        return Path(self.train_path, filename)

    def __call__(self, size: Sequence[int]):
        assert len(size) == 1
        assert size[0] % GlobalConfig.win_size == 0
        noise_file = self.get_random_noise(self.selected_noises)
        if noise_file.name == "Munching_5.wav":
            return torch.zeros(size)

        noise, _ = load_audio(str(noise_file), 16_000, duration=size[0] / 16_000, last_frame='none')

        # Normalize to unit power
        noise /= torch.std(noise)

        # return noise

        padding = size[0] - noise.shape[0]
        if padding == 0:
            return noise
        else:
            assert padding > 0
            split = random.randrange(padding + 1)
            pad_front = torch.zeros(split)
            pad_end = torch.zeros(padding - split)
            return torch.cat([pad_front, noise, pad_end])


def _speed_test(dataset: Dataset):
    for audio, vad, ann in tqdm(dataset):
        pass


def speed_torch_dataset():
    from torchaudio.datasets import VCTK_092

    _speed_test(VCTK_092(root=os.path.join(Paths.data, "VCTK-Corpus-0.92"), mic_id='mic1', download=False))


if __name__ == '__main__':
    pass
    # timestamps = os.path.join(paths.vad_spp, "silero_vad_512_timestamps")
    # labels = LoadVADFromTimestamps(timestamps, cfg)
    # ls_data = LibriSpeech(labels=labels, config=default_config())
    # ls_data = ConvertLSDataset(ls_folder="../data/train-clean-100/LibriSpeech/train-clean-100",
    #                            new_folder=None, fs=16_000)
    # train_data = DataLoader(ls_data, batch_size=1, shuffle=True, num_workers=2)

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

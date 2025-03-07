import os
import random
from pathlib import Path
from typing import Any, Literal, Final, final, ClassVar, Callable
from dataclasses import dataclass, InitVar


@final
class GlobalConfig:
    sample_rate: Final[int] = 16_000  # sample rate in Hz
    win_size: Final[int] = 512
    frame_rate: Final[float] = sample_rate / win_size

    device: Final[str] = 'cpu'


@dataclass
class DataConfig:
    ground_truth: Literal['vad', 'speech', 'energy'] = 'vad'
    multi_channel: bool = False
    clean: bool = False
    reverb: bool = False
    num_devices: Callable[[], int] = lambda: 1
    num_mic: Callable[[], int] = None

    def __post_init__(self):
        if self.num_mic is None:
            if self.multi_channel:
                self.num_mic = lambda: random.choices([1, 2, 3, 4, 5], [0.0, .4, .2, .2, .2])[0]
            else:
                self.num_mic = lambda: 1


class Paths:
    root = Path("/")
    data = Path(root, "data")
    labels = Path(data, "preprocessed")
    silero_vad = Path(root, "silero_vad")

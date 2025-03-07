import torch
import os

from silero_vad.utils_vad import init_jit_model
from torch_framework.config import GlobalConfig, Paths


class SileroVAD:
    def __init__(self):
        self.model = init_jit_model(os.path.join(Paths.silero_vad, 'files', 'silero_vad.jit'))
        # print(self.model)

    @staticmethod
    def __find_silero_folder() -> str:
        folder = os.path.join(Paths.root, 'silero_vad')
        if os.path.exists(folder):
            return folder
        folder, _ = os.path.split(Paths.root)
        folder = os.path.join(folder, 'silero_vad')
        if os.path.exists(folder):
            return folder
        raise Exception("Could not find folder of silero_vad")

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        assert len(audio.shape) == 1, "More than one dimension in audio!"

        self.model.reset_states()

        num_frames = audio.shape[0] // GlobalConfig.win_size
        speech_probs = torch.empty((num_frames,))
        for k in range(num_frames):
            chunk = audio[k * GlobalConfig.win_size:(k + 1) * GlobalConfig.win_size]
            assert len(chunk) == GlobalConfig.win_size
            speech_probs[k] = self.model(chunk, GlobalConfig.sample_rate).item()

        return speech_probs


if __name__ == '__main__':
    from data.dataset import VCTK, MyAudioDataset
    from vad_labeling.labeling import spp2vad
    from utils.utils import plot_vad
    from matplotlib import pyplot as plt

    dataset = MyAudioDataset(data=VCTK().file_iterator(), last_frame='pad')
    silero = SileroVAD()
    heuristic = spp2vad(speech_pad_ms=50)
    print(len(dataset))
    for x, _, _ in dataset:
        y = silero(x)
        vad = heuristic(y)
        ax = plot_vad(x, y)
        ax.plot((torch.arange(0, vad.shape[0]) + .5) / GlobalConfig.frame_rate, 0.5 * vad)
        plt.show()

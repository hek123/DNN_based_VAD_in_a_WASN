import matplotlib.pyplot as plt
import torch
from torchaudio.transforms import MFCC
from tqdm import tqdm

from data.dataset import VCTKPreProcessed
from torch_framework.config import GlobalConfig
from torch_framework.multi_channel_dataset import MultiChannelData, DataConfig
from utils.visualization import plot_vad_on_audo
from vad_labeling.bayesan import gmm_vad, hmm_vad
from vad_labeling.labeling import spp2vad
from vad_labeling.silero import SileroVAD
from vad_labeling.simpleVAD import EnergyVAD, GMM

if __name__ == '__main__':

    dataset = VCTKPreProcessed(duration=15, add_short_silences=False)
    dataset = MultiChannelData(dataset, DataConfig())
    # dataset.add_localized_noise(MS_SNSD_Noise('background', (0, 0)))

    models = {
        'EnergyVAD': EnergyVAD(causal=False),
        'GMM': GMM(),
        'Silero-VAD': SileroVAD()
    }

    heuristic = spp2vad(speech_pad_ms=0.)

    avg_error = 0
    for idx, (audio, vad, ann) in enumerate(tqdm(dataset)):
        ns = audio.shape[1] - (audio.shape[1] % GlobalConfig.win_size)
        audio = audio[0, :ns]

        results = {}
        for name, model in models.items():
            y = model(audio)
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y)
            y_ = heuristic(y)
            results[name] = y_

            plot_vad_on_audo(audio, [y, y_])
            plt.title(name)

        plot_vad_on_audo(audio, list(results.values()), vad_names=list(results.keys()))
        plt.show()

    avg_error += torch.count_nonzero(y != vad) / y.shape[0]
    # print(f"error = {100 * torch.count_nonzero(y != data[1]) / y.shape[0]: .2f}%")

    print(f"error = {100 * avg_error / len(dataset): .2f}%")

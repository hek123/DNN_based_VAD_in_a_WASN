import os.path

import matplotlib.pyplot as plt

from data.dataset import LibriSpeech
from silero_vad.utils_vad import init_jit_model, get_speech_probs, prob_to_vad

import torch
from tqdm import tqdm
import numpy as np

from config import GlobalConfig


def test_silaro(k: int = 0):
    ls = LibriSpeech()

    model = init_jit_model("silero_vad/files/silero_vad.jit")

    x = ls[k][0]
    # get speech timestamps from full audio file
    speech_probs = get_speech_probs(x, model, sampling_rate=GlobalConfig.sample_rate, window_size_samples=GlobalConfig.win_size)
    speech_timestamps = prob_to_vad(speech_probs, x.shape[0])

    speech_probs = torch.repeat_interleave(torch.tensor(speech_probs), GlobalConfig.win_size)
    print(x.shape, len(speech_probs))
    vad = torch.zeros_like(x)
    for timestamp in speech_timestamps:
        vad[GlobalConfig.win_size * timestamp['start']:GlobalConfig.win_size * timestamp['end']] = 1.

    snr = 10 * torch.log10(torch.var(x[vad == 1.]) / torch.var(x[vad == 0.]))
    print("snr = %.2f" % snr + "dB")

    plt.figure()
    plt.plot(x[:10 * GlobalConfig.sample_rate])
    plt.plot(0.2 * speech_probs[:10 * GlobalConfig.sample_rate])
    plt.plot(0.3 * vad[:10 * GlobalConfig.sample_rate])
    plt.show()


def compute_prob(target_folder: str):
    ls = LibriSpeech("./data")

    model = init_jit_model("silero_vad/files/silero_vad.jit")

    # dataloader = DataLoader(ls, batch_size=1, shuffle=False, num_workers=0)
    for idx, data in enumerate(tqdm(ls)):
        probs = get_speech_probs(data[0], model, sampling_rate=ls.fs)

        file = ls.files[idx].replace('\\', '-')
        np.save(os.path.join(target_folder, file), probs)


if __name__ == '__main__':
    test_silaro(k=21)
    # compute_prob("./data/preprocessed/silero_vad_512_preproc")

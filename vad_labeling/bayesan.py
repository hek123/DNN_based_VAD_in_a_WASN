from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from hmmlearn import hmm, vhmm
from sklearn.mixture import GaussianMixture
from torchaudio.transforms import MFCC
from tqdm import tqdm

from utils.utils import sequential
from torch_framework.config import GlobalConfig


def delta(x, z: int = 1):
    print(x.shape)  # (F, T)
    delta_x = torch.zeros_like(x)
    for k in range(1, z + 1):
        delta_x[:, k:-k] += k * (x[:, 2 * k:] - x[:, :-2 * k])
    delta_x /= 2 * sum(k ** 2 for k in range(1, z + 1))
    return delta_x


def hmm_vad(hmm_type: str, cov_type: str = 'diag', verbose: bool = False,
            feature_fn: Callable = None) -> Callable:
    pi = np.array([0., 1.])
    A = np.array([[.99, .01],
                  [.01, .99]])
    if hmm_type == 'hmm':
        # model = hmm.GMMHMM(
        #     n_components=2, n_mix=3, covariance_type=cov_type,
        #     startprob_prior=pi, transmat_prior=A,
        #     params='cwmt',
        #     init_params='cwm',
        #     n_iter=20, tol=1.,
        #     verbose=verbose
        # )
        model = hmm.GaussianHMM(
            n_components=2, covariance_type=cov_type,
            startprob_prior=pi, transmat_prior=A,
            params='cm', init_params='cm',
            n_iter=20, tol=1.,
            verbose=verbose
        )
    elif hmm_type == 'vhmm':
        # Variational HMM seems to be superior to the regular HMM
        model = vhmm.VariationalGaussianHMM(
            n_components=2, covariance_type=cov_type, n_iter=100, tol=1e-3,
            startprob_prior=pi, transmat_prior=A,
            init_params='mc',
            params='mct',
            verbose=verbose
        )
    else:
        raise AssertionError()

    def fn(audio: torch.Tensor) -> torch.Tensor:
        # 1) Compute the feature vector from audio
        if feature_fn is not None:
            x = feature_fn(audio)[:, :-1].T
        else:
            x = audio[..., None]

        if verbose:
            plt.figure()
            # plt.imshow(20 * torch.log10(torch.abs(x.T) + 1e-8), aspect='auto', interpolation='none')
            plt.imshow(x.T, aspect='auto', interpolation='none')
            plt.show()

        # 2) Fit the HMM model onto it
        model.startprob_ = pi.copy()
        model.transmat_ = A.copy()
        model.fit(x)

        # 3) Predict the states
        y = model.predict_proba(x)
        # print(y.shape)
        x = audio.view(-1, GlobalConfig.win_size)
        var1 = torch.var(x[y[:, 0] > .5, :])
        var0 = torch.var(x[y[:, 0] <= .5, :])
        if var0 > var1:
            # var0, var1 = var1, var0
            return y[:, 1]

        return y[:, 0]

    return fn


def gmm_vad(feature_fn: Callable = None, verbose: bool = False, cov_type: str = 'diag') -> Callable:
    model = GaussianMixture(n_components=2, covariance_type=cov_type, verbose=verbose)

    def fn(audio: torch.Tensor) -> torch.Tensor:
        assert len(audio.shape) == 1, audio.shape
        assert audio.shape[0] % GlobalConfig.win_size == 0, audio.shape

        # 1) Compute the feature vector from audio
        if feature_fn is not None:
            x = feature_fn(audio)[:, :-1].T
            # x = feature_fn(audio).T
        else:
            x = audio.view(-1, GlobalConfig.win_size)

        if verbose:
            plt.figure()
            # plt.imshow(20 * torch.log10(torch.abs(x.T) + 1e-8), aspect='auto', interpolation='none')
            plt.imshow(x.T, aspect='auto', interpolation='none')
            plt.show()

        # 2) Fit the GMM model onto it
        model.fit(x)

        # 3) Predict the states
        y = model.predict_proba(x)
        x = audio.view(-1, GlobalConfig.win_size)
        var1 = torch.var(x[y[:, 0] > .5, :])
        var0 = torch.var(x[y[:, 0] <= .5, :])
        if var0 > var1:
            # var0, var1 = var1, var0
            return y[:, 1]

        return y[:, 0]

    return fn


if __name__ == '__main__':
    from data.dataset import LibriSpeech
    from data.inspection import plot_spectrum
    from vad_labeling.labeling import spp2vad

    # out_folder = os.path.join(Paths.vad_spp, "VarHMM_512")
    dataset = LibriSpeech()

    feature_function = sequential(
        MFCC(sample_rate=GlobalConfig.sample_rate, n_mfcc=32,
             melkwargs={
                 'n_fft': GlobalConfig.win_size * 2, 'win_length': 2 * GlobalConfig.win_size,
                 'hop_length': GlobalConfig.win_size, 'center': True,
                 'f_min': 50, 'f_max': GlobalConfig.sample_rate // 2, 'n_mels': 128
             }),
        lambda x: torch.cat([x, delta(x, 2)])
    )

    label = hmm_vad(
        'hmm', verbose=True, cov_type='diag',
        feature_fn=feature_function
        # feature_fn=sequential(
        #     Spectrogram(n_fft=2 * Config.win_size, win_length=2 * Config.win_size, hop_length=Config.win_size,
        #                 onesided=True),
        #     torch.abs,
        #     lambda x: 20 * torch.log10(x + 1e-8)
        # )
    )
    # label = gmm_vad(feature_fn=feature_function, verbose=True)
    heuristic = spp2vad()

    for audio, _, annotations in tqdm(dataset):
        y = label("", audio, None, None)
        # print(f"{var0=}, {var1=}\n"
        #       f"SNR = {10 * torch.log10(var1 / var0): .2f}dB\n"
        #       f"H1 = {100 * np.sum(y) / y.shape[0]: .2f}%")
        # timestamps = prob_to_vad(y, audio.shape[0])
        vad = heuristic(y)

        plot_spectrum(audio, vad, GlobalConfig.win_size * 2)
        # arr = np.empty((len(timestamps), 2), dtype=np.uint32)
        # for t, timestamp in enumerate(timestamps):
        #     arr[t, 0] = timestamp['start']
        #     arr[t, 1] = timestamp['end']
        # np.save(os.path.join(out_folder, annotations['filename']), arr)

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from hmmlearn import hmm
from hmmlearn import vhmm
from torchaudio.transforms import MFCC
from tqdm import tqdm

from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal, norm


class VADHMM:
    def __init__(self, D: int, C: int, covariance: str = 'diag'):
        assert D > 0
        assert C > 0
        assert covariance in {'diag', 'full'}

        self.covariance = covariance

        self._pi = np.array([.5, .5])
        self._A = np.array([[.95, .05],
                            [.05, .95]])
        self.N = 2
        self.D = D
        self.C = C

        self.b_n = self._new_gaussian(covariance, 0, 1e-3)

        self._c = np.random.rand(C) / C
        self.b_s = [self._new_gaussian(covariance, variance=.1) for _ in range(C)]

    def b(self, x):
        return np.array([self.b_n.logpdf(x), sum(self._c[c]*self.b_s[c].logpdf(x) for c in range(self.C))])

    def _alpha(self, b: np.ndarray):
        pass

    def _beta(self, b: np.ndarray):
        pass

    def initial_state_est(self, x):
        """
        :param x: Input data (features, samples)
        :return: VAD estimate
        """
        y = GaussianMixture(n_components=2, covariance_type='diag').fit_predict(x.T)
        var0, var1 = float(torch.median(x[:, y <= .5])), float(torch.median(x[:, y > .5]))
        return y if var0 < var1 else 1 - y

    def _new_gaussian(self, covariance: str, mean: float = None, variance: float = 1.):
        mean = np.random.randn(self.D) if mean is None else mean
        if covariance == 'full':
            return multivariate_normal(mean=mean, cov=variance)
        elif covariance == 'diag':
            return norm(loc=mean, scale=variance)


class BaumWelch:
    def __init__(self, hmm: VADHMM, update: str = 'tmc'):
        self.hmm = hmm

        if 't' in update:
            self._A = np.zeros((hmm.N, hmm.N))

    def step(self, x: np.ndarray):
        pass

    def update(self):
        pass


if __name__ == '__main__':
    from data.dataset import LibriSpeech, LoadVADFromTimestamps, default_path
    from data.inspection import plot_spectrum
    from torch_framework.config import default_config
    from vad_labeling.heuristics import prob_to_vad, timestamps2vad

    cfg = default_config()
    paths = default_path()

    # model = hmm.GMMHMM(n_components=2, n_mix=2, covariance_type='diag',
    #                    startprob_prior=np.array([.5, .5]),
    #                    transmat_prior=np.array([[.99, .01],
    #                                             [.01, .99]]),
    #                    params='mcw',
    #                    init_params='mcw',
    #                    verbose=True
    #                    )

    # Variational HMM seems to be far superior to the regular HMM
    model = vhmm.VariationalGaussianHMM(n_components=2, covariance_type='full', n_iter=100, tol=1e-3,
                                        startprob_prior=np.array([.5, .5]),
                                        transmat_prior=np.array([[.99, .01],
                                                                 [.01, .99]]),
                                        init_params='mc',
                                        params='smct',
                                        verbose=False)

    feature = MFCC(sample_rate=cfg.sample_rate, n_mfcc=12,
                   melkwargs={'n_fft': cfg.win_size * 2, 'hop_length': cfg.win_size, 'center': False})

    out_folder = os.path.join(paths.vad_spp, "VarHMM_512")
    dataset = LibriSpeech(config=cfg)
    for audio, _, annotations in tqdm(dataset):
        # 1) Compute the feature vector from audio
        mfcc = feature(audio).T
        mfcc = torch.cat([torch.zeros(1, 12), mfcc], dim=0)
        # print(mfcc.shape, audio.shape, audio.shape[0] / cfg.win_size)
        # plt.figure()
        # plt.imshow(mfcc, aspect='auto')
        # plt.show()

        # 2) Fit the HMM model onto it
        model.startprob_ = np.array([.5, .5])
        model.transmat_ = np.array([[.99, .01],
                                    [.01, .99]])
        model.fit(mfcc)
        # print(model.transmat_)

        # 3) Predict the states
        y = model.predict(mfcc)
        y_sample = np.repeat(y, cfg.win_size)
        var1 = torch.var(audio[y_sample == 1])
        var0 = torch.var(audio[y_sample == 0])
        if var0 > var1:
            var0, var1 = var1, var0
            y = 1 - y
        # print(f"{var0=}, {var1=}\n"
        #       f"SNR = {10 * torch.log10(var1 / var0): .2f}dB\n"
        #       f"H1 = {100 * np.sum(y) / y.shape[0]: .2f}%")
        timestamps = prob_to_vad(y, audio.shape[0])
        # vad = timestamps2vad(timestamps, audio.shape[0], cfg.win_size)

        # plot_spectrum(audio, vad, cfg.win_size * 2, cfg)
        # arr = np.empty((len(timestamps), 2), dtype=np.uint32)
        # for t, timestamp in enumerate(timestamps):
        #     arr[t, 0] = timestamp['start']
        #     arr[t, 1] = timestamp['end']
        # np.save(os.path.join(out_folder, annotations['filename']), arr)

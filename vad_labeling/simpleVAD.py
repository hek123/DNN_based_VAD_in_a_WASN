import torch
from torch import Tensor
import numpy as np

from hmmlearn import hmm, vhmm
from sklearn.mixture import GaussianMixture
from torchaudio.transforms import MFCC, MelSpectrogram

from torch_framework.config import GlobalConfig


class EnergyVAD:
    def __init__(self, causal: bool = True, increment: float = 1.001, decrement: float = None):
        assert 1. < increment, increment

        self.causal = causal
        self.increment, self.decrement = increment, 1. / increment if decrement is None else decrement

    def __call__(self, speech: Tensor):
        E = self.rmse(speech)
        return self.f(E, self.causal, self.increment, self.decrement)

    @staticmethod
    def rmse(speech: Tensor, window: int = GlobalConfig.win_size, step: int = None):
        x = speech.unfold(size=window, step=window if step is None else step, dimension=-1)
        return torch.sqrt(torch.mean(torch.square(x), dim=-1))

    @staticmethod
    def f(energy: Tensor, causal: bool = True, increment: float = 1.001, decrement: float = None, debug: bool = False):
        assert len(energy.shape) == 1, energy.shape
        assert torch.all(energy >= 0.)
        assert 1. < increment, increment
        decrement = 1. / increment if decrement is None else decrement
        assert decrement < 1., decrement

        threshold = torch.empty_like(energy)
        Emin, Emax = torch.empty_like(energy), torch.empty_like(energy)
        Emin[0] = Emax[0] = energy[0]
        threshold[0] = energy[0]
        for t in range(1, energy.shape[0]):
            Emin[t] = torch.minimum(Emin[t - 1], torch.maximum(energy[t], torch.tensor(1e-6)))
            Emax[t] = torch.maximum(Emax[t - 1], energy[t])

            threshold[t] = Emin[t] + (Emax[t] - Emin[t]) / 20

            if energy[t] > threshold[t]:
                Emax[t] *= decrement
            else:
                Emin[t] *= increment
        if not causal:  # If non-causal, do also a backward pass
            energy, Emin, Emax = energy.flip(0), Emin.flip(0), Emax.flip(0)
            for t in range(1, energy.shape[0]):
                Emin[t] = torch.minimum(Emin[t - 1], Emin[t])
                Emax[t] = torch.maximum(Emax[t - 1], Emax[t])

                threshold[t] = Emin[t] + (Emax[t] - Emin[t]) / 20

                if energy[t] > threshold[t]:
                    Emax[t] *= decrement
                else:
                    Emin[t] *= increment
            energy, Emin, Emax, threshold = energy.flip(0), Emin.flip(0), Emax.flip(0), threshold.flip(0)

        if debug:
            return energy > threshold, {'threshold': threshold, 'e_min': Emin, 'e_max': Emax}
        else:
            return energy > threshold


class GMM:
    def __init__(self):
        mfcc = MFCC(
            sample_rate=GlobalConfig.sample_rate, n_mfcc=32,
            melkwargs={
                'n_fft': GlobalConfig.win_size * 2, 'win_length': 2 * GlobalConfig.win_size,
                'hop_length': GlobalConfig.win_size, 'center': True, 'normalized': True,
                'f_min': 50, 'f_max': GlobalConfig.sample_rate // 2, 'n_mels': 128
            })
        # ms = MelSpectrogram(sample_rate=GlobalConfig.sample_rate, n_fft=2 * GlobalConfig.win_size,
        #                     win_length=2 * GlobalConfig.win_size, hop_length=GlobalConfig.win_size,
        #                     f_min=50, f_max=GlobalConfig.sample_rate // 2, n_mels=128,
        #                     normalized=True, center=True, power=1.)
        self.feature_fn = lambda x: torch.cat([mfcc(x)[:, :-1], torch.log(EnergyVAD.rmse(x))[None, :]], dim=0)
        # self.feature_fn = lambda x: torch.log(EnergyVAD.rmse(x))
        # self.model = GaussianMixture(n_components=2, covariance_type='diag')

        self.model = hmm.GaussianHMM(n_components=2, covariance_type='diag',
                                     params='smc', init_params='mc', algorithm='map')
        p_trans = 1e-8
        self.model.transmat_ = np.array([[1. - p_trans, p_trans],
                                         [p_trans, 1. - p_trans]])

    def __call__(self, speech: Tensor):
        assert len(speech.shape) == 1, speech.shape

        # 1) Compute the feature vectors
        x = self.feature_fn(speech).T

        # 2) Fit the GMM model onto it
        self.model.startprob_ = np.array([.5, .5])
        self.model.fit(x)

        # self.hmm.means_ = self.model.means_
        # self.hmm.covars_ = self.model.covariances_

        # 3) Predict the states
        y1 = self.model.predict_proba(x)

        x = speech.view(-1, GlobalConfig.win_size)
        var1 = torch.var(x[y1[:, 0] > .5, :])
        var0 = torch.var(x[y1[:, 0] <= .5, :])
        if var0 > var1:
            # var0, var1 = var1, var0
            return y1[:, 1]

        return y1[:, 0]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from data.dataset import VCTKPreProcessed
    from tqdm import tqdm

    from torch.utils.data import DataLoader
    from vad_labeling.labeling import spp2vad

    dataset = DataLoader(VCTKPreProcessed(60), shuffle=True, collate_fn=lambda x: x[0], batch_size=1)
    heuristic = spp2vad(speech_pad_ms=0)

    # sounddevice.play(dataset[273], Fs)
    # sounddevice.play(dataset[0], Fs)
    # sounddevice.wait()

    avg_error = 0
    for idx, (audio, vad, ann) in enumerate(tqdm(dataset)):
        # x = torch.nn.functional.pad(audio, (GlobalConfig.win_size, 0))
        # x = x.unfold(size=2 * GlobalConfig.win_size, step=GlobalConfig.win_size, dimension=-1)
        # E = torch.sqrt(torch.mean(torch.square(x), dim=-1))
        E = EnergyVAD.rmse(audio, window=GlobalConfig.win_size, step=GlobalConfig.win_size)
        # plt.hist(torch.log(E[E > 0]), bins=50)
        # plt.show()
        y, info = EnergyVAD.f(E, causal=False, increment=1.001, debug=True)

        y_ = heuristic(y)

        T = 10
        Nf = round(T * GlobalConfig.frame_rate)
        Ns = GlobalConfig.win_size * Nf
        t_sample = torch.linspace(0, T, Ns)
        t_frame = torch.linspace(0, T, Nf)

        ax2: plt.Axes
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all')
        ax1.plot(t_sample, audio[:Ns], label='audio')
        scale = 2. * torch.std(audio[:Ns])
        ax1.plot(t_frame, 1.1 * scale * vad[:Nf], label='vad-Silero')
        ax1.plot(t_frame, .9 * scale * y[:Nf], label='vad-DLED')
        ax1.plot(t_frame, scale * y_[:Nf], label='vad-DLED+h')
        ax1.legend()

        ax2.plot(t_frame, E[:Nf], label='E')
        ax2.plot(t_frame, info['e_min'][:Nf], label='Emin')
        ax2.plot(t_frame, info['e_max'][:Nf], label='Emax')
        ax2.plot(t_frame, info['threshold'][:Nf], label='threshold')
        ax2.legend()

        plt.show()

        avg_error += torch.count_nonzero(y != vad) / y.shape[0]
        # print(f"error = {100 * torch.count_nonzero(y != data[1]) / y.shape[0]: .2f}%")

        # # compress
        # diff = torch.diff(vad, prepend=tensor([False]))
        # switches = torch.argwhere(diff)
        # rlc = torch.diff(torch.squeeze(switches), prepend=tensor([-1]))
        #
        # # decode
        # switches_ = torch.cumsum(rlc, dim=0)
        # diff_ = torch.zeros((data.shape[1] + 1,))
        # diff_[switches_[::2]] = 1
        # diff_[switches_[1::2]] = -1
        # vad_ = torch.cumsum(diff_, dim=0)[1:]
        #
        # if not torch.all(vad_ == vad):
        #     plt.figure()
        #     plt.plot(data[0, :10*Fs])
        #     plt.plot(0.2*vad[:10*Fs])
        #     plt.plot(0.2*diff[:10*Fs])
        #     plt.plot(0.19 * diff_[:10 * Fs], '--')
        #     plt.plot(0.19 * vad_[:10 * Fs], '--')
        #     plt.vlines(switches[switches < 10*Fs], 0, 0.15, colors='k')
        #     plt.vlines(switches_[switches_ < 10 * Fs], 0, 0.15, colors='k', linestyles='--')
        #
        #     plt.figure()
        #     plt.subplot(311)
        #     plt.plot(data[0, :10 * Fs])
        #     plt.subplot(312)
        #     plt.plot((data - torch.mean(data))[0, :10 * Fs])
        #     plt.subplot(313)
        #     plt.plot((data - torch.median(data))[0, :10 * Fs])
        #
        #     plt.show()
        #
        # file = dataset.files[idx].replace('\\', '-')
        # # torch.save(rlc, f"../data/preprocessed/vad/{file}.pt")

    print(f"error = {100 * avg_error / len(dataset): .2f}%")

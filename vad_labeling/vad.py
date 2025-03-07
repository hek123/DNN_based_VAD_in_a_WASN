import numpy as np
from matplotlib import pyplot as plt
import torch
from sklearn.mixture import GaussianMixture as GMM

from utils.utils import isPow2
# from vad_labeling import heuristics


class SpectralVAD:
    def __init__(self, window=None, fusion='late', proba=True):
        assert isPow2(GlobalConfig.win_size)
        self.win = window
        self.proba = proba
        if fusion == 'early':
            self._fn = self._early_fusion
        elif fusion == 'late':
            self._fn = self._late_fusion
        else:
            raise Exception(f"Invalid option for parameter fusion: {fusion}. Options are: 'early' and 'late'")

    def _spectrum(self, x: torch.Tensor) -> torch.Tensor:
        X = torch.stft(x, GlobalConfig.win_size * 2, GlobalConfig.win_size, window=self.win, normalized=True, onesided=True,
                       return_complex=True, center=True, pad_mode="constant")
        return 20 * torch.log10(torch.abs(X[:, :-1]) + 1e-6)

    def _late_fusion(self, S: torch.Tensor) -> torch.Tensor:
        # 1) Fit a GMM with 2 components (speech & noise) to every frequency bin separately
        nfft2, L = S.shape
        labels = np.empty((nfft2 - 2, L))
        snr = torch.empty((nfft2 - 2, 1))
        for k in range(1, nfft2 - 1):
            gmm = GMM(n_components=2, covariance_type='diag').fit(S[k, :, None])
            if self.proba:
                y = gmm.predict_proba(S[k, :, None])[:, 0]
            else:
                y = gmm.predict(S[k, :, None])
            var0, var1 = float(torch.median(S[k, y <= .5])), float(torch.median(S[k, y > .5]))
            if var0 > var1:
                var0, var1 = var1, var0
                y = 1 - y
            labels[k - 1] = y
            snr[k - 1] = var1 - var0

        # 2) Average the freq bins using the SNR as weighting
        weights = 10 ** (snr / 20)
        weights /= torch.sum(weights)

        # TODO: possibility to add proper frequency scaling
        # log_f = 1 / torch.linspace(0, Config.fs // 2, nfft2)[1:-1, None]
        # log_f /= torch.sum(log_f)

        return torch.sum(weights * labels, dim=0)

    def _early_fusion(self, S: torch.Tensor) -> torch.Tensor:
        # 1) Fit a GMM with 2 components (speech & noise) on the signal
        gmm = GMM(n_components=2, covariance_type='diag').fit(S.T)
        if self.proba:
            y = gmm.predict_proba(S.T)[:, 0]
        else:
            y = gmm.predict(S.T)
        # print(y.shape)

        var0, var1 = float(torch.median(S[:, y <= .5])), float(torch.median(S[:, y > .5]))
        return y if var0 < var1 else 1 - y

    def __call__(self, x: torch.Tensor):
        assert len(x.shape) == 1, x.shape

        # Compute spectrum of input signal
        S = self._spectrum(x)
        # print(S.shape)

        return self._fn(S)

        # _, (ax1, ax2) = plt.subplots(2, 1)
        # ax1.plot(snr)
        # ax2.plot(weights)
        #
        # fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all')
        # ax1.imshow(S, aspect='auto', interpolation=None)
        # ax2.imshow(labels, aspect='auto', interpolation=None)
        #
        # T = 60 * 16_000
        # fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all')
        # ax1.plot(x[:T])
        # ax2.plot(np.arange(0, T, 512), spp[:T//512])
        # plt.show()


if __name__ == '__main__':
    from data.dataset import LibriSpeech
    from config import GlobalConfig
    from vad_labeling import labeling

    T = 120
    dataset = LibriSpeech(duration=T, labels=labeling.load_labels(folder="silero_vad_512_timestamp", out="vad"),
                          random_offset=False)
    # dataset = Stimuli(duration=T)

    x, silero_vad, ann = dataset[0]
    print(f"[{ann['idx']}]: {ann['filename']}")
    print(x.shape[0], T * GlobalConfig.sample_rate)
    print(x.shape[0] // GlobalConfig.win_size, silero_vad.shape[0], T * GlobalConfig.sample_rate // GlobalConfig.win_size)
    hmm_vad = labeling.load_labels(folder="VarHMM_512", out="vad")(
        ann['filename'], None, 0, T * GlobalConfig.sample_rate // GlobalConfig.win_size)
    print("loaded HMM")

    spp2vad = labeling.spp2vad()
    spp1 = SpectralVAD(fusion='early')(x)
    vad1 = spp2vad(spp1)

    spp2 = SpectralVAD(fusion='late')(x)
    vad2 = spp2vad(spp2)

    print("Computed GMM")

    num_samples = T * GlobalConfig.sample_rate
    num_frames = T * GlobalConfig.sample_rate // GlobalConfig.win_size

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='all')
    decimate = 8
    ax1.plot(np.linspace(0, T, num_samples // decimate), x[::decimate])
    ax1.set_title('Audio signal')

    t_frame = np.linspace(0, T, num_frames)
    ax2.plot(t_frame, .95*vad1, label='early')
    ax2.plot(t_frame, vad2, label='late')
    ax2.plot(t_frame, 1.05*silero_vad, label='silero')
    ax2.plot(t_frame, 1.1 * hmm_vad, label='hmm')
    ax2.set_title('VAD')
    ax2.legend()

    ax3.plot(t_frame, spp1, label='early', alpha=.5)
    ax3.plot(t_frame, spp2, label='late', alpha=.5)
    ax3.set_title('SPP')
    ax3.legend()

    plt.show()

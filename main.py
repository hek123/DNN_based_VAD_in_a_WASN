import random

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, LinearLR
from torch.utils.data import DataLoader, random_split

import neptune
# from ray import tune

# import data.dataset as data
import config as config
import models.main as m
import models.encoders as encoders

from training_framework.train import TrainConfig


def time_to_num_iterations(time: list[float], train_config: TrainConfig, unit: str = 'sec'):
    audio_time_per_iteration = train_config.duration * train_config.batch_size
    scale = {'sec': 1, 'min': 60, 'hour': 60 * 60}[unit]
    return [round(scale * t / audio_time_per_iteration) for t in time]


def get_dataloader(train_config: TrainConfig, data_config: config.DataConfig, run):
    # vctk_dataset = data.VCTKPreProcessed(duration=train_config.duration, add_full_silence=2, random_offset=True, log=run)
    #
    # dataset = mdata.MultiChannelData(vctk_dataset, config=data_config, log=run)
    # if not data_config.clean:
    #     dataset.add_iid_noise(p.ColoredNoise('white', (10, 60)))
    #     # default: (-5, 15, 0)
    #     dataset.add_localized_noise(data.MS_SNSD_Noise('single source', (0, 30, 10)), p=.95)
    #     # dataset.add_localized_noise(data.NoiseFromDataset(vctk_dataset, (5, 30, 10)))
    # dataset = mdata.SavedDataset('mc-sd')

    # else:
    #     dataset.post_process = [
    #         p.ComputeSNR("SNR_clean"),
    #         p.AddNoise(p.ColoredNoise('white', (15, 25)), add_to_current_noise=True, name='white noise'),
    #         # p.AddNoise(p.ColoredNoise(snr=(5, 20), color='pink'), name='pink noise'),
    #         p.AddNoise(p.SpectralNoise(dataset.meta['spectrum'], snr=(3, 15), nfft=2 * config.GlobalConfig.win_size),
    #                    name='spectral noise'),
    #         p.AddNoise(data.MS_SNSD_Noise('all', snr=(-5, 20)), name="MS-SNSD")
    #         # p.ComputeSNR("SNR_input")
    #     ]
    #     collate_fn = data.default_collate_fn
    #     # collate_fn = data.unbatched
    #     # collate_fn = mdata.multi2single_channel

    train_set, validation_set = random_split(dataset, [1. - train_config.validation_split, train_config.validation_split])
    train_set = DataLoader(train_set, batch_size=train_config.batch_size, shuffle=True, num_workers=0,
                           collate_fn=lambda x: x, persistent_workers=False, drop_last=False, prefetch_factor=None)
    validation_set = DataLoader(validation_set, batch_size=1,
                                shuffle=True, num_workers=0, persistent_workers=False, drop_last=False,
                                collate_fn=lambda x: x)

    return train_set, validation_set


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(False)
    log_run = False

    # for idx, (num_layers, multiplier, linear) in enumerate(product([1, 2, 3, 9], [1, 1.5, 2], [True, False])):
    for _ in range(1):
        # Create a Neptune run object for logging
        run = neptune.init_run(
            project="hektor/thesis",
            # api_token=os.environ["NEPTUNE_API_TOKEN"],
            name="BLSTM_baseline-saved",
            tags=["SingleChannel", 'non-causal', "distr_setting"],  # optional,
            source_files=[],
            description=f""
            # description="STFT+LSTM; TSNormV2(.01, affine=False); FocalLoss(gamma=0); calibration=3"
        ) if log_run else None

        train_config = TrainConfig(duration=30, batch_size=8, epochs=16, learning_rate=1e-3, validation_split=.15,
                                   calibration_duration=3)
        data_config = config.DataConfig(multi_channel=True, clean=False, ground_truth='vad',
                                        num_devices=lambda: random.randint(1, 1))

        train_set, validation_set = get_dataloader(train_config, data_config, run)

        # Configure some layers
        from models.custom_layers import InstanceNorm1d

        InstanceNorm1d.set_mode(use_ema='false', ema_init='reflect')
        # FilterResponseNorm.set_mode(use_ema='false')

        # encoder = encoders.ConvEncoder.default_encoder(num_layers, multiplier,
        #                                                act_fn=None if linear else torch.nn.functional.relu,
        #                                                bias=not linear)
        # encoder = encoders.ConvEncoder.default_encoder(3, 2)
        encoder = encoders.STFT()
        enc_params = sum([p.numel() for p in encoder.parameters() if p.requires_grad])
        print(f"#params encoder: {enc_params}")
        # run['sweep/ConvEnc'] = {'num_layers': num_layers, 'multiplier': float(multiplier), 'linear': linear,
        #                         'num_params': enc_params}

        from models.core_network.baseline import M1FF as Core

        core_network = Core(encoder.out_features)

        hidden = []

        if data_config.ground_truth == 'vad':
            model = m.VADModel(encoder, core_network, hidden=hidden, gamma=1.)
        elif data_config.ground_truth == 'energy':
            model = m.EnergyModel(encoder, core_network, hidden=hidden)
        else:
            raise AssertionError("TODO")

        print(model.model)
        # from torchinfo import summary
        # summary(model.model, (1, round(train_config.duration * 16e3)))

        # for p in model.parameters():
        #     p.register_hook(lambda grad: torch.clamp(grad, -1e6, 1e6))

        optimizer = Adam(model.model.parameters(), lr=train_config.learning_rate, weight_decay=0., amsgrad=False)
        milestones = time_to_num_iterations([5, 15, 30, 60], train_config, unit='hour')
        print(f"{milestones=}")
        schedulers = [
            MultiStepLR(optimizer, milestones, gamma=0.1),
            LinearLR(optimizer, .1, 1., total_iters=10),
        ]

        # try:
        #     train_model(model, train_set=train_set, validation_set=validation_set,
        #                 optimizer=optimizer, schedulers=schedulers,
        #                 train_config=train_config, run=run, visualize=False)
        # except KeyboardInterrupt:
        #     pass
        # except Exception as e:
        #     raise e

        # torch.save(model.core.state_dict(),
        #            "C:\\Users\\hekto\\PycharmProjects\\MyThesis\\code\\training_framework\\models\\saved_models\\lstm-nc.pt")

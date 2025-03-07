import math

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from dataclasses import dataclass, InitVar
from neptune import Run
from neptune_pytorch import NeptuneLogger
from neptune.types import File

from config import GlobalConfig
import utils.utils as utils
import utils.visualization as plot
from training_framework.multi_channel_dataset import align_labels
from vad_labeling.labeling import spp2vad
from models.main import Model


def train_model(model: Model, train_set: DataLoader, validation_set: DataLoader,
                optimizer: Optimizer, schedulers: list, train_config: "TrainConfig", run: Run | None,
                visualize: bool = False):

    # Log training settings
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    if run is not None:
        # Set up neptune logger
        npt_logger = NeptuneLogger(run=run, model=model.model, log_model_diagram=False, log_parameters=False, log_freq=10)
        run[npt_logger.base_namespace]["hyperparams"] = train_config.__dict__

        run["trainable_params"] = trainable_params
        run["train_config/optimizer"] = repr(optimizer)
        for scheduler in schedulers:
            run["train_config/lr_schedulers"].append(str(scheduler))

    print(f"trainable params: {trainable_params}")

    vad_heuristic = spp2vad(speech_pad_ms=0)
    num_plots = 1 if run is None else 8

    def apply_sample(audio: Tensor, y_true: Tensor):
        assert len(y_true.shape) == 2, y_true.shape
        # print(audio.shape)
        y_pred = model.model(audio)
        audio, y_pred, y_true = align_labels(audio, y_pred, y_true, ann['source_pos'][:, 0], ann['mic_pos'],
                                             train_config.output_valid)
        assert y_pred.shape == y_true.shape, f"{y_pred.shape=}, {y_true.shape=}, audio: {audio.shape[-1] / GlobalConfig.win_size}"

        (audio, y_true, y_pred), snr = sort_by_snr(audio, y_true, y_pred)

        loss = model.apply_loss(y_pred, y_true, snr=snr)

        return audio, y_pred, y_true, loss, snr

    it = 0
    for epoch in range(train_config.epochs):
        model.model.train(True)
        avg_loss, train_err = torch.zeros(()), torch.zeros((0,))
        if run is not None:
            run['current_epoch'] = epoch + 1
        for batch in train_set:
            # Zero your gradients for every batch!
            optimizer.zero_grad()

            loss, err = torch.empty(len(batch)), torch.empty(len(batch))
            for b, (audio, y_true, ann) in enumerate(batch):
                _, y_pred, y_true, l, _ = apply_sample(audio, y_true)
                loss[b] = l
                err[b] = error(y_pred >= .5, y_true)

            loss = torch.mean(loss)
            # if epoch < 5:
            #     loss = torch.mean(loss)
            # elif epoch < 12:
            #     # Only use worst 50% for backprop; -> ignore easy cases
            #     loss, _ = torch.sort(loss)
            #     loss = torch.mean(loss[len(batch) // 2:])
            # else:
            #     # Train on worst case only
            #     loss = torch.max(loss)

            loss.backward()

            # for name, param in model.model.named_parameters():
            #     # print(name, torch.isfinite(param.grad).all())
            #     print(name, torch.max(torch.abs(param.grad)))

            optimizer.step()

            if run is not None:
                run["train/loss"].append(loss.item())
                run["train/learning_rate"].append(optimizer.param_groups[-1]['lr'])
                run["train/error"].append(100 * torch.mean(err).item())

            print(f"\rEpoch {epoch+1}, it {it%len(train_set)+1:02d}/{len(train_set)}: "
                  f"Loss={loss.item():.3e}, MeanError={100 * torch.mean(err).item():05.2f}%, "
                  f"lr={optimizer.param_groups[-1]['lr']: .2e}",
                  end='')

            for s in schedulers:
                s.step()

            avg_loss += loss.detach()
            train_err = torch.cat([train_err, err])

            it += 1

        print(f"\ntrain loss: {avg_loss.item() / len(train_set):.3e}, average train error: {100 * torch.mean(train_err).item():05.2f}%")

        for name, param in model.model.named_parameters():
            if name.endswith('_ema._momentum'):
                print(name, param.data.item(), torch.nn.functional.sigmoid(param.data).item())
                if run is not None:
                    run['other/TimeSeriesNorm._momentum'][name].append(torch.nn.functional.sigmoid(param.data).item())

        if run is not None:
            npt_logger.log_checkpoint()

        # Validation
        model.model.train(False)
        with torch.no_grad():
            loss, err_wo_h, err_w_h = 0, 0, 0
            val_err = torch.empty((len(validation_set),))
            for k, batch in enumerate(validation_set):
                assert len(batch) == 1, len(batch)
                audio, y_true, ann = batch[0]

                # assert len(y_true.shape) == 2, y_true.shape
                # y_pred = model.model(audio)
                # audio, y_pred, y_true = align_labels(audio, y_pred, y_true, ann['source_pos'][:, 0], ann['mic_pos'],
                #                                      train_config.output_valid)
                # assert y_pred.shape == y_true.shape, f"{y_pred.shape=}, {y_true.shape=}"
                #
                # # Val loss
                # loss += model.apply_loss(y_pred, y_true)
                audio, y_pred, y_true, l, _ = apply_sample(audio, y_true)
                loss += l

                err = error(y_pred >= .5, y_true)
                err_wo_h += err
                val_err[k] = err
                err_w_h += error(vad_heuristic(y_pred), y_true)

                # Visualize
                if (visualize or epoch == train_config.epochs - 1) and k < num_plots:
                    # idx = random.randrange(audio.shape[0])
                    # y_pred = model.finalize_output(y_pred)
                    # if audio.shape[0] == y_true.shape[0]:  # Single channel
                    #     audio, vad, pred = audio[idx], y_true[idx], y_pred[idx]
                    # else:
                    #     assert vad.shape[0] == pred.shape[0] == 1, (audio.shape, vad.shape, pred.shape)
                    #     audio, vad, pred = audio[idx], vad[0], pred[0]
                    #
                    # fig = plot.visualize_prediction(audio, vad, pred, T=5, heuristic=vad_heuristic)
                    fig = plot.viz_pred_distributed(audio, y_true, y_pred, T=5)

                    if run is not None:
                        # try:
                        #     doa_speaker = (ann['speaker']['theta'] - ann['LMA']['phi']) % (2 * torch.pi)
                        #     doa_inference = (ann['interfering_noise']['theta'] - ann['LMA']['phi']) % (2 * torch.pi)
                        #     descr = f"num mic = {ann['LMA']['M']}\n" \
                        #             f"doa speaker = {doa_speaker * 180 / torch.pi: .2f}\n" \
                        #             f"doa inference = {doa_inference * 180 / torch.pi: .2f}"
                        # except TypeError or KeyError:
                        descr = ""
                        run["val/prediction"].append(File.as_image(fig), name=f"epoch{epoch}-{k}", description=descr)
                        run.wait()
                        plt.close(fig)
                    elif visualize:
                        plt.show()

            loss /= len(validation_set)
            err_w_h /= len(validation_set)
            err_wo_h /= len(validation_set)

            if visualize or (epoch == train_config.epochs - 1):
                fig = plot.err_distribution(train_err, val_err)

                if run is not None:
                    run["val/prediction"].append(File.as_image(fig), name=f"epoch{epoch}")
                    run.wait()
                    plt.close(fig)
                elif visualize:
                    plt.show()

            if run is not None:
                run["val/loss"].append(loss)
                run["val/error"].append(100 * err_wo_h)

            print(f"validation loss: {loss:.3e}, validation error: (w){100 * err_w_h:05.2f}%, (wo){100 * err_wo_h:05.2f}%")

    if run is not None:
        npt_logger.log_model("model")
        run.stop()


def binary_accuracy(y_pred: Tensor, y_true: Tensor, spp_vad=torch.round) -> float:
    assert len(y_pred.shape) == 2
    assert y_true.shape == y_pred.shape, f"{y_true.shape=}, {y_pred.shape=}"
    assert utils.is_binary(y_true)
    assert utils.is_probability(y_pred)
    return float(torch.mean(torch.sum(spp_vad(y_pred) == spp_vad(y_true), dim=1) / y_true.shape[1]))


def error(y_pred: Tensor, y_true: Tensor) -> float:
    assert len(y_pred.shape) == 2 or len(y_pred.shape) == 1
    assert y_true.shape == y_pred.shape, f"{y_true.shape=}, {y_pred.shape=}"
    assert utils.is_binary(y_true)
    assert utils.is_binary(y_pred)

    if len(y_pred.shape) == 1:
        return float(torch.sum(y_pred != y_true) / y_true.shape[0])
    else:
        return float(torch.mean(torch.sum(y_pred != y_true, dim=1) / y_true.shape[1]))


@dataclass
class TrainConfig:
    epochs: int
    learning_rate: float

    duration: float | None

    validation_split: float

    batch_size: int = 8

    calibration_duration: InitVar[float] = None
    output_valid: int = 0

    def __post_init__(self, calibration_duration: float = None):
        if calibration_duration is not None:
            self.output_valid = round(calibration_duration * GlobalConfig.frame_rate)

        assert 0. <= self.validation_split < 1., self.validation_split


def sort_by_snr(audio: Tensor, vad: Tensor, *other: Tensor):
    assert utils.is_binary(vad)

    x = audio.view(audio.shape[0], -1, GlobalConfig.win_size)
    N0, N1 = torch.count_nonzero(vad == 0.), torch.count_nonzero(vad == 1.)

    if not (N0 > 2 and N1 > 2):
        return (audio, vad, *other), None

    snr = torch.empty(audio.shape[0])
    for m in range(audio.shape[0]):
        S = torch.mean(torch.square(x[m, vad[m] == 1]))
        N = torch.mean(torch.square(x[m, vad[m] == 0]))
        snr[m] = 10 * math.log(S / (N + 1e-8))

    sorted_idx = torch.argsort(snr)
    # print(snr, sorted_idx)

    return (audio[sorted_idx, ...], vad[sorted_idx, ...], *[x[sorted_idx, ...] for x in other]), snr[sorted_idx]

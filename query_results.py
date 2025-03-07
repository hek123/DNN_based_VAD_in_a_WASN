import math

import matplotlib.pyplot as plt
import neptune
import numpy as np
from scipy.signal import lfilter, filtfilt
from pandas import DataFrame


def load_runs(*run_id: int):
    return [neptune.init_run(
        project="hektor/thesis",
        with_id=f"THES-{id_}",
        mode='read-only',
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5YTczYjZhYS1iZmJiLTRiNmMtOTE1Zi1jNGE1NmExODAwODcifQ=="
    ) for id_ in run_id]


def ema(x: np.ndarray, momentum: float, initialization: str = 'unbiased'):
    if momentum is None:
        return x
    assert isinstance(momentum, float) and 0. < momentum < 1., momentum
    assert len(x.shape) == 1, x.shape

    a = np.array([1., momentum - 1.])
    b = np.array([momentum, 0.])

    if initialization == 'unbiased':
        power = np.arange(1, 1 + x.shape[0])
        correction = np.power((1. - momentum), power)  # (num_filters, time)
        return lfilter(b, a, x) / (1. - correction)
    elif initialization == 'reflect':
        T60 = math.ceil(1. / momentum)
        ns = min(T60, x.shape[0])
        x = np.concatenate([np.flip(x[:ns]), x])
        x = lfilter(b, a, x)
        return x[ns:]
    elif initialization == 'zero':
        return lfilter(b, a, x)
    else:
        raise AssertionError(initialization)


def compare_norm():
    runs = DataFrame({
        'id': [128, 129, 130, 131, 132, 133, 138],
        'name': ['InstanceNorm', 'LayerNorm', 'TSNorm-c', 'TSNorm-nc', 'IN(aff)', 'IN(not-aff)', 'TSNorm-3s']
    })
    runs['run'] = load_runs(*runs['id'])

    for x in ('loss', 'error'):
        data = [run[f'train/{x}'].fetch_values()['value'] for run in runs['run']]

        ax: plt.Axes
        fig, ax = plt.subplots()
        for i in runs.index:
            if runs['name'][i] in ('InstanceNorm', 'LayerNorm', 'TSNorm-c', 'TSNorm-nc', 'TSNorm-3s'):
                ax.plot(ema(data[i], .1), label=runs['name'][i])

        ax.legend()
        ax.set_title(f'Train {x.capitalize()}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('error [%]' if x == 'error' else 'loss')

    plt.show()


def compare_enc(metric: str, momentum: float = .1, b=None, t=None):
    runs = DataFrame({
        'id': [i for i in range(147, 173) if i not in (154, 156)]
    })
    runs['run'] = load_runs(*runs['id'])
    stft = ema(load_runs(173)[0][metric].fetch_values()['value'], momentum)

    # Number of Layers
    data = [run[metric].fetch_values()['value'] for run in runs['run']]

    grouped_data = dict((name, set()) for name in ['lin', 'nlin', 'm1', 'm1.5', 'm2', 'l1', 'l2', 'l3', 'l9'])
    for i in runs.index:
        if runs['run'][i]['sweep/ConvEnc/linear'].fetch():
            grouped_data['lin'].add(i)
        else:
            grouped_data['nlin'].add(i)

        m = runs['run'][i]['sweep/ConvEnc/multiplier'].fetch()
        if m == 1.:
            grouped_data['m1'].add(i)
        elif m == 1.5:
            grouped_data['m1.5'].add(i)
        elif m == 2.:
            grouped_data['m2'].add(i)
        else:
            raise AssertionError(m)
        l = runs['run'][i]['sweep/ConvEnc/num_layers'].fetch()
        if l == 1:
            grouped_data['l1'].add(i)
        elif l == 2:
            grouped_data['l2'].add(i)
        elif l == 3:
            grouped_data['l3'].add(i)
        elif l == 9:
            grouped_data['l9'].add(i)
        else:
            raise AssertionError(l)

    # Global averages
    global_avg = dict()
    for k in grouped_data:
        global_avg[k] = ema(np.mean([data[i] for i in grouped_data[k]], axis=0), momentum)

    fig: plt.Figure
    axes: list[list[plt.Axes]]
    fig, axes = plt.subplots(2, 2)
    fig.suptitle('Global Averages')

    axes[0][0].plot(global_avg['lin'], label='linear')
    axes[0][0].plot(global_avg['nlin'], label='non-linear')
    axes[0][0].set_ylim(b, t)
    axes[0][0].legend()

    axes[0][1].plot(global_avg['m1'], label='m = 1')
    axes[0][1].plot(global_avg['m1.5'], label='m = 1.5')
    axes[0][1].plot(global_avg['m2'], label='m = 2')
    axes[0][1].set_ylim(b, t)
    axes[0][1].legend()

    axes[1][1].plot(global_avg['l1'], label='layers = 1')
    axes[1][1].plot(global_avg['l2'], label='layers = 2')
    axes[1][1].plot(global_avg['l3'], label='layers = 3')
    axes[1][1].plot(global_avg['l9'], label='layers = 9')
    axes[1][1].set_ylim(b, t)
    axes[1][1].legend()

    axes[1][0].plot(stft, label='stft')
    axes[1][0].legend()
    axes[1][0].set_ylim(b, t)

    # Only non-lin, and wo l1 and l9
    fine_avg = dict()
    for k in ['l2', 'l3', 'm1', 'm1.5', 'm2']:
        fine_avg[k] = ema(np.mean([data[i] for i in grouped_data[k]
                                   if i in grouped_data['nlin'] and
                                   (i not in grouped_data['l1'] and i not in grouped_data['l9'])],
                                  axis=0), momentum)

    axes: list[plt.Axes]
    fig, axes = plt.subplots(2, 1)
    fig.suptitle('Refined Averages')

    axes[0].plot(fine_avg['m1'], label='m = 1')
    axes[0].plot(fine_avg['m1.5'], label='m = 1.5')
    axes[0].plot(fine_avg['m2'], label='m = 2')
    axes[0].plot(stft, label='stft')
    axes[0].set_ylim(b, t)
    axes[0].legend()

    axes[1].plot(fine_avg['l2'], label='layers = 2')
    axes[1].plot(fine_avg['l3'], label='layers = 3')
    axes[1].plot(stft, label='stft')
    axes[1].set_ylim(b, t)
    axes[1].legend()

    plt.show()


if __name__ == '__main__':
    # compare_norm()
    compare_enc('val/error', None)

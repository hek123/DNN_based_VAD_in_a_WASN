import pyroomacoustics as pra
import numpy as np
from matplotlib import pyplot as plt
import soundfile


if __name__ == '__main__':
    # Create a room
    corners = np.array([[0, 0], [0, 3], [5, 3], [5, 1], [3, 1], [3, 0]]).T  # [x,y]
    room = pra.Room.from_corners(corners, fs=16_000,
                                 max_order=3, materials=pra.Material(.2, .15),
                                 ray_tracing=True, air_absorption=True)
    room.set_ray_tracing(receiver_radius=.5, n_rays=10_000, energy_thres=1e-5)

    # add source to 2D room
    signal, fs_source = soundfile.read("../data/train-clean-100/LibriSpeech/train-clean-100/19/198/19-198-0000.flac")
    assert fs_source == room.sample_rate, f"global frequency = {room.sample_rate}, but audio has fs = {fs_source}"

    room.add_source([1., 1.], signal=signal)

    # add a microphone array
    R = pra.linear_2D_array([2., 2.], 5, phi=0., d=.1)
    room.add_microphone_array(pra.MicrophoneArray(R, room.sample_rate))

    # compute image sources -> Now RIR is available
    room.image_source_model()

    fig, ax = room.plot()
    ax.set_xlim([-1, 6])
    ax.set_ylim([-1, 4])

    fig, ax = room.plot_rir()

    # simulate
    room.simulate()

    # plot micsigs
    plt.figure()
    plt.plot(room.mic_array.signals[1,:])

    plt.show()

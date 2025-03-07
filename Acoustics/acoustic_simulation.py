import os.path

import numpy as np
import pyroomacoustics as pra
import soundfile


def create_room(corners: np.ndarray, sources: list, microphones: dict,
                fs: int) -> pra.Room:
    # Create a room
    room = pra.Room.from_corners(corners, fs=fs,
                                 max_order=3, materials=pra.Material(.2, .15),
                                 ray_tracing=True, air_absorption=True)
    room.set_ray_tracing(receiver_radius=.5, n_rays=10_000, energy_thres=1e-5)

    # add sources to 2D room
    for source in sources:
        room.add_source(source)

    # add microphones
    for key, value in microphones.items():
        if key == "mic":
            for pos in value:
                room.add_microphone(pos, fs=fs)
        elif key == "LMA":
            for data in value:
                R = pra.linear_2D_array(**data)
                room.add_microphone_array(pra.MicrophoneArray(R, fs=fs))

    return room


def load_soundfile(path: str) -> (np.ndarray, int):
    return soundfile.read(path)


def load_LibriSpeech_folder(libriSpeech_folder: str, subject: int, trial: int, fs: int) -> np.ndarray:
    idx = 0
    file = libriSpeech_folder + f"/{subject}/{trial}/{subject}-{trial}-{format(idx, '04d')}.flac"

    out = np.zeros((0,), dtype="int16")
    while os.path.exists(file):
        data, Fs = load_soundfile(file)
        assert fs == Fs
        out = np.concatenate([out, data], 0)
        idx += 1
        file = libriSpeech_folder + f"/{subject}/{trial}/{subject}-{trial}-{format(idx, '04d')}.flac"
    else:
        print(file)

    return out


def getmicsigs(room: pra.Room, sources_sound: list[np.ndarray]) -> pra.Room:
    assert len(sources_sound) == room.n_sources, "incompatible number of sources"

    for idx, source in enumerate(room.sources):
        source.add_signal(sources_sound[idx])

    room.simulate()
    return room


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fs = 16_000

    corners = np.array([[0, 0], [0, 3], [5, 3], [5, 1], [3, 1], [3, 0]]).T  # [x,y]
    room = create_room(corners, [[1, 1]], {"LMA": [{"center": [2., 2.], "M": 5, "phi": 0., "d": .1}]}, fs=fs)

    source_data = load_LibriSpeech_folder("../train-clean-100/LibriSpeech/train-clean-100", 19, 198, fs)

    # room = getmicsigs(room, [source_data])

    # sounddevice.play(room.mic_array.signals[0, :], fs)
    # sounddevice.wait()

    # fig, ax = room.plot()
    # ax.set_xlim([-1, 6])
    # ax.set_ylim([-1, 4])

    # plt.show()

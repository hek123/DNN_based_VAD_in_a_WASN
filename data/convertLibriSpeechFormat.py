import os

import soundfile
import numpy as np
from tqdm import tqdm


class ConvertLSDataset:
    def __init__(self, ls_folder: str, new_folder: str, fs: int):
        self.ls_folder = ls_folder
        self.new_folder = new_folder
        self.fs = fs

        self.folders = []
        for s in os.listdir(self.ls_folder):
            self.folders += [(s, t, os.path.join(s, t)) for t in os.listdir(os.path.join(self.ls_folder, s))]

    def __getitem__(self, k):
        idx = 0
        s, t, folder = self.folders[k]
        folder = os.path.join(self.ls_folder, folder)
        file = os.path.join(folder, f"{s}-{t}-%04i.flac" % idx)
        assert os.path.exists(file), f"Could not find the file: {file}"

        info = soundfile.info(file)
        assert info.samplerate == self.fs

        out = np.empty((0,), dtype="float32")
        durations = []
        while os.path.exists(file):
            data, _ = soundfile.read(file, dtype="float32")
            out = np.concatenate([out, data], 0)
            durations.append(data.shape[-1])
            idx += 1
            file = os.path.join(folder, f"{s}-{t}-%04i.flac" % idx)

        return out, info, np.array(durations)

    def __len__(self):
        return len(self.folders)

    def _save_new_audio(self, k, data: np.ndarray, info: soundfile._SoundFileInfo, annotations: dict[str, np.ndarray]):
        s, t, folder = self.folders[k]
        new_folder = os.path.join(self.new_folder, folder)
        sound_file = os.path.join(new_folder, f"speech-{s}-{t}.wav")

        soundfile.write(sound_file, data, info.samplerate)

        text_file = os.path.join(new_folder, f"annotation-{s}-{t}")
        np.savez(text_file, **annotations)

    def process_dataset(self, preprocessing=None):
        if preprocessing is None:
            preprocessing = lambda x: x

        for k, (_, _, folder) in enumerate(tqdm(self.folders)):
            # new_s_folder = os.path.join(self.new_folder, subject)
            # if not os.path.exists(new_s_folder):
            #     os.mkdir(new_s_folder)
            new_folder = os.path.join(self.new_folder, folder)
            if not os.path.exists(new_folder):
                os.mkdir(new_folder)

            data, info, segments = self[k]
            data = preprocessing(data)
            self._save_new_audio(k, data, info, {"segment_durations": segments})


if __name__ == '__main__':
    from data.dataset import preprocess
    root = "data"

    ls_root = os.path.join(root, "train-clean-100", "LibriSpeech", "train-clean-100")
    new_root = os.path.join(root, "LibriSpeechConcat")

    fs = 16_000
    pp = ConvertLSDataset(ls_root, new_root, fs=fs)
    pp.process_dataset(preprocess(fs=fs))

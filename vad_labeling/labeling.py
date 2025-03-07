import math
from typing import Callable, Any
import os

import torch
from torch import Tensor
import numpy as np
from torch.utils.data import DataLoader

from config import Paths, GlobalConfig
from data.dataset import MyAudioDataset
from tqdm import tqdm


def timestamps2vad(input_type: str = 'array') -> Callable[[Any, int], torch.Tensor]:
    def list_of_dicts(timestamps: list[dict[str, int]], num_frames: int) -> torch.Tensor:
        vad = torch.zeros((num_frames,))
        for timestamp in timestamps:
            vad[timestamp['start']:timestamp['end']] = 1.
        return vad

    def array(timestamps: torch.Tensor | np.ndarray, num_frames: int) -> torch.Tensor:
        vad = torch.zeros((num_frames,))
        for timestamp in timestamps:
            vad[timestamp[0]:timestamp[1]] = 1.
        return vad

    if input_type == 'list_of_dicts':
        return list_of_dicts
    elif input_type == 'array':
        return array
    else:
        raise Exception(f"Invalid input type: '{input_type}', expected 'list_of_dicts' or 'array'")


def spp2timestamps(fs: int = GlobalConfig.sample_rate, win_size: int = GlobalConfig.win_size, threshold: float = 0.5,
                   min_speech_duration_ms: int = 250, max_speech_duration_s: float = float('inf'),
                   min_silence_duration_ms: int = 100, speech_pad_ms: int = 30):
    assert fs == 16_000, "Sampling rate is different from 16kHz!"
    assert win_size == 512, "Window size different from 512!"

    # Convert ms to number of frames
    frame_rate = fs / win_size
    min_speech_frames = round(frame_rate * min_speech_duration_ms / 1000)
    speech_pad_frames = round(frame_rate * speech_pad_ms / 1000)
    max_speech_frames = round(frame_rate * max_speech_duration_s) - 1 - 2 * speech_pad_frames \
        if math.isfinite(max_speech_duration_s) else max_speech_duration_s
    min_silence_frames = round(frame_rate * min_silence_duration_ms / 1000)
    min_silence_frames_at_max_speech = round(frame_rate * 98 / 1000)

    neg_threshold = threshold - 0.15

    def fn(speech_probs: Tensor) -> list[dict[str, int]]:
        assert len(speech_probs.shape) == 1, f"More than one dimension in audio!, {speech_probs.shape}"

        audio_length_frames = speech_probs.shape[0]

        triggered = False
        speeches = []
        current_speech = {}
        temp_end = 0  # to save potential segment end (and tolerate some silence)
        prev_end = next_start = 0  # to save potential segment limits in case of maximum segment size reached

        for i, spp in enumerate(speech_probs):
            if (spp >= threshold) and temp_end:
                temp_end = 0
                if next_start < prev_end:
                    next_start = i

            if (spp >= threshold) and not triggered:
                triggered = True
                current_speech['start'] = i
                continue

            if triggered and i - current_speech['start'] > max_speech_frames:
                if prev_end:
                    current_speech['end'] = prev_end
                    speeches.append(current_speech)
                    current_speech = {}
                    if next_start < prev_end:  # previously reached silence (< neg_thres) and is still not speech (< thres)
                        triggered = False
                    else:
                        current_speech['start'] = next_start
                    prev_end = next_start = temp_end = 0
                else:
                    current_speech['end'] = i
                    speeches.append(current_speech)
                    current_speech = {}
                    prev_end = next_start = temp_end = 0
                    triggered = False
                    continue

            if (spp < neg_threshold) and triggered:
                if not temp_end:
                    temp_end = i
                if (i - temp_end) > min_silence_frames_at_max_speech:  # condition to avoid cutting in very short silence
                    prev_end = temp_end
                if i - temp_end < min_silence_frames:
                    continue
                else:
                    current_speech['end'] = temp_end
                    if (current_speech['end'] - current_speech['start']) > min_speech_frames:
                        speeches.append(current_speech)
                    current_speech = {}
                    prev_end = next_start = temp_end = 0
                    triggered = False
                    continue

        # Final segment on the end
        if current_speech and (audio_length_frames - current_speech['start']) > min_speech_frames:
            current_speech['end'] = audio_length_frames
            speeches.append(current_speech)

        # # Apply padding
        # for i, speech in enumerate(speeches):
        #     if i == 0:
        #         speech['start'] = int(max(0, speech['start'] - speech_pad_frames))
        #     if i != len(speeches) - 1:
        #         silence_duration = speeches[i+1]['start'] - speech['end']
        #         if silence_duration < 2 * speech_pad_frames:
        #             speech['end'] += int(silence_duration // 2)
        #             speeches[i+1]['start'] = int(max(0, speeches[i+1]['start'] - silence_duration // 2))
        #         else:
        #             speech['end'] = int(min(audio_length_frames, speech['end'] + speech_pad_frames))
        #             speeches[i+1]['start'] = int(max(0, speeches[i+1]['start'] - speech_pad_frames))
        #     else:
        #         speech['end'] = int(min(audio_length_frames, speech['end'] + speech_pad_frames))

        # Apply padding
        if len(speeches):
            i = 0
            speeches[0]['start'] = int(max(0, speeches[0]['start'] - speech_pad_frames))
            while i < len(speeches) - 1:
                speech = speeches[i]
                silence_duration = speeches[i+1]['start'] - speech['end']
                if silence_duration < 2 * speech_pad_frames:  # Padding merges the speeches
                    speech['end'] = speeches[i+1]['end']
                    del speeches[i+1]
                else:
                    speech['end'] = speech['end'] + speech_pad_frames
                    speeches[i+1]['start'] = speeches[i+1]['start'] - speech_pad_frames
                    i += 1

            speeches[-1]['end'] = min(audio_length_frames, speeches[-1]['end'] + speech_pad_frames)

        return speeches

        # arr = torch.empty((len(speeches), 2), dtype=torch.int32)
        # for t, timestamp in enumerate(speeches):
        #     arr[t, 0] = timestamp['start']
        #     arr[t, 1] = timestamp['end']
        #
        # return arr

    return fn


def spp2vad(**kwargs) -> Callable[[Tensor], Tensor]:
    spp2ts = spp2timestamps(**kwargs)
    ts2vad = timestamps2vad('list_of_dicts')

    def fn(spp: Tensor) -> Tensor:
        if len(spp.shape) == 2:
            return torch.stack([ts2vad(spp2ts(x), spp.shape[1]) for x in spp])
        elif len(spp.shape) == 1:
            return ts2vad(spp2ts(spp), spp.shape[0])
        raise AssertionError(f"{spp.shape=}")

    return fn


def timestamps_as_array(timestamps: list[dict[str, int]]) -> Tensor:
    arr = torch.empty((len(timestamps), 2), dtype=torch.int32)
    for t, timestamp in enumerate(timestamps):
        arr[t, 0] = timestamp['start']
        arr[t, 1] = timestamp['end']
    return arr


# def spp2vad_fast(fs: int = Config.sample_rate, win_size: int = Config.win_size, thresh_speech: float = 0.67, thresh_noise=.33,
#                  min_speech_duration_ms: int = 250, max_speech_duration_s: float = float('inf'),
#                  min_silence_duration_ms: int = 100, speech_pad_ms: int = 30) -> Callable[[Tensor], Tensor]:
#     assert fs == 16_000, "Sampling rate is different from 16kHz!"
#     assert win_size == 512, "Window size different from 512!"
#
#     # Convert ms to number of frames
#     frame_rate = fs / win_size
#     min_speech_frames = round(frame_rate * min_speech_duration_ms / 1000)
#     speech_pad_frames = round(frame_rate * speech_pad_ms / 1000)
#     max_speech_frames = round(frame_rate * max_speech_duration_s) - 1 - 2 * speech_pad_frames \
#         if math.isfinite(max_speech_duration_s) else max_speech_duration_s
#     min_silence_frames = round(frame_rate * min_silence_duration_ms / 1000)
#     min_silence_frames_at_max_speech = round(frame_rate * 98 / 1000)
#
#     def fn(speech_probs: Tensor) -> Tensor:
#         assert len(speech_probs.shape) == 1, "More than one dimension in audio!"
#
#         audio_length_frames = speech_probs.shape[0]
#
#         # Check probs against threshold
#         speech = speech_probs >= thresh_speech
#         noise = speech_probs < thresh_noise
#         unknown = thresh_noise <= speech_probs < thresh_speech
#
#         vad = torch.zeros_like(speech_probs)
#         vad[speech] = 1
#         vad[unknown] = .5
#
#         return vad
#
#     return fn


def save_labels(dataset: MyAudioDataset | DataLoader, output_folder: str, get_label: Callable[[Tensor], Tensor]):

    output_folder = os.path.join(Paths.vad_spp, output_folder)
    assert not os.path.exists(output_folder), \
        f"The provided ouput folder already exists, please provide another path or remove the existing folder first"
    os.mkdir(output_folder)

    for audio, _, annotations in tqdm(dataset, desc="Computing VAD/SPP"):
        if isinstance(dataset, DataLoader):
            assert len(audio.shape) == 2 and audio.shape[0] == 1
            audio = audio[0, :]
        y = get_label(audio)

        file = os.path.join(output_folder, annotations["filename"])
        np.save(file, y)


if __name__ == '__main__':
    from data.dataset import VCTK
    from vad_labeling.silero import SileroVAD

    dataset = MyAudioDataset(data=VCTK().file_iterator(), random_offset=False, last_frame='pad')
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=default_collate_fn)

    save_labels(
        dataset=dataset,
        output_folder=os.path.join("VCTK", "silero_512_spp"),
        get_label=SileroVAD()
    )

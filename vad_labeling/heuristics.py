import math

import numpy as np
import torch


def prob_to_vad(speech_probs: torch.Tensor, audio_length_samples: int, fs: int, win_size: int,
                threshold: float = 0.5, min_speech_duration_ms: int = 250, max_speech_duration_s: float = float('inf'),
                min_silence_duration_ms: int = 100, speech_pad_ms: int = 30):

    """
    Returns a function that takes the speech probabilities at each time instance and converts it to a VAD

    Parameters
    ----------
    speech_probs: torch.Tensor, one dimensional
        One dimensional float torch.Tensor, other types are casted to torch if possible

    audio_length_samples: The length of the audio signal

    threshold: float (default - 0.5)
        Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
        It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.

    fs: Sampling frequency
    win_size: Frame size

    min_speech_duration_ms: int (default - 250 milliseconds)
        Final speech chunks shorter min_speech_duration_ms are thrown out

    max_speech_duration_s: int (default -  inf)
        Maximum duration of speech chunks in seconds
        Chunks longer than max_speech_duration_s will be split at the timestamp of the last silence that lasts more than 100ms (if any), to prevent agressive cutting.
        Otherwise, they will be split aggressively just before max_speech_duration_s.

    min_silence_duration_ms: int (default - 100 milliseconds)
        In the end of each speech chunk wait for min_silence_duration_ms before separating it

    speech_pad_ms: int (default - 30 milliseconds)
        Final speech chunks are padded by speech_pad_ms each side

    Returns
    ----------
    speeches: list of dicts
        list containing ends and beginnings of speech chunks (samples or seconds based on return_seconds)
    """

    # Check arguments
    if not torch.is_tensor(speech_probs):
        try:
            speech_probs = torch.Tensor(speech_probs)
        except:
            raise TypeError("Audio cannot be casted to tensor. Cast it manually")

    if len(speech_probs.shape) > 1:
        for i in range(len(speech_probs.shape)):  # trying to squeeze empty dimensions
            speech_probs = speech_probs.squeeze(0)
        if len(speech_probs.shape) > 1:
            raise ValueError("More than one dimension in audio. Are you trying to process audio with 2 channels?")

    assert fs == 16_000, "Sampling rate is different from 16kHz!"
    assert win_size == 512, "Window size different from 512!"

    audio_length_frames = audio_length_samples // win_size

    # Convert ms to number of frames
    frame_rate = fs / win_size
    min_speech_frames = round(frame_rate * min_speech_duration_ms / 1000)
    speech_pad_frames = round(frame_rate * speech_pad_ms / 1000)
    max_speech_frames = round(frame_rate * max_speech_duration_s) - 1 - 2 * speech_pad_frames \
        if math.isfinite(max_speech_duration_s) else max_speech_duration_s
    min_silence_frames = round(frame_rate * min_silence_duration_ms / 1000)
    min_silence_frames_at_max_speech = round(frame_rate * 98 / 1000)

    # Check probs against threshold
    vad_thresh = speech_probs >= threshold

    triggered = False
    speeches = []
    current_speech = {}
    temp_end = 0  # to save potential segment end (and tolerate some silence)
    prev_end = next_start = 0  # to save potential segment limits in case of maximum segment size reached

    for i, vad in enumerate(vad_thresh):
        if vad and temp_end:
            temp_end = 0
            if next_start < prev_end:
               next_start = i

        if vad and not triggered:
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

        if (not vad) and triggered:
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

    for i, speech in enumerate(speeches):
        if i == 0:
            speech['start'] = int(max(0, speech['start'] - speech_pad_frames))
        if i != len(speeches) - 1:
            silence_duration = speeches[i+1]['start'] - speech['end']
            if silence_duration < 2 * speech_pad_frames:
                speech['end'] += int(silence_duration // 2)
                speeches[i+1]['start'] = int(max(0, speeches[i+1]['start'] - silence_duration // 2))
            else:
                speech['end'] = int(min(audio_length_frames, speech['end'] + speech_pad_frames))
                speeches[i+1]['start'] = int(max(0, speeches[i+1]['start'] - speech_pad_frames))
        else:
            speech['end'] = int(min(audio_length_frames, speech['end'] + speech_pad_frames))

    return speeches


def timestamps2vad(timestamps: list[dict[str, int]] | np.ndarray, audio_len: int, win_size: int) -> torch.Tensor:
    vad = torch.zeros((audio_len // win_size,))
    if isinstance(timestamps, list):
        for timestamp in timestamps:
            vad[timestamp['start']:timestamp['end']] = 1.
        return vad
    elif isinstance(timestamps, np.ndarray):
        for timestamp in timestamps:
            vad[timestamp[0]:timestamp[1]] = 1.
        return vad
    else:
        raise Exception(f"Invalid input type: '{type(timestamps)}', expected 'list[dict[str, int]]' or 'np.ndarray'")

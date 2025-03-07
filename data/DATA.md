# LibriSpeech
___

## General

* Duration of audio tracks: 30s-1500s, mean=600s
* Some samples have strong DC value -> zero-mean
* Strong pollution by 50Hz tone & harmonics -> notch filters?

## LibriSpeechConcat(preprocessing)

1) Concatenate all fragments
2) Remove 60Hz, 120Hz, 180Hz and 240Hz tones: Notch filter (quality factor Q=30)
3) High pass filter to remove DC: 2nd order Butterworth filter, cutoff at 20Hz

## Labeling

### vad
old - to be removed

### silero_vad_512

Using silero_vad with a window of 512 samples on the raw audio (so without preprocessing)

### silero_vad_512_preproc & silero_vad_512_timestamp

Silero_vad with window of 512 samples on the preprocessed audio, only minor differences.

* Class unbalance: 85% active
* SNR (after preprocessing): 7.1dB - 29.7dB, mean=17.5dB
  * This is not great
* SNR below 10dB after preprocessing: 38, 39 & 410
* \# samples with unbalance > 90%: 163 -> **Data(aka VAD labels) is not OK !!!**
* \# samples with less than 1 switch in 10sec: 64 **Not very probable -> classify these as wrongly labeld**


### VarHMM

A VariationalGaussianHMM with two states, is trained on every sample separately and then used to infer the labels.
The VAD is then refined using the same heuristics as the silero_vad.

The HMM takes 12 MFC Coefficients as input (computed on a FFT with nfft=512)

* Class unbalance: 85% active (slightly lower than silero-vad)
* SNR (after preprocessing): from -0.48dB to 32.2dB, mean=18.1dB
  * This worse than Silero-VAD, is this a problem?
* \# samples with SNR below 10dB: 53
  * Discard all these?
* \# samples with unbalance > 90%: 136 -> Maybe this is just correct
* \# samples with less than 1 switch in 10sec: 0 -> Much better than Silero-VAD

---

- 90: File 198-126831; fails on all VAD methods (silero fails to detect silences, and bayesian methods cut off lots of speech)
- 38 & 39: Low SNR, but VAD seems to be correct


# VCTK-0.92
___

## General

- Unbalance (using thresholded Silero_VAD): 56.03%
- SNR: (thresholded Silero_VAD): min=-2.26dB, mean=22.42dB, max=36.12dB
- SNR < 10: 

## Preprocessed

Preprocessing steps:
1) Resampled (from 48kHz) to 16kHz
2) Utterances concatenated
3) High-pass filter at 50Hz
4) Labeled using silero-vad

Results:
- SNR (with original padding):
  - mic1: min=18.6dB, mean=30.6dB, max=39.4dB
  - mic2: min=19.23dB, mean=33.76dB, max=47.21dB

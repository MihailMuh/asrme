import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import BASE_PATH
from .configs import *

RESAMPLING_ENGINE = 'soxr'


def load_audio(audio_signal, sr=16000, return_duration=False):
    audio_duration = len(audio_signal) / sr

    if return_duration:
        return audio_signal, audio_duration
    else:
        return audio_signal


def audio_batch_generator(audio_files):
    return audio_files


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """

    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


class TorchSTFT(nn.Module):
    def __init__(self, n_fft, hop_length):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

        window = torch.hann_window(n_fft)
        self.register_buffer("window", window)

    def forward(self, x):
        return torch.stft(x, self.n_fft, self.hop_length, window=self.window, return_complex=True)


class LogMelSpectogram(nn.Module):
    def __init__(self,
                 n_mels=N_MELS,
                 n_fft=N_FFT,
                 hop_length=HOP_LENGTH,
                 padding=0):
        super().__init__()

        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.padding = padding

        mel_filters = np.load(os.path.join(BASE_PATH, "assets/mel_filters.npz"))
        mel_filters = torch.from_numpy(mel_filters[f"mel_{n_mels}"])
        self.register_buffer("mel_filters", mel_filters)

        self.stft = TorchSTFT(n_fft, hop_length)

    def get_seq_len(self, seq_len):
        seq_len = torch.floor(seq_len / self.hop_length)
        return seq_len.to(dtype=torch.long)

    @torch.no_grad()
    def forward(self, x, seq_len):
        seq_len = self.get_seq_len(seq_len.float())

        if self.padding > 0:
            x = F.pad(x, (0, self.padding))

        x = self.stft(x)

        x = x[..., :-1].abs() ** 2
        x = self.mel_filters @ x  # mels

        x = torch.clamp(x, min=1e-10).log10()  # log_mels
        x = torch.maximum(x, torch.amax(x, dim=(1, 2), keepdims=True) - 8.0)  # clip
        x = (x + 4.0) / 4.0  # scale

        return x, seq_len

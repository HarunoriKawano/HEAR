import torch
from torch import nn
from torchaudio.transforms import MelSpectrogram, Spectrogram, Resample

from src.config import AcousticModelConfig, TokenizerConfig, STATIC_CONFIG


class Preprocessor(nn.Module):
    def __init__(self, config: AcousticModelConfig | TokenizerConfig):
        super().__init__()
        self.sampling_rate = config.spectrogram_sampling_rate
        self.spectrogram = Spectrogram(
            n_fft=config.spectrogram_n_fft, win_length=int(STATIC_CONFIG.win_seconds * config.spectrogram_sampling_rate),
            hop_length=int(STATIC_CONFIG.hop_seconds * config.spectrogram_sampling_rate),
            center=True, window_fn=torch.hann_window, power=2
        )
        self.mel_spectrogram = MelSpectrogram(
            n_fft=STATIC_CONFIG.n_fft, sample_rate=STATIC_CONFIG.sampling_rate, n_mels=STATIC_CONFIG.n_mel, win_length=int(STATIC_CONFIG.win_seconds * STATIC_CONFIG.sampling_rate),
            hop_length=int(STATIC_CONFIG.hop_seconds * STATIC_CONFIG.sampling_rate), center=True, power=2, window_fn=torch.hann_window
        )
        if config.spectrogram_sampling_rate == STATIC_CONFIG.sampling_rate:
            self.resampler = None
        else:
            self.resampler = Resample(config.spectrogram_sampling_rate, STATIC_CONFIG.sampling_rate)

        self.eps = STATIC_CONFIG.eps

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor, input_lengths: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = inputs - inputs.mean(dim=-1, keepdim=True)
        inputs = inputs.mean(dim=1, keepdim=True)

        power_spectrum = self.spectrogram(inputs)
        log_power_spectrum = torch.log(power_spectrum + self.eps)
        normed_log_power_spectrum = self.instance_wise_norm(log_power_spectrum)

        if self.resampler:
            inputs = self.resampler(inputs)
        mel_spectrogram = self.mel_spectrogram(inputs)
        log_mel_spectrogram = torch.log(mel_spectrogram + self.eps)
        normed_log_mel_spectrogram = self.instance_wise_norm(log_mel_spectrogram)
        input_lengths = self.transform_input_lengths(input_lengths)

        return normed_log_mel_spectrogram[:, 0], input_lengths, normed_log_power_spectrum[:, 0]

    def instance_wise_norm(self, spectrogram):
        mean = spectrogram.mean(dim=[-2, -1], keepdim=True)
        std = spectrogram.std(dim=[-2, -1], unbiased=False, keepdim=True)
        return (spectrogram - mean) / (std + self.eps)

    def transform_input_lengths(self, input_lengths):
        input_lengths = torch.floor((input_lengths - int(STATIC_CONFIG.hop_seconds * self.sampling_rate)) / int(STATIC_CONFIG.hop_seconds * self.sampling_rate) + 1) + 1
        return input_lengths

import torch
from torch import nn

from src.config import AcousticModelConfig, TokenizerConfig, DecoderConfig, STATIC_CONFIG


class FeatureExtractor(nn.Module):
    """Convolutional front-end that downsamples the mel spectrogram by a factor of 2."""

    def __init__(self, config: AcousticModelConfig | TokenizerConfig | DecoderConfig):
        super().__init__()
        self.conv = nn.Conv1d(STATIC_CONFIG.n_mel, config.hidden_size, 2, 2)
        self.relu = nn.ReLU()

    def forward(
        self,
        log_mel_spectrogram: torch.Tensor,
        input_lengths: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.LongTensor]:
        """
        Args:
            log_mel_spectrogram: (batch, n_mel, time)
            input_lengths: Valid frame count per sample, shape (batch,).
        Returns:
            hidden_states: (batch, time // 2, hidden_size)
            input_lengths: Lengths halved to match the downsampled time axis.
        """
        hidden_states = self.conv(log_mel_spectrogram)
        hidden_states = self.relu(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        input_lengths = self.transform_input_lengths(input_lengths)

        return hidden_states, input_lengths

    @staticmethod
    def transform_input_lengths(input_lengths):
        input_lengths //= 2
        return input_lengths

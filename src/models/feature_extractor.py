import torch
from torch import nn

from src.config import AcousticModelConfig, TokenizerConfig, DecoderConfig, STATIC_CONFIG


class FeatureExtractor(nn.Module):
    def __init__(self, config: AcousticModelConfig | TokenizerConfig | DecoderConfig):
        super().__init__()
        self.conv = nn.Conv1d(STATIC_CONFIG.n_mel, config.hidden_size, 2, 2)
        self.relu = nn.ReLU()

    def forward(self, log_mel_spectrogram: torch.Tensor, input_lengths: torch.LongTensor) -> tuple[torch.Tensor, torch.LongTensor]:
        hidden_states = self.conv(log_mel_spectrogram)
        hidden_states = self.relu(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        input_lengths = self.transform_input_lengths(input_lengths)

        return hidden_states, input_lengths

    @staticmethod
    def transform_input_lengths(input_lengths):
        input_lengths //= 2
        return input_lengths

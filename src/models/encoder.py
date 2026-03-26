import torch
from torch import nn
import torch.nn.functional as F

from src.models.contextual_encoder import ContextualEncoder
from src.models.feature_extractor import FeatureExtractor
from src.config import AcousticModelConfig, TokenizerConfig, DecoderConfig


class Encoder(nn.Module):
    """Base encoder composed of a convolutional feature extractor and a transformer contextual encoder."""

    def __init__(self, config: AcousticModelConfig | TokenizerConfig | DecoderConfig):
        super().__init__()
        self.feature_extractor = FeatureExtractor(config)
        self.contextual_encoder = ContextualEncoder(config)

    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.LongTensor]:
        """
        Args:
            inputs: Log-mel spectrogram of shape (batch, n_mel, time).
            input_lengths: Valid frame count per sample, shape (batch,).
        Returns:
            hidden_states: Encoded features of shape (batch, time', hidden_size).
            input_lengths: Updated lengths after feature extractor subsampling.
        """
        hidden_states, input_lengths = self.feature_extractor(inputs, input_lengths)
        hidden_states = self.contextual_encoder(hidden_states, input_lengths)

        return hidden_states, input_lengths

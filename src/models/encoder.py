import torch
from torch import nn
import torch.nn.functional as F

from src.models.contextual_encoder import ContextualEncoder
from src.models.feature_extractor import FeatureExtractor
from src.config import AcousticModelConfig, TokenizerConfig, DecoderConfig


class Encoder(nn.Module):
    def __init__(self, config: AcousticModelConfig | TokenizerConfig | DecoderConfig):
        super().__init__()
        self.feature_extractor = FeatureExtractor(config)
        self.contextual_encoder = ContextualEncoder(config)

    def forward(self, inputs, input_lengths):
        hidden_states, input_lengths = self.feature_extractor(inputs, input_lengths)
        hidden_states = self.contextual_encoder(hidden_states, input_lengths)

        return hidden_states, input_lengths

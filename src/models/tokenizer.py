from torch import nn

from src.config import TokenizerConfig
from src.models.encoder import Encoder


class Tokenizer(nn.Module):
    """Encodes a mel spectrogram and projects it to codebook logits for discrete tokenization."""

    def __init__(self, config: TokenizerConfig):
        super().__init__()
        self.encoder = Encoder(config)
        self.linear = nn.Linear(config.hidden_size, config.num_codebooks)

    def forward(self, inputs, input_lengths):
        """
        Args:
            inputs: Log-mel spectrogram of shape (batch, n_mel, time).
            input_lengths: Valid frame count per sample, shape (batch,).
        Returns:
            out: Codebook logits of shape (batch, time', num_codebooks).
            input_lengths: Updated lengths after feature extractor subsampling.
        """
        hidden_states, input_lengths = self.encoder(inputs, input_lengths)
        out = self.linear(hidden_states)

        return out, input_lengths

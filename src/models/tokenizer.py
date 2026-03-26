from torch import nn

from src.config import TokenizerConfig
from src.models.encoder import Encoder


class Tokenizer(nn.Module):
    def __init__(self, config: TokenizerConfig):
        super().__init__()
        self.encoder = Encoder(config)
        self.linear = nn.Linear(config.hidden_size, config.num_codebooks)

    def forward(self, inputs, input_lengths):
        hidden_states, input_lengths = self.encoder(inputs, input_lengths)
        out = self.linear(hidden_states)

        return out, input_lengths

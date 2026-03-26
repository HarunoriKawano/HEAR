import torch
from torch import nn

from src.utils import make_attention_mask
from src.config import ClassificationDecoderConfig

class Decoder(nn.Module):
    def __init__(self, decoder_config: ClassificationDecoderConfig, input_hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_hidden_size*3, decoder_config.hidden_size)
        self.batch_norm = nn.BatchNorm1d(decoder_config.hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(decoder_config.dropout_rate)
        self.out = nn.Linear(decoder_config.hidden_size, decoder_config.num_classes)

    def forward(self, hidden_states, input_lengths):
        mask = make_attention_mask(hidden_states, input_lengths).unsqueeze(-1)
        len_clamped = input_lengths.view(-1, 1).clamp(min=1)
        max_pool = hidden_states.masked_fill(~mask, -float('inf')).max(dim=1)[0]
        mean_pool = hidden_states.masked_fill(~mask, 0.0).sum(dim=1) / len_clamped
        var = ((hidden_states - mean_pool.unsqueeze(1))**2).masked_fill(~mask, 0.0).sum(dim=1) / (len_clamped - 1).clamp(min=1e-5)
        std_pool = torch.sqrt(var + 1e-8)
        hidden_states = torch.cat([mean_pool, max_pool, std_pool], dim=1)

        hidden_states = self.linear(hidden_states)
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        out = self.out(hidden_states)

        return out

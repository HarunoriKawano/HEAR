import torch
from torch import nn

from src.config import TaskModelConfig
from src.utils import make_attention_mask


class TaskModel(nn.Module):
    def __init__(self, config: TaskModelConfig):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_length, config.hidden_size))
        self.transformer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size, dim_feedforward=config.intermediate_size, nhead=config.num_attention_heads,
            batch_first=True, dropout=config.dropout_rate
        )

    def forward(self, hidden_states, input_lengths):
        hidden_states = hidden_states + self.pos_embed[:, :hidden_states.size(1), :]
        mask = make_attention_mask(hidden_states, input_lengths)
        hidden_states = self.transformer(hidden_states, src_key_padding_mask=~mask)

        return hidden_states

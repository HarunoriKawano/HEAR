import torch
from torch import nn

from src.models.transformer import TransformerWithRelativePosition
from src.config import DecoderConfig, STATIC_CONFIG
from src.utils import make_attention_mask


class Decoder(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.layers = nn.ModuleList([TransformerWithRelativePosition(
            config.hidden_size, config.intermediate_size,
            config.num_attention_heads, config.dropout_probability, max_length=config.max_length
        ) for _ in range(config.num_transformer_layers)])
        self.out_layer = OutLayer(config)

    def forward(self, hidden_states: torch.Tensor, input_lengths: torch.Tensor):
        attention_mask = make_attention_mask(hidden_states, input_lengths)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask
            )

        predicted_spectrogram, knowledge_prediction = self.out_layer(hidden_states)

        return predicted_spectrogram, knowledge_prediction, input_lengths * 2


class OutLayer(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.out_linear = nn.Linear(config.hidden_size, STATIC_CONFIG.n_mel * 2)
        self.out_linear2 = nn.Linear(config.hidden_size, config.knowledge_hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        predicted_spectrogram = self.out_linear(hidden_states)
        predicted_spectrogram = predicted_spectrogram.view(hidden_states.size(0), hidden_states.size(1) * 2, -1)

        knowledge_prediction = self.out_linear2(hidden_states)

        return predicted_spectrogram, knowledge_prediction

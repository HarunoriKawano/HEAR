from torch import nn

from src.config import AcousticModelConfig, TokenizerConfig, DecoderConfig
from src.models.transformer import TransformerWithRelativePosition
from src.utils import make_attention_mask


class ContextualEncoder(nn.Module):
    """Stack of transformer layers that models contextual dependencies across the sequence."""

    def __init__(self, config: AcousticModelConfig | TokenizerConfig | DecoderConfig):
        super().__init__()
        self.layers = nn.ModuleList([TransformerWithRelativePosition(
            config.hidden_size, config.intermediate_size,
            config.num_attention_heads, config.dropout_probability, max_length=config.max_length
        ) for _ in range(config.num_transformer_layers)])

    def forward(self, hidden_states, input_lengths):
        """
        Args:
            hidden_states: (batch, time, hidden_size)
            input_lengths: Valid frame count per sample, shape (batch,).
        Returns:
            hidden_states: (batch, time, hidden_size)
        """
        attention_mask = make_attention_mask(hidden_states, input_lengths)

        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask
            )

        return hidden_states

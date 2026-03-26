from typing import Optional
import math

import torch
from torch import nn
from torch.nn.init import xavier_uniform_

from src.models.positional_encoder import RelativePositionEncoder


class TransformerWithRelativePosition(nn.Module):
    """Transformer encoder layer with pre-norm and relative position attention."""

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            num_attention_heads: int,
            dropout_probability: float,
            max_length: int
    ):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.self_attn = MultiHeadSelfAttentionWithRelativePosition(
            hidden_size, num_attention_heads, dropout_probability, max_length
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, intermediate_size, dropout_probability)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, time, hidden_size)
            attention_mask: (batch, time), True for valid positions.
        Returns:
            hidden_states: (batch, time, hidden_size)
        """
        residual = hidden_states.clone()
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states.clone()
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MultiHeadSelfAttentionWithRelativePosition(nn.Module):
    """Multi-head self-attention using Shaw-style relative position encodings.

    Attention scores are computed as the sum of content-based (QK^T) and
    position-based (Q * R^T) terms, following the formulation in
    "Self-Attention with Relative Position Representations" (Shaw et al., 2018).
    """

    def __init__(
            self,
            hidden_size: int,
            num_attention_heads: int,
            dropout_probability: float,
            max_length: int,
    ):
        super().__init__()
        self.head_size = hidden_size // num_attention_heads
        self.num_heads = num_attention_heads

        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(p=dropout_probability)
        self.linear_out = nn.Linear(hidden_size, hidden_size)

        self.relative_position_k = RelativePositionEncoder(self.head_size, max_length)
        self.relative_position_v = RelativePositionEncoder(self.head_size, max_length)
        self.query_bias1 = nn.Parameter(torch.zeros(self.num_heads, self.head_size))
        self.query_bias2 = nn.Parameter(torch.zeros(self.num_heads, self.head_size))

        xavier_uniform_(self.linear_q.weight)
        xavier_uniform_(self.linear_k.weight)
        xavier_uniform_(self.linear_v.weight)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, time, hidden_size)
            attention_mask: (batch, time), True for valid positions.
        Returns:
            out: (batch, time, hidden_size)
        """
        batch_size, length, hidden_size = hidden_states.size()

        query = self.linear_q(hidden_states).view(batch_size, -1, self.num_heads, self.head_size)
        query1 = query + self.query_bias1
        query2 = query + self.query_bias2
        key = self.linear_k(hidden_states).view(batch_size, -1, self.num_heads, self.head_size)
        value = self.linear_v(hidden_states).view(batch_size, -1, self.num_heads, self.head_size)

        query1 = query1.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attention1 = torch.matmul(query1, key.transpose(-1, -2))

        query2 = query2.transpose(0, 1).contiguous().view(length, -1, self.head_size)
        position_embeddings_k = self.relative_position_k(hidden_states)
        attention2 = torch.matmul(query2, position_embeddings_k.transpose(1, 2)).transpose(0, 1)
        attention2 = attention2.contiguous().view(batch_size, self.num_heads, length, length)
        attention = (attention1 + attention2) / math.sqrt(self.head_size)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attention = attention.masked_fill(attention_mask == 0, torch.finfo(attention.dtype).min)

        probs = torch.softmax(attention, dim=-1)
        probs = self.dropout(probs)
        weight1 = torch.matmul(probs, value)

        position_embeddings_v = self.relative_position_v(hidden_states)
        weight2 = probs.permute(2, 0, 1, 3).contiguous().view(length, -1, length)
        weight2 = torch.matmul(weight2, position_embeddings_v)
        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.num_heads, -1, self.head_size)

        hidden_states = weight1 + weight2

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_size)
        out = self.linear_out(hidden_states)

        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network with SiLU activation and pre-norm."""

    def __init__(self, hidden_size: int, intermediate_size: int, dropout_probability: float):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.intermediate_dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.SiLU()
        self.intermediate_dropout = nn.Dropout(p=dropout_probability)

        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.output_dropout = nn.Dropout(p=dropout_probability)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.intermediate_dropout(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states

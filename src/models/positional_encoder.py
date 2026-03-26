import torch
from torch import nn


class RelativePositionEncoder(nn.Module):
    """Learnable relative position embeddings indexed by pairwise token distance.

    Supports distances in the range [-max_length, max_length), stored as an
    embedding table of size 2 * max_length.
    """

    def __init__(self, hidden_size: int, max_length: int):
        super().__init__()
        self.max_length = max_length
        self.positional_embedding = nn.Embedding(self.max_length * 2, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, time, hidden_size) — used only to determine sequence length and device.
        Returns:
            position_embeddings: (time, time, head_size)
        """
        range_tensor = torch.arange(hidden_states.size(1), device=hidden_states.device)
        distance_mat = range_tensor[None, :] - range_tensor[:, None] + self.max_length

        position_embeddings = self.positional_embedding(distance_mat)
        return position_embeddings

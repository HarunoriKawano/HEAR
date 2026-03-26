import torch
import torch.nn.functional as F

from src.models.encoder import Encoder
from src.config import AcousticModelConfig


class AcousticModel(Encoder):
    """Acoustic encoder with sliding-window processing for long sequences.

    For sequences longer than max_length, the input is split into overlapping
    patches, encoded independently, and merged back via weighted overlap-add.
    Short sequences are passed through the contextual encoder directly.
    """

    def __init__(self, config: AcousticModelConfig) -> None:
        super().__init__(config)
        self.max_length = config.max_length
        self.overlap = config.overlap
        self.stride = self.max_length - self.overlap

        self.window = torch.ones(self.max_length)
        ramp = torch.linspace(0, 1, self.overlap)
        curve = 0.5 - 0.5 * torch.cos(torch.pi * ramp)
        curve = torch.clamp(curve, min=1e-3)
        self.window[:self.overlap] = curve
        self.window[-self.overlap:] = curve.flip(0)

    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.LongTensor]:
        """Encode a batch of log-mel spectrograms.

        Args:
            inputs: Log-mel spectrogram of shape (batch, n_mel, time).
            input_lengths: Number of valid frames per sample, shape (batch,).
        Returns:
            hidden_states: Encoded features of shape (batch, time', hidden_size).
            input_lengths: Updated lengths after feature extractor subsampling.
        """
        hidden_states, input_lengths = self.feature_extractor(inputs, input_lengths)
        is_long = hidden_states.size(1) > self.max_length

        if is_long:
            hidden_states, encoder_input_lengths, meta = self._unfold(hidden_states, input_lengths)
            hidden_states = self.contextual_encoder(hidden_states, encoder_input_lengths)
            hidden_states = self._fold(hidden_states, meta)
        else:
            hidden_states = self.contextual_encoder(hidden_states, input_lengths)

        return hidden_states, input_lengths

    def _unfold(
        self,
        hidden_states: torch.Tensor,
        input_lengths: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.LongTensor, dict]:
        """Split a long sequence into overlapping patches.

        Args:
            hidden_states: Feature tensor of shape (batch, time, hidden_size).
            input_lengths: Valid length per sample, shape (batch,).
        Returns:
            valid_patches: Packed valid patches, shape (num_valid, max_length, hidden_size).
            valid_patch_lengths: Valid frame count within each patch, shape (num_valid,).
            meta: Reconstruction metadata used by _fold.
        """
        batch_size, length, hidden_size = hidden_states.shape

        pad_len = (self.stride - (length - self.max_length) % self.stride) % self.stride
        if pad_len > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))
            length = hidden_states.shape[1]

        num_patches = (length - self.max_length) // self.stride + 1
        patches = hidden_states.unfold(1, self.max_length, self.stride).transpose(2, 3)

        patch_starts = torch.arange(0, num_patches * self.stride, self.stride, device=hidden_states.device)

        is_start_valid = patch_starts.unsqueeze(0) < input_lengths.unsqueeze(1)

        previous_patch_ends = patch_starts - self.stride + self.max_length
        is_needed = (patch_starts == 0).unsqueeze(0) | (input_lengths.unsqueeze(1) > previous_patch_ends.unsqueeze(0))

        valid_mask = is_start_valid & is_needed

        patch_lengths = torch.clamp(input_lengths.unsqueeze(1) - patch_starts.unsqueeze(0), min=0, max=self.max_length)

        valid_patches = patches[valid_mask]
        valid_patch_lengths = patch_lengths[valid_mask]

        meta = {
            'B': batch_size,
            'T': length,
            'H': hidden_size,
            'original_T': length - pad_len,
            'valid_mask': valid_mask,
        }

        return valid_patches, valid_patch_lengths, meta

    def _fold(self, valid_patches: torch.Tensor, meta: dict) -> torch.Tensor:
        """Reconstruct a sequence from encoded overlapping patches via weighted overlap-add.

        Args:
            valid_patches: Encoded patches of shape (num_valid, max_length, hidden_size).
            meta: Metadata dictionary returned by _unfold.
        Returns:
            hidden_states: Reconstructed sequence of shape (batch, original_time, hidden_size).
        """
        batch_size, length, hidden_size = meta['B'], meta['T'], meta['H']
        valid_mask = meta['valid_mask']
        num_patches = valid_mask.shape[1]

        window = self.window.to(valid_patches.device).view(1, self.max_length, 1)

        valid_patches = valid_patches * window

        patches = torch.zeros((batch_size, num_patches, self.max_length, hidden_size), dtype=valid_patches.dtype, device=valid_patches.device)
        patches[valid_mask] = valid_patches

        window_weights = torch.zeros((batch_size, num_patches, self.max_length, 1), dtype=valid_patches.dtype, device=valid_patches.device)
        window_weights[valid_mask] = window

        hidden_states = torch.zeros((batch_size, length, hidden_size), dtype=valid_patches.dtype, device=valid_patches.device)
        hidden_states_weights = torch.zeros((batch_size, length, 1), dtype=valid_patches.dtype, device=valid_patches.device)

        for i in range(num_patches):
            start = i * self.stride
            end = start + self.max_length
            hidden_states[:, start:end, :] += patches[:, i, :, :]
            hidden_states_weights[:, start:end, :] += window_weights[:, i, :, :]

        hidden_states = hidden_states / hidden_states_weights.clamp(min=1e-8)

        hidden_states = hidden_states[:, :meta['original_T'], :]

        return hidden_states

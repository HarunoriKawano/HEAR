import torch
import torch.nn.functional as F

from src.models.encoder import Encoder
from src.config import AcousticModelConfig


class AcousticModel(Encoder):
    def __init__(self, config: AcousticModelConfig):
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

    def forward(self, inputs, input_lengths):
        hidden_states, input_lengths = self.feature_extractor(inputs, input_lengths)
        is_long = hidden_states.size(1) > self.max_length

        if is_long:
            hidden_states, encoder_input_lengths, meta = self._unfold(hidden_states, input_lengths)
            hidden_states = self.contextual_encoder(hidden_states, encoder_input_lengths)
            hidden_states = self._fold(hidden_states, meta)
        else:
            hidden_states = self.contextual_encoder(hidden_states, input_lengths)


        return hidden_states, input_lengths


    def _unfold(self, x: torch.Tensor, lengths: torch.Tensor):
        B, T, H = x.shape
        W = self.max_length
        S = self.stride

        pad_len = (S - (T - W) % S) % S
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            T = x.shape[1]

        num_patches = (T - W) // S + 1
        patches = x.unfold(1, W, S).transpose(2, 3)

        patch_starts = torch.arange(0, num_patches * S, S, device=x.device)

        is_start_valid = patch_starts.unsqueeze(0) < lengths.unsqueeze(1)

        previous_patch_ends = patch_starts - S + W
        is_needed = (patch_starts == 0).unsqueeze(0) | (lengths.unsqueeze(1) > previous_patch_ends.unsqueeze(0))

        valid_mask = is_start_valid & is_needed

        patch_lengths = torch.clamp(lengths.unsqueeze(1) - patch_starts.unsqueeze(0), min=0, max=W)

        valid_patches = patches[valid_mask]
        valid_patch_lengths = patch_lengths[valid_mask]

        meta = {
            'B': B,
            'T': T,
            'H': H,
            'original_T': T - pad_len,
            'valid_mask': valid_mask,
        }

        return valid_patches, valid_patch_lengths, meta

    def _fold(self, valid_patches: torch.Tensor, meta: dict):

        B, T, H = meta['B'], meta['T'], meta['H']
        valid_mask = meta['valid_mask']
        W = self.max_length
        S = self.stride
        num_patches = valid_mask.shape[1]

        window = self.window.to(valid_patches.device).view(1, W, 1)

        valid_patches = valid_patches * window

        patches = torch.zeros((B, num_patches, W, H), dtype=valid_patches.dtype, device=valid_patches.device)
        patches[valid_mask] = valid_patches

        window_weights = torch.zeros((B, num_patches, W, 1), dtype=valid_patches.dtype, device=valid_patches.device)
        window_weights[valid_mask] = window

        out = torch.zeros((B, T, H), dtype=valid_patches.dtype, device=valid_patches.device)
        out_weights = torch.zeros((B, T, 1), dtype=valid_patches.dtype, device=valid_patches.device)

        for i in range(num_patches):
            start = i * S
            end = start + W
            out[:, start:end, :] += patches[:, i, :, :]
            out_weights[:, start:end, :] += window_weights[:, i, :, :]

        out = out / out_weights.clamp(min=1e-8)

        out = out[:, :meta['original_T'], :]

        return out

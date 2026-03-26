import torch
from torch import nn
from torch.nn.functional import unfold

from src.config import AcousticModelConfig


class SpectrogramMixture(nn.Module):
    """Fuses acoustic hidden states with gated power spectrum features.

    The power spectrum is linearly projected and modulated by a sigmoid gate
    derived from the hidden states, then concatenated and projected to the
    output hidden size.
    """

    def __init__(self, config: AcousticModelConfig, out_hidden_size: int):
        super().__init__()
        self.gate_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.spectrum_linear = nn.Linear((config.spectrogram_n_fft // 2 + 1) * 2, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.gelu = nn.GELU()

        self.mixture = nn.Linear(config.hidden_size * 2, out_hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        power_spectrum: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Acoustic encoder output, shape (batch, time, hidden_size).
            power_spectrum: Power spectrogram, shape (batch, freq_bins, time).
        Returns:
            hidden_states: Fused features of shape (batch, time, out_hidden_size).
        """
        gate = self.gate_linear(hidden_states)
        gate = torch.sigmoid(gate)

        power_spectrum = power_spectrum.unsqueeze(2)
        power_spectrum = unfold(power_spectrum, kernel_size=(1, 2), stride=(1, 2))
        power_spectrum = power_spectrum.transpose(1, 2)

        spectrum_features = self.spectrum_linear(power_spectrum)
        spectrum_features = spectrum_features * gate
        spectrum_features = self.layer_norm(spectrum_features)
        spectrum_features = self.gelu(spectrum_features)

        hidden_states = torch.cat([hidden_states, spectrum_features], dim=-1)
        hidden_states = self.mixture(hidden_states)

        return hidden_states

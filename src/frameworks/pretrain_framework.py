import torch
from torch import nn

from src.utils import make_attention_mask
from src.config import TokenizerConfig, AcousticModelConfig
from src.models.acoustic_model import AcousticModel
from src.models.tokenizer import Tokenizer
from src.preprocessing import Preprocessor


class PretrainFramework(nn.Module):
    """Self-supervised pre-training via masked audio prediction.

    The frozen tokenizer generates discrete token labels. The acoustic encoder
    is trained to predict these labels at randomly masked positions.
    """

    def __init__(
        self,
        acoustic_model_config: AcousticModelConfig,
        tokenizer_config: TokenizerConfig,
        mask_rate: float = 0.4,
    ):
        super().__init__()
        self.preprocessor = Preprocessor(acoustic_model_config)
        self.tokenizer = Tokenizer(tokenizer_config)
        for param in self.tokenizer.parameters():
            param.requires_grad = False
        self.encoder = AcousticModel(acoustic_model_config)
        self.out_linear = nn.Linear(acoustic_model_config.hidden_size, tokenizer_config.num_codebooks)

        self.masked_audio_prediction_loss = nn.CrossEntropyLoss()
        self.mask_rate = mask_rate

    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            inputs: Raw waveform of shape (batch, channels, samples).
            input_lengths: Valid sample count per waveform, shape (batch,).
        Returns:
            loss: Masked audio prediction cross-entropy loss.
        """
        with torch.no_grad():
            log_mel_spectrogram, input_lengths, _ = self.preprocessor(inputs, input_lengths)
            labels, _ = self.tokenizer(log_mel_spectrogram.detach(), input_lengths)
            labels = labels.argmax(dim=-1)

        hidden_states, lengths = self.encoder.feature_extractor(log_mel_spectrogram, input_lengths)
        masked_hidden_states, mask_position = self.masking(hidden_states, lengths)
        hidden_states = self.encoder.contextual_encoder(masked_hidden_states, lengths)
        out = self.out_linear(hidden_states)
        loss = self.masked_audio_prediction_loss(out[mask_position], labels[mask_position])

        return loss

    @torch.no_grad()
    def masking(
        self,
        hidden_states: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple]:
        """Randomly zero out a fraction of valid frames and return their positions.

        Args:
            hidden_states: (batch, time, hidden_size)
            lengths: Valid frame count per sample, shape (batch,).
        Returns:
            hidden_states: Masked feature tensor with the same shape.
            mask_position: Index tuple of masked positions for loss computation.
        """
        mask_position = torch.randn(hidden_states.size(0), hidden_states.size(1), device=hidden_states.device)
        mask_position = (make_attention_mask(hidden_states, lengths) == 0) + mask_position
        mask_position = torch.where(mask_position < self.mask_rate)
        mask = torch.ones_like(hidden_states)
        mask[mask_position] = 0
        hidden_states = hidden_states * mask

        return hidden_states, mask_position

import torch
from torch import nn

from src.utils import make_attention_mask
from src.config import TokenizerConfig, AcousticModelConfig
from src.models.acoustic_model import AcousticModel
from src.models.tokenizer import Tokenizer
from src.preprocessing import Preprocessor


class PretrainFramework(nn.Module):
    def __init__(self, acoustic_model_config: AcousticModelConfig, tokenizer_config: TokenizerConfig, mask_rate: float = 0.4):
        super().__init__()
        self.preprocessor = Preprocessor(acoustic_model_config)
        self.tokenizer = Tokenizer(tokenizer_config)
        for param in self.tokenizer.parameters():
            param.requires_grad = False
        self.encoder = AcousticModel(acoustic_model_config)
        self.out_linear = nn.Linear(acoustic_model_config.hidden_size, tokenizer_config.num_codebooks)

        self.masked_audio_prediction_loss = nn.CrossEntropyLoss()
        self.mask_rate = mask_rate

    def forward(self, inputs, input_lengths):
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
    def masking(self, hidden_states: torch.Tensor, lengths):
        mask_position = torch.randn(hidden_states.size(0), hidden_states.size(1), device=hidden_states.device)
        mask_position = (make_attention_mask(hidden_states, lengths) == 0) + mask_position
        mask_position = torch.where(mask_position < self.mask_rate)
        mask = torch.ones_like(hidden_states)
        mask[mask_position] = 0
        hidden_states = hidden_states * mask

        return hidden_states, mask_position

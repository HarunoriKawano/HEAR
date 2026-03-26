import torch
from torch import nn

from src.preprocessing import Preprocessor
from src.models.task_model import TaskModel
from src.models.spectrogram_mixture import SpectrogramMixture
from src.models.acoustic_model import AcousticModel
from src.models.classification_decoder import Decoder
from src.config import AcousticModelConfig, TaskModelConfig, ClassificationDecoderConfig


class HEAR(nn.Module):
    """Full HEAR pipeline for audio classification.

    Pipeline: waveform → Preprocessor → AcousticModel → SpectrogramMixture
              → TaskModel → ClassificationDecoder → class logits.
    """

    def __init__(
        self,
        acoustic_model_config: AcousticModelConfig,
        task_model_config: TaskModelConfig,
        decoder_config: ClassificationDecoderConfig,
    ):
        super().__init__()
        self.preprocessor = Preprocessor(acoustic_model_config)
        self.acoustic_model = AcousticModel(acoustic_model_config)
        self.spectrogram_mixture = SpectrogramMixture(acoustic_model_config, task_model_config.hidden_size)
        self.task_model = TaskModel(task_model_config)
        self.decoder = Decoder(decoder_config, task_model_config.hidden_size)

    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Args:
            inputs: Raw waveform of shape (batch, channels, samples).
            input_lengths: Valid sample count per waveform, shape (batch,).
        Returns:
            out: Class logits of shape (batch, num_classes).
        """
        log_mel_spectrogram, input_lengths, power_spectrum = self.preprocessor(inputs, input_lengths)
        hidden_states, input_lengths = self.acoustic_model(log_mel_spectrogram, input_lengths)
        hidden_states = self.spectrogram_mixture(hidden_states, power_spectrum)
        hidden_states = self.task_model(hidden_states, input_lengths)
        out = self.decoder(hidden_states, input_lengths)

        return out

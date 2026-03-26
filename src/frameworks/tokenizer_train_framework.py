import torch
from torch import nn

from src.utils import make_attention_mask
from src.config import TokenizerConfig, DecoderConfig
from src.params import TokenizerTrainParams
from src.models.tokenizer import Tokenizer
from src.preprocessing import Preprocessor
from src.models.decoder import Decoder


class TokenizerTrainFramework(nn.Module):
    """Training framework for the neural tokenizer.

    Trains the tokenizer end-to-end with three objectives:
    spectrogram reconstruction, knowledge distillation from a teacher model,
    and codebook diversity regularization.
    """

    def __init__(
        self,
        tokenizer_config: TokenizerConfig,
        decoder_config: DecoderConfig,
        params: TokenizerTrainParams,
    ):
        super().__init__()
        self.preprocessor = Preprocessor(tokenizer_config)
        self.tokenizer = Tokenizer(tokenizer_config)
        self.reconstruction_loss = nn.MSELoss()
        self.distillation_loss = nn.CosineEmbeddingLoss()

        self.decoder = Decoder(decoder_config)
        self.vector_quantizer = GumbelVectorQuantizer(tokenizer_config=tokenizer_config, decoder_config=decoder_config, params=params)

        self.diversity_loss_weight = params.diversity_loss_weight
        self.reconstruction_loss_weight = params.reconstruction_loss_weight
        self.distillation_loss_weight = params.distillation_loss_weight

    def forward(
            self,
            inputs: torch.Tensor,
            input_lengths: torch.Tensor,
            knowledge_target: torch.Tensor,
            training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: Raw waveform of shape (batch, channels, samples).
            input_lengths: Valid sample count per waveform, shape (batch,).
            knowledge_target: Teacher model hidden states for distillation, shape (batch, time, knowledge_hidden_size).
            training: Whether to update the Gumbel temperature.
        Returns:
            reconstruction_loss, distillation_loss, diversity_loss: Weighted scalar losses.
        """
        log_mel_spectrogram, input_lengths, _ = self.preprocessor(inputs, input_lengths)
        log_mel_spectrogram_clone = log_mel_spectrogram.clone().detach()

        discrete_hidden_states, encoder_lengths = self.tokenizer(log_mel_spectrogram.clone(), input_lengths)

        quantized_vector, diversity_loss = self.vector_quantizer(discrete_hidden_states, encoder_lengths, training)

        predicted, knowledge_prediction, out_lengths = self.decoder(quantized_vector, encoder_lengths)
        target_position = torch.where(make_attention_mask(predicted, out_lengths))
        predicted = predicted[target_position]

        labels = self.make_labels(log_mel_spectrogram_clone, out_lengths)
        reconstruction_loss = self.reconstruction_loss(predicted, labels) * self.reconstruction_loss_weight

        target_position = make_attention_mask(knowledge_prediction, encoder_lengths)
        knowledge_prediction = knowledge_prediction[torch.where(target_position)]
        knowledge_target = knowledge_target[torch.where(target_position)]
        target = torch.ones(knowledge_target.size(0), device=knowledge_target.device)
        distillation_loss = self.distillation_loss(knowledge_prediction, knowledge_target, target) * self.distillation_loss_weight

        diversity_loss *= self.diversity_loss_weight

        return reconstruction_loss, distillation_loss, diversity_loss

    @torch.no_grad()
    def make_labels(self, log_mel_spectrogram: torch.Tensor, out_lengths: torch.Tensor) -> torch.Tensor:
        """Extract valid mel spectrogram frames as reconstruction targets.

        Args:
            log_mel_spectrogram: (batch, n_mel, time)
            out_lengths: Valid frame count for the decoder output, shape (batch,).
        Returns:
            labels: Flattened valid frames, shape (num_valid, n_mel).
        """
        max_length = out_lengths.max()

        labels = log_mel_spectrogram.transpose(1, 2)
        labels = labels[:, :int(max_length)]

        target_position = torch.where(make_attention_mask(labels, out_lengths))
        labels = labels[target_position]

        return labels


class GumbelVectorQuantizer(nn.Module):
    """Gumbel-softmax vector quantizer that maps encoder logits to continuous code vectors.

    During training the temperature is annealed from gumbel_init_temperature
    down to gumbel_min_temperature, encouraging sharper codebook assignments.
    """

    def __init__(self, params: TokenizerTrainParams, tokenizer_config: TokenizerConfig, decoder_config: DecoderConfig):
        super().__init__()

        self.code_book = nn.Parameter(
            torch.randn(1, tokenizer_config.num_codebooks, decoder_config.code_vector_size)
        )
        self.linear = nn.Linear(decoder_config.code_vector_size, decoder_config.hidden_size)
        self.diversity_loss_label = torch.softmax(torch.ones(tokenizer_config.num_codebooks, dtype=torch.float), dim=-1)

        self.temperature = params.gumbel_init_temperature
        self.min_temperature = params.gumbel_min_temperature

    @staticmethod
    def _compute_perplexity(probs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Average the soft codebook distribution over valid positions.

        Args:
            probs: Soft assignment distribution, shape (batch, time, num_codebooks).
            lengths: Valid frame count per sample, shape (batch,).
        Returns:
            perplexity: Mean distribution over valid frames, shape (num_codebooks,).
        """
        where_calculate_probs = torch.arange(probs.size(1), device=probs.device).unsqueeze(0) < lengths.unsqueeze(-1)
        probs = probs[where_calculate_probs == 1]

        num_values = probs.size(0)
        perplexity = probs.sum(0) / num_values

        return perplexity

    def forward(
        self,
        discrete_hidden_states: torch.Tensor,
        lengths: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            discrete_hidden_states: Tokenizer logits, shape (batch, time, num_codebooks).
            lengths: Valid frame count per sample, shape (batch,).
            training: Whether to anneal the Gumbel temperature.
        Returns:
            hidden_states: Quantized and projected vectors, shape (batch, time, decoder_hidden_size).
            diversity_loss: Scalar diversity regularization loss.
        """
        batch_size, length, _ = discrete_hidden_states.shape

        discrete_hidden_states = discrete_hidden_states.view(batch_size * length, -1)

        code_vector_probs = nn.functional.gumbel_softmax(
            discrete_hidden_states.float(), tau=self.temperature, hard=True
        ).type_as(discrete_hidden_states)
        code_vector_soft_dist = torch.softmax(
            discrete_hidden_states.view(batch_size, length, -1).float(), dim=-1
        )
        perplexity = self._compute_perplexity(code_vector_soft_dist, lengths)

        code_vector_probs = code_vector_probs.view(batch_size * length, -1).unsqueeze(-1)

        code_vectors = code_vector_probs * self.code_book
        code_vectors = code_vectors.sum(-2).view(batch_size, length, -1)
        hidden_states = self.linear(code_vectors)
        diversity_loss = self.diversity_loss(perplexity)

        if training:
            self.update_temperature()

        return hidden_states, diversity_loss

    def update_temperature(self) -> None:
        """Exponentially decay the Gumbel temperature towards the minimum."""
        self.temperature = max([self.min_temperature, self.temperature * 0.999995])

    def finished_temperature_updating(self) -> bool:
        """Return True when the temperature has reached its minimum."""
        return self.temperature == self.min_temperature

    def diversity_loss(self, target: torch.Tensor) -> torch.Tensor:
        """Penalize deviation of the average codebook distribution from uniform.

        Args:
            target: Mean soft assignment distribution, shape (num_codebooks,).
        Returns:
            diversity_loss: Scalar loss encouraging uniform codebook usage.
        """
        diversity_loss = ((target - self.diversity_loss_label.to(target.device)) ** 2).sum()

        return diversity_loss

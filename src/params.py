from pydantic import BaseModel


class TokenizerTrainParams(BaseModel):
    """Hyperparameters for tokenizer training, including loss weights and Gumbel temperature schedule."""
    gumbel_init_temperature: float = 2.0
    gumbel_min_temperature: float = 0.5
    diversity_loss_weight: float = 0.1
    reconstruction_loss_weight: float = 0.3
    distillation_loss_weight: float = 0.7

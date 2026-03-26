import torch

from src.frameworks.pretrain_framework import PretrainFramework
from src.frameworks.tokenizer_train_framework import TokenizerTrainFramework
from src.hear import HEAR
from src.ssast import SSAST


def tokenizer_train_step(
        tokenizer_train_framework: TokenizerTrainFramework,
        inputs: torch.Tensor, input_lengths: torch.Tensor,
        teacher_model: SSAST, optimizer, scheduler
):
    """Single training step for the tokenizer with knowledge distillation."""
    with torch.no_grad():
        knowledge_targets = teacher_model(inputs)
        knowledge_targets = knowledge_targets.detach()

    reconstruction_loss, distillation_loss, diversity_loss = tokenizer_train_framework(inputs, input_lengths, knowledge_targets, training=True)
    loss = reconstruction_loss + distillation_loss + diversity_loss
    loss.backward()
    optimizer.step()
    scheduler.step()


def pretrain_step(
        pretrain_framework: PretrainFramework,
        inputs: torch.Tensor, input_lengths: torch.Tensor,
        optimizer, scheduler
):
    """Single masked audio prediction pre-training step."""
    loss = pretrain_framework(inputs, input_lengths)
    loss.backward()
    optimizer.step()
    scheduler.step()


def downstream_train_step(
        hear: HEAR,
        inputs: torch.Tensor, input_lengths: torch.Tensor, labels: torch.Tensor,
        optimizer, scheduler
):
    """Single downstream classification training step."""
    out = hear(inputs, input_lengths)
    loss = torch.nn.functional.cross_entropy(out, labels)
    loss.backward()
    optimizer.step()
    scheduler.step()

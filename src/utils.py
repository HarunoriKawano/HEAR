import torch
import pathlib
from typing import Type
from pydantic import BaseModel

def make_attention_mask(hidden_states: torch.Tensor, lengths: torch.Tensor):
    batch_size, max_length = hidden_states.size(0), hidden_states.size(1)
    range_tensor = torch.arange(max_length, device=hidden_states.device).repeat(batch_size, 1)
    attention_mask = torch.as_tensor(range_tensor < lengths.unsqueeze(1), device=hidden_states.device)

    return attention_mask

def json_to_instance(path: str, structure: Type[BaseModel]):
    json_string = pathlib.Path(path).read_text()

    instance = structure.model_validate_json(json_string)
    return instance

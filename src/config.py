from pydantic import BaseModel


class AcousticModelConfig(BaseModel):
    hidden_size: int = 384
    intermediate_size: int = 1536
    num_attention_heads: int = 4
    num_transformer_layers: int = 6
    max_length: int = 300
    overlap: int = 50
    dropout_probability: float = 0.1
    spectrogram_sampling_rate: int = 16000
    spectrogram_n_fft: int = 1024

    def __post_init__(self):
        assert self.hidden_size % self.num_attention_heads == 0


class TokenizerConfig(BaseModel):
    hidden_size: int = 128
    intermediate_size: int = 512
    num_attention_heads: int = 4
    num_transformer_layers: int = 6
    max_length: int = 300
    dropout_probability: float = 0.1
    num_codebooks: int = 1024
    spectrogram_sampling_rate: int = 16000
    spectrogram_n_fft: int = 1024

    def __post_init__(self):
        assert self.hidden_size % self.num_attention_heads == 0


class DecoderConfig(BaseModel):
    hidden_size: int = 128
    intermediate_size: int = 512
    num_attention_heads: int = 4
    num_transformer_layers: int = 2
    max_length: int = 300
    dropout_probability: float = 0.1
    knowledge_hidden_size: int = 768
    code_vector_size: int = 32

    def __post_init__(self):
        assert self.hidden_size % self.num_attention_heads == 0


class TaskModelConfig(BaseModel):
    hidden_size: int = 384
    intermediate_size: int = 1536
    num_transformer_layers: int = 1
    num_attention_heads: int = 4
    dropout_rate: float = 0.1
    max_length: int = 1500


class ClassificationDecoderConfig(BaseModel):
    num_classes: int
    hidden_size: int = 384
    dropout_rate: float = 0.1


class StaticConfig(BaseModel):
    n_fft: int = 1024
    sampling_rate: int = 16000
    win_seconds: float = 0.025
    hop_seconds: float = 0.01
    n_mel: int = 128
    eps: float = 1e-6

STATIC_CONFIG = StaticConfig()

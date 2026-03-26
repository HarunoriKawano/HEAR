import torch
from torch import nn
from torchaudio.compliance import kaldi

from ssast.src.models.ast_models import ASTModel


class SSAST(nn.Module):
    """Wrapper around a pretrained SSAST model used as a knowledge distillation teacher.

    The model extracts hidden states from fixed-length mel spectrograms,
    which serve as targets for the tokenizer's distillation objective.
    """

    def __init__(self, model_weight_path: str):
        super().__init__()
        self.model = ASTModel(label_dim=527,
                              fshape=128, tshape=2, fstride=128, tstride=2,
                              input_fdim=128, input_tdim=1024, model_size='base',
                              pretrain_stage=False, load_pretrained_mdl_path=model_weight_path)
        self.fixed_input_length = 1024
        self.norm_mean = -4.2677393
        self.norm_std = 4.5689974

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Raw waveform of shape (batch, channels, samples).
        Returns:
            out: Hidden states extracted by the SSAST model.
        """
        mel_spectrogram = torch.stack(list(map(self.pre_processing, inputs)), dim=0)
        max_input_length = mel_spectrogram.size(1)
        padding_length = self.fixed_input_length - max_input_length
        padding = torch.zeros(
            mel_spectrogram.size(0), padding_length, mel_spectrogram.size(2),
            dtype=torch.float, device=inputs.device
        )
        mel_spectrogram = torch.cat([mel_spectrogram, padding], dim=1)

        mel_spectrogram = (mel_spectrogram - self.norm_mean) / (self.norm_std * 2)

        out = self.model(mel_spectrogram, task='extract_hidden_states')

        return out

    @staticmethod
    def pre_processing(waveform: torch.Tensor) -> torch.Tensor:
        """Convert a single waveform to a Kaldi-style mel filterbank spectrogram.

        Args:
            waveform: (channels, samples)
        Returns:
            mel_spectrogram: (time, n_mels)
        """
        waveform = waveform - waveform.mean()

        mel_spectrogram = kaldi.fbank(waveform, htk_compat=True, sample_frequency=16000, use_energy=False,
                                      window_type='hanning', num_mel_bins=128, dither=0.0,
                                      frame_shift=10)

        return mel_spectrogram

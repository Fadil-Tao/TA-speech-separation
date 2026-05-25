from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch_complex.tensor import ComplexTensor
from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.separator.abs_separator import AbsSeparator
from implementation.skim_attention.skim_attention import SkiM

class SkiMAttentionSeparator(AbsSeparator):

    def __init__(self, input_dim: int, causal: bool=True, num_spk: int=2, predict_noise: bool=False, nonlinear: str='relu', layer: int=3, unit: int=512, segment_size: int=20, dropout: float=0.0, mem_type: str='hc', seg_overlap: bool=False, num_heads: int=4):
        super().__init__()
        self._num_spk = num_spk
        self.predict_noise = predict_noise
        self.segment_size = segment_size
        if mem_type not in ('hc', 'h', 'c', 'id', None):
            raise ValueError(f'Not supporting mem_type={mem_type}')
        self.num_outputs = self.num_spk + 1 if self.predict_noise else self.num_spk
        self.skim = SkiM(input_size=input_dim, hidden_size=unit, output_size=input_dim * self.num_outputs, dropout=dropout, num_blocks=layer, bidirectional=not causal, norm_type='cLN' if causal else 'gLN', segment_size=segment_size, seg_overlap=seg_overlap, mem_type=mem_type, num_heads=num_heads)
        if nonlinear not in ('sigmoid', 'relu', 'tanh'):
            raise ValueError(f'Not supporting nonlinear={nonlinear}')
        self.nonlinear = {'sigmoid': nn.Sigmoid(), 'relu': nn.ReLU(), 'tanh': nn.Tanh()}[nonlinear]

    def forward(self, input: Union[torch.Tensor, ComplexTensor], ilens: torch.Tensor, additional: Optional[Dict]=None) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        feature = abs(input) if is_complex(input) else input
        B, T, N = feature.shape
        processed = self.skim(feature)
        processed = processed.view(B, T, N, self.num_outputs)
        masks = self.nonlinear(processed).unbind(dim=3)
        if self.predict_noise:
            *masks, mask_noise = masks
        masked = [input * m for m in masks]
        others = OrderedDict(zip([f'mask_spk{i + 1}' for i in range(len(masks))], masks))
        if self.predict_noise:
            others['noise1'] = input * mask_noise
        return (masked, ilens, others)

    def forward_streaming(self, input_frame: torch.Tensor, states=None):
        raise NotImplementedError('Streaming inference is not supported for SkiMAttentionSeparator.')

    @property
    def num_spk(self):
        return self._num_spk

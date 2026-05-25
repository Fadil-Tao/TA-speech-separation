import torch
from espnet2.enh.encoder.conv_encoder import ConvEncoder

class ConvEncoderAbs(ConvEncoder):

    def forward(self, input: torch.Tensor, ilens: torch.Tensor):
        assert input.dim() == 2, 'Currently only support single channel input'
        input = torch.unsqueeze(input, 1)
        feature = self.conv1d(input)
        feature = torch.abs(feature)
        feature = feature.transpose(1, 2)
        flens = (ilens - self.kernel_size) // self.stride + 1
        return (feature, flens)

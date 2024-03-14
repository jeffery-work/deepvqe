import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class CausalConv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 2), stride=(2, 1), padding=(0, 1)):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding
        )
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.PReLU()

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[..., :-1]  # chomp size, the last dimension num minus one, example: (3,4)->(3,3)
        x = self.norm(x)
        x = self.activation(x)
        return x


class CausalTransConv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_last=False, kernel_size=(3, 2), stride=(2, 1),
                 output_padding=(1, 0)):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            output_padding=output_padding
        )
        self.norm = nn.InstanceNorm2d(num_features=out_channels)
        if is_last:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.PReLU()

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[..., :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x

if __name__ == '__main__':
    a = CausalConv2dBlock(1, 1)
    b = CausalTransConv2dBlock(1, 1)
    input = torch.rand((1, 1, 4, 8))  # B C F T
    output = a.forward(input)
    print(output.shape)

    input = torch.rand((1, 1, 5, 8))
    output = a.forward(input)
    print(output.shape)
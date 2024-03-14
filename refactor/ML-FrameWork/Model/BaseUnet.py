import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchvision import models
#from torchsummary import summary
#from ptflops import get_model_complexity_info
from Model.Component.ConvModule import CausalConv2dBlock, CausalTransConv2dBlock

class BaseUnet(torch.nn.Module):
    def __init__(self):
        super(BaseUnet, self).__init__()
        # container to save nn.Module, such as conv, pRelu, and module will be automatically registered into the network
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.tcmlayer = nn.ModuleList()
        self.tcmlayer_num = 2
        dilation = [1, 2, 4, 8, 16, 32]
        kernel_num = [1, 8, 16, 32, 32, 64, 128]
        pading = [(0, 0), (0, 0), (1, 0), (0, 0), (0, 0), (0, 0), (1, 0)]
        for i in range(len(kernel_num) - 1):
            self.encoder.append(
                # Sequential has internal "forward" function
                nn.Sequential(
                    CausalConv2dBlock(
                        in_channels=kernel_num[i],
                        out_channels=kernel_num[i+1],
                    )
                )
            )
        for i in range(self.tcmlayer_num):
            # gru: input_size = kernel_num[-1] = 128, hidden_size = 128, num_layers = 1
            self.tcmlayer.append(nn.GRU(kernel_num[-1], kernel_num[-1], 1, batch_first=True))
        for i in range(len(kernel_num) - 1, 0, -1):
            self.decoder.append(
                nn.Sequential(
                    CausalTransConv2dBlock(
                        in_channels=kernel_num[i],
                        out_channels=kernel_num[i - 1] if i != 1 else 1,
                        output_padding=pading[i]
                    )
                )
            )

    def forward(self, input):
        outs = input
        encoder_out = []
        for i in range(len(self.encoder)):
            outs = self.encoder[i](outs)
            encoder_out.append(outs)

        B, C, D, T = outs.size() # tensor:[batch, channel, height, width]
        outs = outs.reshape(B, T, -1) # tensor: 2*3 -> reshape(-1,2) -> tensor: 3*2, reshape(-1)-> tensor: 1*6
        for i in range(len(self.tcmlayer)):
            outs, h = self.tcmlayer[i](outs) # [batch, sequence_len, other_size]
        outs = outs.reshape(B, -1, T)
        outs = outs.view(B, C, D, T)

        for i in range(len(self.decoder)):
            outs = self.decoder[i](outs + encoder_out[-1-i])
        return outs


if __name__ == "__main__":
    kernel_num = [1, 8, 16, 32, 32, 64, 128]
    a = kernel_num[-1]
    x = torch.arange(12).view(2,2,3)
    x = x.reshape(2, -1)
    Batch_SIZE = 10
    T = 1001
    F = 161
    input = torch.rand((Batch_SIZE, 1, F, T))  # B C F T
    net = BaseUnet()
    output = net(input)
    print(output.shape)

    # 估计模型大小以及计算量  inputs size don't include the batch dim
    #ops, params = get_model_complexity_info(net, (1, F, T), as_strings=True,
                                            #print_per_layer_stat=True, verbose=True)

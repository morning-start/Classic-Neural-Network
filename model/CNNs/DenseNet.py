import torch
from torch import nn


class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(
                nn.Sequential(
                    nn.BatchNorm2d(num_channels * i + input_channels),
                    nn.ReLU(),
                    nn.Conv2d(
                        num_channels * i + input_channels,
                        num_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                )
            )
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 连接通道维度上每个块的输入和输出
            X = torch.cat((X, Y), dim=1)
        return X

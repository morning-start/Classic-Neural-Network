from typing import NamedTuple

from torch import nn


class VGGConfig(NamedTuple):
    """
    VGG Config

    - num_conv: 卷积层的数量
    - out_channels: 输出通道数
    """

    num_conv: int
    out_channels: int


class VGGBlock(nn.Module):
    """VGG Block"""

    def __init__(self, num_conv: int, in_channels: int, out_channels: int):
        super().__init__()
        layers = []
        for _ in range(num_conv):
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class VGG(nn.Module):
    """VGG Model"""

    def __init__(
        self, in_channels: int, conv_arch: list[VGGConfig], num_classes: int = 10
    ):
        super().__init__()
        self.cnn, out_channels = self._vgg(in_channels, conv_arch)
        self.fc = nn.Sequential(
            nn.Flatten(),
            # 全连接层部分
            nn.Linear(out_channels * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def _vgg(self, in_channels, conv_arch):
        conv_blocks = []
        # 卷积层部分
        for num_conv, out_channels in conv_arch:
            conv_blocks.append(VGGBlock(num_conv, in_channels, out_channels))
            in_channels = out_channels

        return nn.Sequential(*conv_blocks), out_channels

    def forward(self, x):
        x = self.cnn(x)
        return self.fc(x)

from torch import nn


class NiNBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        strides: int,
        padding: int,
    ):
        """
        NINBlock类中定义了nin_block函数中创建的串联模块。

        - in_channels: 输入通道数
        - out_channels: 输出通道数
        - kernel_size: 第一层卷积层的卷积核窗口形状
        - strides: 第一层卷积层的步幅
        - padding: 第一层卷积层的填充
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class NiN(nn.Module):
    """
    NiN类定义了NiN模型。

    - in_channels: 输入通道数
    - num_classes: 输出层的输出数
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            NiNBlock(in_channels, 96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            NiNBlock(96, 256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            NiNBlock(256, 384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5),
            NiNBlock(384, num_classes, kernel_size=3, strides=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            # 将四维的输出转成二维的输出，其形状为(批量大小,10)
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)

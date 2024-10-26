import torch
from torch import nn


# LINK https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
class AlexNet(nn.Module):
    """
    AlexNet模型类，继承自nn.Module。

    该模型包含特征提取部分(self.features)、平均池化部分(self.avgpool)和分类器部分(self.classifier)。
    """

    def __init__(self, num_classes=10):
        """
        初始化AlexNet模型。

        Args:
            num_classes (int): 分类器的输出类别数量，默认为10。
            由于这里使用 Fashion-MNIST，所以用类别数为10，而非论文中的1000
        """

        super(AlexNet, self).__init__()
        # NOTE 定义特征提取部分，使用Sequential容器按顺序定义卷积层、激活函数层和池化层
        self.features = nn.Sequential(
            # 这里使用一个11*11的更大窗口来捕捉对象。
            # 同时，步幅为4，以减少输出的高度和宽度。
            # 另外，输出通道的数目远大于LeNet
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 使用三个连续的卷积层和较小的卷积窗口。
            # 除了最后的卷积层，输出通道的数量进一步增加。
            # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
        )
        # # NOTE 定义分类器部分，使用Sequential容器按顺序定义全连接层、激活函数层和Dropout层
        self.classifier = nn.Sequential(
            # 全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
            nn.Linear(6400, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            # 输出层
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

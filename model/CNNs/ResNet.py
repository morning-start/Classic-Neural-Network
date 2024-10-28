import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                input_channels, num_channels, kernel_size=3, padding=1, stride=strides
            ),
            nn.BatchNorm2d(num_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
        )
        if use_1x1conv:
            self.conv3 = nn.Conv2d(
                input_channels, num_channels, kernel_size=1, stride=strides
            )
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(y)
        y = F.relu(y)
        y = self.conv2(y)
        if self.conv3:
            y = self.conv3(y)
        y += x
        return F.relu(y)


class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.b2 = nn.Sequential(*ResNet.resnet_block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*ResNet.resnet_block(64, 128, 2))
        self.b4 = nn.Sequential(*ResNet.resnet_block(128, 256, 2))
        self.b5 = nn.Sequential(*ResNet.resnet_block(256, 512, 2))
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        return self.fc(x)

    @staticmethod
    def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(
                    Residual(input_channels, num_channels, use_1x1conv=True, strides=2)
                )
            else:
                blk.append(Residual(num_channels, num_channels))
        return blk

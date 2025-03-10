import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.reflection_pad = torch.nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        out = self.reflection_pad(x)
        out = self.conv1(out)
        out = self.in1(out)
        out = self.relu(out)
        out = self.reflection_pad(out)
        out = self.conv2(out)
        out = self.in2(out)
        out += residual
        return out
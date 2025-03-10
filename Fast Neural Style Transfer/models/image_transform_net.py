import torch
import torch.nn as nn
from models.residual_block import ResidualBlock


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(
                x_in, mode="nearest", scale_factor=self.upsample
            )
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class ImageTransformationNetwork(nn.Module):
    def __init__(self):
        super(ImageTransformationNetwork, self).__init__()
        self.ref_pad1 = nn.ReflectionPad2d(padding=4)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=9, stride=1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)

        self.ref_pad2 = nn.ReflectionPad2d(padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(64, affine=True)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(128, affine=True)

        self.residuals = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
        )

        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(64, affine=True)

        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(32, affine=True)

        self.deconv3 = UpsampleConvLayer(32, 3, kernel_size=9, stride=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.ref_pad1(x)
        x = self.relu(self.in1(self.conv1(x)))

        x = self.ref_pad2(x)
        x = self.relu(self.in2(self.conv2(x)))

        x = self.ref_pad2(x)
        x = self.relu(self.in3(self.conv3(x)))

        x = self.residuals(x)

        x = self.relu(self.in4(self.deconv1(x)))
        x = self.relu(self.in5(self.deconv2(x)))
        x = self.deconv3(x)
        return x

        """
        ref: http://distill.pub/2016/deconv-checkerboard/
        """
import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet18
from torchvision.models.segmentation import deeplabv3_resnet50
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
import torchvision
import torch.nn.functional as F

from funcs.upernet import UPerNet


'''vgg16-unet'''


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock1(nn.Module):
    """
    Paramaters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlock1, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class VGG16UNet(nn.Module):
    def __init__(self, in_channels=6, out_channel=1, num_filters=32, pretrained=False):
        """
        in_channels: 输入图像的通道数
        out_channel: 输出图像的通道数
        pretrained：是否加载预训练模型
        """
        super().__init__()
        self.out_channel = out_channel

        self.pool = nn.MaxPool2d(2, 2)

        # 加载预训练的VGG16模型
        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)

        # 修改第一层卷积层的输入通道数
        self.encoder[0] = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)  # 支持多通道输入

        # 第一层卷积部分（conv1）
        self.conv1 = nn.Sequential(self.encoder[0],  # 使用修改后的卷积层
                                   self.relu,
                                   self.encoder[2],
                                   self.relu)

        # 其余卷积层部分
        self.conv2 = nn.Sequential(self.encoder[5],
                                   self.relu,
                                   self.encoder[7],
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   self.relu,
                                   self.encoder[12],
                                   self.relu,
                                   self.encoder[14],
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],
                                   self.relu,
                                   self.encoder[19],
                                   self.relu,
                                   self.encoder[21],
                                   self.relu)

        self.conv5 = nn.Sequential(self.encoder[24],
                                   self.relu,
                                   self.encoder[26],
                                   self.relu,
                                   self.encoder[28],
                                   self.relu)

        # UNet 解码部分
        self.center = DecoderBlock1(512, num_filters * 8 * 2, num_filters * 8)
        self.dec5 = DecoderBlock1(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock1(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3 = DecoderBlock1(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock1(128 + num_filters * 2, num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)

        # 最后的输出卷积层
        self.final = nn.Conv2d(num_filters, out_channel, kernel_size=1)

    def forward(self, x):
        # 编码器部分
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        # 中心层
        center = self.center(self.pool(conv5))

        # 解码器部分
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        # 最终输出
        x_out = F.relu(self.final(dec1))

        return x_out


'''vgg16-unet'''




class ResNet50_UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=1):
        super(ResNet50_UNet, self).__init__()
        # 使用ResNet50的预训练模型作为编码器
        self.encoder = resnet50(pretrained=True)
        self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 编码器的特征提取部分
        self.enc1 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu,
                                  self.encoder.maxpool)  # 64x128x128
        self.enc2 = self.encoder.layer1  # 256x128x128
        self.enc3 = self.encoder.layer2  # 512x64x64
        self.enc4 = self.encoder.layer3  # 1024x32x32
        self.enc5 = self.encoder.layer4  # 2048x16x16

        # UNet解码器部分
        self.upconv4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(2048, 1024)

        self.upconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(1024, 512)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(512, 256)

        self.upconv1 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=4)
        self.dec1 = self.conv_block(64, 32)

        # 输出层，将输出通道数设为1，用于回归
        # self.out_conv = nn.Sequential(nn.Conv2d(32, out_channels, kernel_size=1), nn.ReLU())
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # 编码器部分
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        # 解码器部分 + 跳跃连接
        dec4 = self.upconv4(enc5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = self.dec1(dec1)

        # 输出
        out = self.out_conv(dec1)
        return out


class DeepLabV3_Regression(nn.Module):
    def __init__(self, in_channels=6, out_channels=1):
        super(DeepLabV3_Regression, self).__init__()
        # 使用预训练的DeepLabV3模型
        self.deeplab = deeplabv3_resnet50(pretrained=True)
        # 修改第一层卷积层
        self.deeplab.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 修改最后的分类头部，输出1通道
        self.deeplab.classifier[4] = nn.Conv2d(256, out_channels, kernel_size=1)

    def forward(self, x):
        return self.deeplab(x)['out']


class DecoderBlock(nn.Module):
    def __init__(self,
                 in_channels=512,
                 n_filters=256,
                 kernel_size=3,
                 is_deconv=False,
                 ):
        super().__init__()

        if kernel_size == 3:
            conv_padding = 1
        elif kernel_size == 1:
            conv_padding = 0

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels,
                               in_channels // 4,
                               kernel_size,
                               padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        if is_deconv == True:
            self.deconv2 = nn.ConvTranspose2d(in_channels // 4,
                                              in_channels // 4,
                                              3,
                                              stride=2,
                                              padding=1,
                                              output_padding=conv_padding, bias=False)
        else:
            up_kwargs = {'mode': 'bilinear', 'align_corners': True}
            self.deconv2 = nn.Upsample(scale_factor=2, **up_kwargs)

        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4,
                               n_filters,
                               kernel_size,
                               padding=conv_padding, bias=False)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class Eb3net(nn.Module):
    def __init__(self,
                 num_classes,
                 num_channels=6,
                 is_deconv=False,
                 decoder_kernel_size=3,
                 ):
        super().__init__()

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        filters = [32, 48, 136, 1536, 40]
        efficientnet_b3 = models.efficientnet_b3()
        self.base_size = 512
        self.crop_size = 512
        if num_channels == 3:
            self.firstconv = efficientnet_b3.features[0][0]
        else:
            self.firstconv = nn.Conv2d(num_channels, 40, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.firstbn = efficientnet_b3.features[0][1]
        self.firstsilu = efficientnet_b3.features[0][2]  # 128

        self.encoder1 = efficientnet_b3.features[1:3]  # 64
        self.encoder2 = efficientnet_b3.features[3:4]  # 32
        self.encoder3 = efficientnet_b3.features[4:6]  # 16
        self.encoder4 = efficientnet_b3.features[6:]  # 8

        # Decoder
        self.center = DecoderBlock(in_channels=filters[3],
                                   n_filters=filters[3],
                                   kernel_size=decoder_kernel_size,
                                   is_deconv=is_deconv)

        self.decoder4 = DecoderBlock(in_channels=filters[3] + filters[2],
                                     n_filters=filters[2],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder3 = DecoderBlock(in_channels=filters[2] + filters[1],
                                     n_filters=filters[1],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder2 = DecoderBlock(in_channels=filters[1] + filters[0],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder1 = DecoderBlock(in_channels=filters[0] + filters[4],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)

        self.finalconv = nn.Sequential(nn.Conv2d(filters[0], 32, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(),
                                       nn.Dropout2d(0.1, False),
                                       nn.Conv2d(32, num_classes, 1))

    def require_encoder_grad(self, requires_grad):
        blocks = [self.firstconv,
                  self.encoder1,
                  self.encoder2,
                  self.encoder3,
                  self.encoder4]

        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

    def forward(self, x):
        # stem
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstsilu(x)

        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        center = self.center(e4)

        d4 = self.decoder4(torch.cat([center, e3], 1))
        d3 = self.decoder3(torch.cat([d4, e2], 1))
        d2 = self.decoder2(torch.cat([d3, e1], 1))
        d1 = self.decoder1(torch.cat([d2, x], 1))

        f = self.finalconv(d1)

        return f


if __name__ == '__main__':
    test = torch.randn(2,6,256,256)
    # test2 = torch.randn(2, 3, 256, 256)
    m1 = ResNet50_UNet()
    m2 = DeepLabV3_Regression()
    m3 = Eb3net(num_classes=1)
    m4=VGG16UNet()
    m5=UPerNet(1)
    a = m1(test)
    b=m2(test)
    c=m3(test)
    d=m4(test)
    e=m5(test)
    print(a.shape)
    print(b.shape)
    print(c.shape)
    print(d.shape)
    print(e.shape)
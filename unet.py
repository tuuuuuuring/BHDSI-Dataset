import torch
import torch.nn as nn
from torchvision.models import ResNet
import torch.nn.functional as f


"""
    构造下采样模块--左边特征提取基础模块    
"""


class conv_block_(nn.Module):
    """
    Convolution Block No BN
    """

    def __init__(self, in_ch, out_ch, dropout_rate=0.5):
        super(conv_block_, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            # 在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        y = self.conv(x)
        return y


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch, dropout_rate=0.5):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            # 在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        y = self.conv(x)
        return y


"""
    构造上采样模块--右边特征融合基础模块    
"""


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch, ks=2):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, ks, stride=2, padding=0)
            # nn.Upsample(scale_factor=2),
            # nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


"""
    resstructure
"""


class BasicBlock(nn.Module):
    expansion = 1  # 残差结构中主分支所采用的卷积核的个数是否发生变化。对于浅层网络，每个残差结构的第一层和第二层卷积核个数一样，故是1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, dropout_rate=0.1):  # 添加 dropout_rate 参数
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)  # 使用 bn 层时不使用 bias
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)  # 添加 Dropout 层
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)  # 实/虚线残差结构主分支中第二层 stride 都为1
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.dropout2 = nn.Dropout(dropout_rate)  # 添加 Dropout 层
        self.downsample = downsample  # 默认是 None

    def forward(self, x):
        identity = x  # 捷径分支的输出值
        if self.downsample is not None:  # 对应虚线残差结构
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)  # Dropout after first ReLU

        out = self.conv2(out)
        out = self.bn2(out)  # 这里不经过 ReLU 激活函数
        out = self.dropout2(out)  # Dropout before adding the identity

        out += identity
        out = self.relu(out)

        return out


# 定义一个函数来创建 downsample
def make_downsample(in_channel, out_channel, stride):
    if stride != 1 or in_channel != out_channel:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        return downsample
    else:
        return None


def generate_coordinates(t):
    N, _, H, W = t.size()
    # 生成 x 和 y 坐标网格
    y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')

    # 将 x 和 y 坐标合并到一个张量中，形状为 (2, H, W)
    coords = torch.stack((x_coords, y_coords), dim=0).float()

    # 将坐标扩展到批次维度，形状变为 (N, 2, H, W)
    coords = coords.unsqueeze(0).expand(N, -1, -1, -1)

    return coords


"""
    模型主架构
"""


class UNet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, feature_scale=4, in_ch=2, out_ch=1, block='cbr'):
        super(UNet, self).__init__()

        # 卷积参数设置
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        filters = [int(x / feature_scale) for x in filters]

        # 最大池化层
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 左边特征提取卷积层
        ''' which block '''
        if block == 'cbr':
            self.Conv1 = conv_block(in_ch, filters[0])
            self.Conv2 = conv_block(filters[0], filters[1])
            self.Conv3 = conv_block(filters[1], filters[2])
            self.Conv4 = conv_block(filters[2], filters[3])
            self.Conv5 = conv_block(filters[3], filters[4])
        elif block == 'bt':# bottleneck
            self.Conv1 = BasicBlock(in_ch, filters[0], downsample=make_downsample(in_ch, filters[0], 1))
            self.Conv2 = BasicBlock(filters[0], filters[1], downsample=make_downsample(filters[0], filters[1], 1))
            self.Conv3 = BasicBlock(filters[1], filters[2], downsample=make_downsample(filters[1], filters[2], 1))
            self.Conv4 = BasicBlock(filters[2], filters[3], downsample=make_downsample(filters[2], filters[3], 1))
            self.Conv5 = BasicBlock(filters[3], filters[4], downsample=make_downsample(filters[3], filters[4], 1))
        else:
            self.Conv1 = conv_block(in_ch, filters[0])
            self.Conv2 = conv_block(filters[0], filters[1])
            self.Conv3 = conv_block(filters[1], filters[2])
            self.Conv4 = conv_block(filters[2], filters[3])
            self.Conv5 = conv_block(filters[3], filters[4])

        # 右边特征融合反卷积层
        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)  # 将e4特征图与d5特征图横向拼接

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)  # 将e3特征图与d4特征图横向拼接
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)  # 将e2特征图与d3特征图横向拼接
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)  # 将e1特征图与d1特征图横向拼接
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out


class left_UNet(nn.Module):
    def __init__(self, feature_scale=4, in_ch=2, block='cbr'):
        super(left_UNet, self).__init__()

        # 卷积参数设置
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        filters = [int(x / feature_scale) for x in filters]

        # 最大池化层
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 左边特征提取卷积层
        ''' which block '''
        if block == 'cbr':
            self.Conv1 = conv_block(in_ch, filters[0])
            self.Conv2 = conv_block(filters[0], filters[1])
            self.Conv3 = conv_block(filters[1], filters[2])
            self.Conv4 = conv_block(filters[2], filters[3])
            self.Conv5 = conv_block(filters[3], filters[4])
        elif block == 'bt':  # bottleneck
            self.Conv1 = BasicBlock(in_ch, filters[0], downsample=make_downsample(in_ch, filters[0], 1))
            self.Conv2 = BasicBlock(filters[0], filters[1], downsample=make_downsample(filters[0], filters[1], 1))
            self.Conv3 = BasicBlock(filters[1], filters[2], downsample=make_downsample(filters[1], filters[2], 1))
            self.Conv4 = BasicBlock(filters[2], filters[3], downsample=make_downsample(filters[2], filters[3], 1))
            self.Conv5 = BasicBlock(filters[3], filters[4], downsample=make_downsample(filters[3], filters[4], 1))
        else:
            self.Conv1 = conv_block(in_ch, filters[0])
            self.Conv2 = conv_block(filters[0], filters[1])
            self.Conv3 = conv_block(filters[1], filters[2])
            self.Conv4 = conv_block(filters[2], filters[3])
            self.Conv5 = conv_block(filters[3], filters[4])

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        return e1, e2, e3, e4, e5


class right_UNet(nn.Module):
    def __init__(self, out_ch, feature_scale=4, dropout_rate=0.2):
        super(right_UNet, self).__init__()
        # 卷积参数设置
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        filters = [int(x / feature_scale) for x in filters]
        # 右边特征融合反卷积层
        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block_(filters[4], filters[3], dropout_rate=dropout_rate)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block_(filters[3], filters[2], dropout_rate=dropout_rate)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block_(filters[2], filters[1], dropout_rate=dropout_rate)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block_(filters[1], filters[0], dropout_rate=dropout_rate)

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.conv_pink = nn.ModuleList()
        self.conv_blue = nn.ModuleList()
        self.convblank = nn.ModuleList()
        for i in range(4):
            self.conv_pink.append(up_conv(filters[4-i], filters[3-i]))
            self.conv_blue.append(nn.Conv2d(filters[3-i], filters[3-i], 1))
            self.convblank.append(nn.Conv2d(filters[4-i], 2, kernel_size=3, stride=1, padding=1))

    def forward(self, e1, e2, e3, e4, e5):
        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)  # 将e4特征图与d5特征图横向拼接
        d5 = self.Up_conv5(d5)
        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)  # 将e3特征图与d4特征图横向拼接
        d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)  # 将e2特征图与d3特征图横向拼接
        d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)  # 将e1特征图与d1特征图横向拼接
        d2 = self.Up_conv2(d2)

        # te5 = self.conv_pink[0](e5)
        # te4 = self.conv_blue[0](e4)
        # s5 = torch.cat((te4, te5), dim=1)
        # s5 = self.convblank[0](s5).permute(0, 2, 3, 1)
        # far5 = f.grid_sample(te5, s5,align_corners=True)
        # d5 = te4 + far5
        #
        # td5 = self.conv_pink[1](d5)
        # te3 = self.conv_blue[1](e3)
        # s4 = torch.cat((te3, td5), dim=1)
        # s4 = self.convblank[1](s4).permute(0, 2, 3, 1)
        # far4 = f.grid_sample(td5, s4,align_corners=True)
        # d4 = te3 + far4
        #
        # td4 = self.conv_pink[2](d4)
        # te2 = self.conv_blue[2](e2)
        # s3 = torch.cat((te2, td4), dim=1)
        # s3 = self.convblank[2](s3).permute(0, 2, 3, 1)
        # far3 = f.grid_sample(td4, s3,align_corners=True)
        # d3 = te2 + far3
        #
        # td3 = self.conv_pink[3](d3)
        # te1 = self.conv_blue[3](e1)
        # s2 = torch.cat((te1, td3), dim=1)
        # s2 = self.convblank[3](s2).permute(0, 2, 3, 1)
        # far2 = f.grid_sample(td3, s2,align_corners=True)
        # d2 = te1 + far2

        out = self.Conv(d2)
        return out


class left_UNet4(nn.Module):
    def __init__(self, feature_scale=4, in_ch=2, block='cbr', dropout_rate=0.2):
        super(left_UNet4, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        filters = [int(x / feature_scale) for x in filters]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        if block == 'cbr':
            self.Conv1 = conv_block(in_ch, filters[0], dropout_rate=dropout_rate)
            self.Conv2 = conv_block(filters[0], filters[1], dropout_rate=dropout_rate)
            self.Conv3 = conv_block(filters[1], filters[2], dropout_rate=dropout_rate)
            self.Conv4 = conv_block(filters[2], filters[3], dropout_rate=dropout_rate)
        elif block == 'bt':
            self.Conv1 = BasicBlock(in_ch, filters[0], downsample=make_downsample(in_ch, filters[0], 1), dropout_rate=dropout_rate)
            self.Conv2 = BasicBlock(filters[0], filters[1], downsample=make_downsample(filters[0], filters[1], 1), dropout_rate=dropout_rate)
            self.Conv3 = BasicBlock(filters[1], filters[2], downsample=make_downsample(filters[1], filters[2], 1), dropout_rate=dropout_rate)
            self.Conv4 = BasicBlock(filters[2], filters[3], downsample=make_downsample(filters[2], filters[3], 1), dropout_rate=dropout_rate)
        else:
            self.Conv1 = conv_block(in_ch, filters[0], dropout_rate=dropout_rate)
            self.Conv2 = conv_block(filters[0], filters[1], dropout_rate=dropout_rate)
            self.Conv3 = conv_block(filters[1], filters[2], dropout_rate=dropout_rate)
            self.Conv4 = conv_block(filters[2], filters[3], dropout_rate=dropout_rate)

    def forward(self, x):
        e1 = self.Conv1(x)
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        return e1, e2, e3, e4


class right_UNet4(nn.Module):
    def __init__(self, feature_scale=4, dropout_rate=0.2,out_ch=1):
        super(right_UNet4, self).__init__()
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        filters = [int(x / feature_scale) for x in filters]

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block_(filters[4], filters[3], dropout_rate=dropout_rate)
        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block_(filters[3], filters[2], dropout_rate=dropout_rate)
        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block_(filters[2], filters[1], dropout_rate=dropout_rate)
        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block_(filters[1], filters[0], dropout_rate=dropout_rate)
        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.conv_pink = nn.ModuleList()
        self.conv_blue = nn.ModuleList()
        self.convblank = nn.ModuleList()
        for i in range(4):
            self.conv_pink.append(up_conv(filters[4 - i], filters[3 - i]))
            self.conv_blue.append(nn.Conv2d(filters[3 - i], filters[3 - i], 1))
            self.convblank.append(nn.Conv2d(filters[4 - i], 2, kernel_size=3, stride=1, padding=1))

    def forward(self, e1, e2, e3, e4, e5):
        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        # te5 = self.conv_pink[0](e5)
        # te4 = self.conv_blue[0](e4)
        # s5 = torch.cat((te4, te5), dim=1)
        # s5 = self.convblank[0](s5).permute(0, 2, 3, 1)
        # far5 = f.grid_sample(te5, s5,align_corners=True)
        # d5 = te4 + far5
        # td5 = self.conv_pink[1](d5)
        # te3 = self.conv_blue[1](e3)
        # s4 = torch.cat((te3, td5), dim=1)
        # s4 = self.convblank[1](s4).permute(0, 2, 3, 1)
        # far4 = f.grid_sample(td5, s4,align_corners=True)
        # d4 = te3 + far4
        # td4 = self.conv_pink[2](d4)
        # te2 = self.conv_blue[2](e2)
        # s3 = torch.cat((te2, td4), dim=1)
        # s3 = self.convblank[2](s3).permute(0, 2, 3, 1)
        # far3 = f.grid_sample(td4, s3,align_corners=True)
        # d3 = te2 + far3
        # td3 = self.conv_pink[3](d3)
        # te1 = self.conv_blue[3](e1)
        # s2 = torch.cat((te1, td3), dim=1)
        # s2 = self.convblank[3](s2).permute(0, 2, 3, 1)
        # far2 = f.grid_sample(td3, s2,align_corners=True)
        # d2 = te1 + far2

        out = self.Conv(d2)
        return out


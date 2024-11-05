import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from newcrfs.networks.NewCRFDepth import NewCRFDepthwithuper
from sffm import SFFM, SEF, SKF, SKFA
from unet import left_UNet, right_UNet, left_UNet4, BasicBlock, make_downsample, conv_block, right_UNet4


class ConcatLayer(nn.Module):
    def __init__(self):
        super(ConcatLayer, self).__init__()

    def forward(self, x1,x2):

        return torch.cat((x1, x2), dim=1)


class little_mdf(nn.Module):

    def __init__(self, block='none', fusion='none'):
        super(little_mdf, self).__init__()
        # 第一个UNet网络，输入通道数为2，输出通道数为1
        self.lunet1 = left_UNet4(in_ch=2, block=block)
        self.runet1 = right_UNet4()

        # 第二个UNet网络，输入通道数为4，输出通道数为1
        self.lunet2 = left_UNet4(in_ch=4, block=block)
        self.runet2 = right_UNet(out_ch=2)

        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 1x1卷积合并层，将两个UNet网络的输出合并为一个
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.last_step = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

        if block == 'bt':
            self.Conv = BasicBlock(128, 256, downsample=make_downsample(128, 256, 1))
        elif block == 'cbr':
            self.Conv = BasicBlock(128, 256)
        else:
            self.Conv = BasicBlock(128, 256)
        if fusion == 'sffm':
            self.fusi4 = SFFM(128)
            self.fusi5 = SFFM()
        elif fusion == 'senet':
            self.fusi4 = SEF(128)
            self.fusi5 = SEF(256)
        elif fusion == 'sknet':
            self.fusi4 = SKF(256, 256)
            self.fusi5 = SKF(512, 512)
        else:
            self.fusi = ConcatLayer()

    def forward(self, x1, x2):

        # 将两个UNet网络的输出进行合并
        s1e1, s1e2, s1e3, s1e4 = self.lunet1(x1)
        s2e1, s2e2, s2e3, s2e4 = self.lunet2(x2)
        e4up = self.fusi4(s1e4, s2e4)
        e4down = self.fusi4(s2e4, s1e4)

        s1e5 = self.Maxpool4(e4up)
        s2e5 = self.Maxpool4(e4down)
        s1e5 = self.Conv(s1e5)
        s2e5 = self.Conv(s2e5)

        e5up = self.fusi5(s1e5, s2e5)
        e5down = self.fusi5(s2e5, s1e5)

        out1 = self.runet1(s1e1, s1e2, s1e3, s1e4, e5up)
        out2 = self.runet2(s2e1, s2e2, s2e3, s2e4, e5down)
        merged_output = torch.cat((out1, out2), dim=1)
        # 使用1x1卷积层进行合并
        merged_output = self.last_step(merged_output)
        if self.training:
            return out2, merged_output
        else:
            return merged_output


class separate_way(nn.Module):

    def __init__(self, block='none', fusion='none'):
        super(separate_way, self).__init__()
        # 第一个UNet网络，输入通道数为2，输出通道数为1
        self.lunet1 = left_UNet(in_ch=2, block=block)
        self.runet1 = right_UNet(out_ch=1)

        # 第二个UNet网络，输入通道数为4，输出通道数为1
        self.lunet2 = left_UNet(in_ch=4, block=block)
        self.runet2 = right_UNet(out_ch=2)

        # 1x1卷积合并层，将两个UNet网络的输出合并为一个
        self.conv_merge = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
        self.conv_reduce_channels = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        if fusion == 'sffm':
            self.fusi = SFFM()
        elif fusion == 'senet':
            self.fusi = SEF()
        elif fusion == 'sknet':
            self.fusi = SKF()
        else:
            self.fusi = ConcatLayer()

    def forward(self, x1, x2):

        # 将两个UNet网络的输出进行合并
        s1e1, s1e2, s1e3, s1e4, s1e5 = self.lunet1(x1)
        s2e1, s2e2, s2e3, s2e4, s2e5 = self.lunet2(x2)
        e5 = self.fusi(s1e5, s2e5)
        _, b, _, _ = e5.size()
        if b != 256:
            e5 = self.conv_reduce_channels(e5)
        s1e5 = torch.cat((e5, s1e5), dim=1)
        s2e5 = torch.cat((e5, s2e5), dim=1)
        s1e5 = self.conv_reduce_channels(s1e5)
        s2e5 = self.conv_reduce_channels(s2e5)
        out1 = self.runet1(s1e1, s1e2, s1e3, s1e4, s1e5)
        out2 = self.runet2(s2e1, s2e2, s2e3, s2e4, s2e5)
        merged_output = torch.cat((out1, out2), dim=1)
        # 使用1x1卷积层进行合并
        merged_output = self.conv_merge(merged_output)
        if self.training:
            return out1, out2, merged_output
        else:
            return merged_output


class original(nn.Module):

    def __init__(self, block='none', fusion='none',in_ch1=2,in_ch2=4):
        super(original, self).__init__()
        # 第一个UNet网络，输入通道数为2，输出通道数为1
        self.lunet1 = left_UNet4(in_ch=in_ch1, block=block)
        self.runet1 = right_UNet4()

        # 第二个UNet网络，输入通道数为4，输出通道数为1
        self.lunet2 = left_UNet4(in_ch=in_ch2, block=block)
        self.runet2 = right_UNet(out_ch=3)

        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 1x1卷积合并层，将两个UNet网络的输出合并为一个
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.last_step = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

        if block == 'bt':
            self.Conv = BasicBlock(128, 256, downsample=make_downsample(128, 256, 1))
        elif block == 'cbr':
            self.Conv = BasicBlock(128, 256)
        else:
            self.Conv = BasicBlock(128, 256)
        if fusion == 'sffm':
            self.fusi4 = SFFM(128)
            self.fusi5 = SFFM()
        elif fusion == 'senet':
            self.fusi4 = SEF(128)
            self.fusi5 = SEF(256)
        elif fusion == 'sknet':
            self.fusi4 = SKF(256,256)
            self.fusi5 = SKF(512,512 )
        else:
            self.fusi = ConcatLayer()

    def forward(self, x1, x2):

        # 将两个UNet网络的输出进行合并
        s1e1, s1e2, s1e3, s1e4 = self.lunet1(x1)
        s2e1, s2e2, s2e3, s2e4 = self.lunet2(x2)
        e4up = self.fusi4(s1e4, s2e4)
        e4down = self.fusi4(s2e4, s1e4)

        s1e5 = self.Maxpool4(e4up)
        s2e5 = self.Maxpool4(e4down)
        s1e5 = self.Conv(s1e5)
        s2e5 = self.Conv(s2e5)

        e5up = self.fusi5(s1e5, s2e5)
        e5down = self.fusi5(s2e5, s1e5)

        out1 = self.runet1(s1e1, s1e2, s1e3, s1e4, e5up)
        out2 = self.runet2(s2e1, s2e2, s2e3, s2e4, e5down)
        merged_output = torch.cat((out1, out2), dim=1)
        # 使用1x1卷积层进行合并
        merged_output = self.last_step(merged_output)
        if self.training:
            return out2, merged_output
        else:
            return merged_output


class original_for_dfc(nn.Module):

    def __init__(self, block='none', fusion='none'):
        super(original_for_dfc, self).__init__()
        # 第一个UNet网络，输入通道数为2，输出通道数为1
        self.lunet1 = left_UNet4(in_ch=1, block=block)
        self.runet1 = right_UNet4()

        # 第二个UNet网络，输入通道数为4，输出通道数为1
        self.lunet2 = left_UNet4(in_ch=3, block=block)
        self.runet2 = right_UNet4()

        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 1x1卷积合并层，将两个UNet网络的输出合并为一个
        self.conv_merge = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        self.conv_reduce_channels = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv_reduce_channels2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        if block == 'bt':
            self.Conv = BasicBlock(128, 256, downsample=make_downsample(128, 256, 1))
        elif block == 'cbr':
            self.Conv = conv_block(128, 256)
        else:
            self.Conv = conv_block(128, 256)
        if fusion == 'sffm':
            self.fusi4 = SFFM(128)
            self.fusi5 = SFFM()
        elif fusion == 'senet':
            self.fusi4 = SEF(128)
            self.fusi5 = SEF(256)
        elif fusion == 'sknet':
            self.fusi4 = SKF(256,256)
            self.fusi5 = SKF(512,512 )
        else:
            self.fusi = ConcatLayer()

    def forward(self, x1, x2):

        # 将两个UNet网络的输出进行合并
        s1e1, s1e2, s1e3, s1e4 = self.lunet1(x1)
        s2e1, s2e2, s2e3, s2e4 = self.lunet2(x2)
        e4up = self.fusi4(s1e4, s2e4)
        e4down = self.fusi4(s2e4, s1e4)
        s1e4 = torch.cat((e4up, s1e4), dim=1)
        s2e4 = torch.cat((e4down, s2e4), dim=1)
        s1e4 = self.conv_reduce_channels2(s1e4)
        s2e4 = self.conv_reduce_channels2(s2e4)

        s1e5 = self.Maxpool4(s1e4)
        s2e5 = self.Maxpool4(s2e4)
        s1e5 = self.Conv(s1e5)
        s2e5 = self.Conv(s2e5)

        e5up = self.fusi5(s1e5, s2e5)
        e5down = self.fusi5(s2e5, s1e5)
        s1e5 = torch.cat((e5up, s1e5), dim=1)
        s2e5 = torch.cat((e5down, s2e5), dim=1)
        s1e5 = self.conv_reduce_channels(s1e5)
        s2e5 = self.conv_reduce_channels(s2e5)
        out1 = self.runet1(s1e1, s1e2, s1e3, s1e4, s1e5)
        out2 = self.runet2(s2e1, s2e2, s2e3, s2e4, s2e5)
        merged_output = torch.cat((out1, out2), dim=1)
        # 使用1x1卷积层进行合并
        merged_output = self.conv_merge(merged_output)
        return merged_output


if __name__ == '__main__':
    input1 = torch.ones(2, 2 ,256,256)
    input2 = torch.randn(2, 3 ,256,256)
    model = NewCRFDepthwithuper(version='tiny07')

    off,ts = model(input2)
    print(off.shape)

    print(ts.shape)
    # y_true = torch.randn(16, 1 ,256 ,256)

#     y_truebi = y_truebi.long().view(-1)
#     print(y_truebi.shape)
#     lmf = separate_way2(block='bt')
#     output1, output2, output3, mergeout = lmf(input1, input2)
#     print(output1.shape)
#     print(output2.shape)
#     print(output3.shape)
#     print(mergeout.shape)
#     ypred3 = output2.transpose(1, 2).transpose(2, 3).contiguous().view(-1,2)
#     print(ypred3.shape)
#     a = torch.randn(1048576)
#     a = a.long()
#     a = torch.where(a > 0, torch.ones_like(a), torch.zeros_like(a))
#     b = torch.randn(1048576,2)
#     print(a.shape)
#     print(b)
#     loss=torch.nn.functional.cross_entropy(ypred3, y_truebi, reduction='mean')
#     print(loss)

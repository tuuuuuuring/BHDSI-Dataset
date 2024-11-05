import torch
import torch.nn as nn
from funcs.dptcp import get_readout_oper, Transpose, _make_scratch, _make_fusion_block, Interpolate


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=4, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class VisionTransformerEncoder(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_dim=3072, dropout=0.1):
        super(VisionTransformerEncoder, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        B, N, _ = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1 + num_patches, embed_dim]
        x = x + self.pos_embed[:, :(N + 1)]
        x = self.pos_drop(x)

        outputs = []
        for i in range(len(self.encoder.layers)):
            residual = x
            x = self.encoder.layers[i](x)
            x=x+residual
            if (i+1)%3==0:
                outputs.append(x)

        return outputs


class post_process(nn.Module):
    def __init__(self,vit_features=768,size=[256,256],features=[96, 192, 384, 768]):
        super(post_process, self).__init__()
        readout_oper = get_readout_oper(vit_features, features, 'ignore', start_index=1)
        self.postprocess1 = nn.Sequential(
            readout_oper[0],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
            nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[0],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=features[0],
                out_channels=features[0],
                kernel_size=4,
                stride=4,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            ),
        )
        self.postprocess2 = nn.Sequential(
            readout_oper[1],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
            nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[1],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=features[1],
                out_channels=features[1],
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            ),
        )
        self.postprocess3 = nn.Sequential(
            readout_oper[2],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
            nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[2],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        self.postprocess4 = nn.Sequential(
            readout_oper[3],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
            nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[3],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Conv2d(
                in_channels=features[3],
                out_channels=features[3],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )

    def forward(self,x):
        x1 = self.postprocess1(x[0])
        x2 = self.postprocess2(x[1])
        x3 = self.postprocess3(x[2])
        x4 = self.postprocess4(x[3])
        return [x1,x2,x3,x4]


class fusions(nn.Module):
    def __init__(self,features=[96, 192, 384, 768],channals=256,use_bn = True):
        super(fusions, self).__init__()
        self.scratch = _make_scratch(features, channals)
        self.refinenet1 = _make_fusion_block(channals, use_bn)
        self.refinenet2 = _make_fusion_block(channals, use_bn)
        self.refinenet3 = _make_fusion_block(channals, use_bn)
        self.refinenet4 = _make_fusion_block(channals, use_bn)

    def forward(self,x):
        sc = [self.scratch.layer1_rn, self.scratch.layer2_rn, self.scratch.layer3_rn, self.scratch.layer4_rn]
        for i, scs in enumerate(sc):
            x[i] = scs(x[i])
        path_4 = self.refinenet4(x[3])
        path_3 = self.refinenet3(path_4, x[2])
        path_2 = self.refinenet2(path_3, x[1])
        path_1 = self.refinenet1(path_2, x[0])
        return path_1


class DPT(nn.Module):
    def __init__(self,imsz=256,inchnl=3):
        super(DPT, self).__init__()
        self.vits = VisionTransformerEncoder(img_size=imsz,in_chans=inchnl)
        self.reassemble = post_process(size=[imsz,imsz])
        self.fusions = fusions()
        self.tem = nn.Sequential(
            nn.Conv2d(inchnl,16,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.head=nn.Sequential(
            nn.Conv2d(256, 256 // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # nn.ConvTranspose2d(128, 128 // 2, 2, stride=2, padding=0),
            # Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        self.upend = nn.ConvTranspose2d(64, 32, 2, stride=2, padding=0)
        self.lastconv = nn.Conv2d(32,1,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        bs = self.tem(x)
        X = self.vits(x)
        X = self.reassemble(X)
        y = self.fusions(X)
        y = self.head(y)
        y = torch.cat((bs,y),dim=1)
        y = self.upend(y)
        y = self.lastconv(y)
        return y


# Example usage
if __name__ == '__main__':
    # 初始化模型
    model = DPT(imsz=512,inchnl=3)

    # 创建示例输入数据和目标数据
    input_data = torch.randn(2, 3, 512, 512)  # 示例输入
    target_data = torch.randn(1, 1, 256, 256)  # 示例目标
    output=model(input_data)
    print(output.shape)
    sdfcv = torch.isnan(output).any()

    print('dfshfs')

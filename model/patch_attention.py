import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from model.position_encoding import build_position_encoding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)

    def forward(self, x):
        out = self.deconv(x)
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H * 2 * W * 2 * self.in_channel * self.out_channel * 2 * 2
        print("Upsample:{%.2f}" % (flops / 1e9))
        return flops


class TransformerBlock(nn.Module):
    def __init__(self, patch_size, dim, channels, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.patch_size = patch_size
        patch_dim = channels * patch_size * patch_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim)
        )
        self.pos_embedding = build_position_encoding('sine', dim)
        self.dropout = nn.Dropout(dropout)

        self.head = num_heads
        inner_dim = int(mlp_ratio * dim)
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.img0_to_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.img1_to_qv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.mlp = Mlp(in_features=dim, hidden_features=inner_dim, out_features=patch_dim, drop=dropout)
        self.mlp2 = Mlp(in_features=dim, hidden_features=inner_dim, out_features=patch_dim, drop=dropout)
        self.out = nn.Conv2d(dim, channels, 3, 1, 1)

    def forward(self, img0, img1, mask=None):
        B, C, H, W = img0.shape
        # img0 : B, 256, 64, 64
        img0 = self.to_patch_embedding(img0)
        img1 = self.to_patch_embedding(img1)
        b, n, c = img0.shape

        img0 += self.pos_embedding(img0)
        img0 = self.dropout(img0)
        img1 += self.pos_embedding(img1)
        img1 = self.dropout(img1)

        img0_kv = self.img0_to_kv(img0).reshape(b, n, 2, self.head, c // self.head).permute(2, 0, 3, 1, 4)
        img0_k, img0_v = img0_kv[0], img0_kv[1]

        img1_qv = self.img1_to_qv(img1).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        img1_q, img1_v = img1_qv[0], img1_qv[1]
        img1_q = img1_q.reshape(b, n, self.head, c // self.head).permute(0, 2, 1, 3)

        img1_dots = img1_q @ img0_k.transpose(-2, -1).contiguous() * self.scale
        mask = mask.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        mask = mask.contiguous().view(B, 1, -1, self.patch_size, self.patch_size)
        mask = mask.sum(dim=[3, 4])
        mask[mask > 0] = 1
        mask = mask.unsqueeze(1).repeat(1, self.head, n, 1)
        mask = mask.masked_fill(mask == 0, float(-100.0)).masked_fill(mask == 1, float(0.0))
        img1_dots = img1_dots + mask

        img1_attn = self.softmax(img1_dots)
        img1_out = (img1_attn @ img0_v).transpose(1, 2).reshape(b, n, c)
        img1_res = self.mlp2(img1_out)
        img1_res = rearrange(img1_res, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size,
                             h=H // self.patch_size)
        return img1_res


class DetailsSup(nn.Module):
    def __init__(self, patch_size, dim, channels, heads=8):
        super(DetailsSup, self).__init__()
        self.transformer_block = TransformerBlock(patch_size=patch_size, dim=dim, channels=channels, num_heads=heads)
        self.upsample = Upsample(channels * 2, channels)
        self.fusion_block = nn.Sequential(
            nn.Conv2d(channels * 5, channels, 3, 1, 1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(channels, channels * 4, 3, 1, 1)
        )

    def forward(self, imgt, img0, img1, f_up, corr0, corr1):
        imgt_from_img0 = imgt + self.transformer_block(img0, imgt, corr0)
        imgt_from_img1 = imgt + self.transformer_block(img1, imgt, corr1)
        imgt = self.upsample(torch.cat((imgt_from_img0, imgt_from_img1), 1))
        out = self.fusion_block(torch.cat((imgt, f_up), 1))
        return out

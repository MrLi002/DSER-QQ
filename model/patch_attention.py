import torch
import torch.nn as nn
import torch.nn.functional as F
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
        """Gather both K/V and Q from HCS patches, then scatter a local residual.

        img0 is the supporting keyframe and img1 is the synthesized feature.
        mask is already reversed by MyNet, so the frame-0 branch receives H1.
        Empty selections produce zero residuals without running a softmax.
        """
        if img0.shape != img1.shape or img0.ndim != 4:
            raise ValueError("support and query features must have identical [B,C,H,W] shapes")
        B, C, H, W = img0.shape
        if H % self.patch_size or W % self.patch_size:
            raise ValueError("feature height and width must be divisible by patch_size")
        if mask is None:
            mask = img1.new_ones((B, 1, H, W))
        if mask.shape != (B, 1, H, W):
            raise ValueError("mask must have shape [B,1,H,W] matching the features")
        selected = F.max_pool2d(
            (mask > 0).float(), kernel_size=self.patch_size, stride=self.patch_size
        ).flatten(1).bool()
        if not selected.any():
            return img1 * 0.0

        support_tokens = self.to_patch_embedding(img0)
        query_tokens = self.to_patch_embedding(img1)
        _, n, c = query_tokens.shape
        positions = self.pos_embedding(query_tokens).to(dtype=query_tokens.dtype)
        support_tokens = self.dropout(support_tokens + positions)
        query_tokens = self.dropout(query_tokens + positions)

        residuals = []
        for batch_index in range(B):
            indices = selected[batch_index].nonzero(as_tuple=True)[0]
            if indices.numel() == 0:
                residuals.append(img1[batch_index:batch_index + 1] * 0.0)
                continue
            support = support_tokens[batch_index:batch_index + 1].index_select(1, indices)
            query = query_tokens[batch_index:batch_index + 1].index_select(1, indices)
            count = indices.numel()
            kv = self.img0_to_kv(support).reshape(
                1, count, 2, self.head, c // self.head
            ).permute(2, 0, 3, 1, 4)
            key, value = kv[0], kv[1]
            # Keep the released Q/V projection shape for checkpoint compatibility.
            query = self.img1_to_qv(query).chunk(2, dim=-1)[0]
            query = query.reshape(1, count, self.head, c // self.head).transpose(1, 2)
            attention = self.softmax((query @ key.transpose(-2, -1)) * self.scale)
            attended = (attention @ value).transpose(1, 2).reshape(1, count, c)
            local_residual = self.mlp2(attended)
            full_residual = local_residual.new_zeros(
                (1, n, C * self.patch_size * self.patch_size)
            ).index_copy(1, indices, local_residual)
            residuals.append(rearrange(
                full_residual, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                p1=self.patch_size, p2=self.patch_size, h=H // self.patch_size,
            ))
        return torch.cat(residuals, dim=0)


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
        # The global decoder's upsampled feature is the baseline outside H0 U H1.
        repair_mask = ((corr0 > 0) | (corr1 > 0)).float()
        if not repair_mask.any():
            return f_up
        imgt_from_img0 = imgt + self.transformer_block(img0, imgt, corr0)
        imgt_from_img1 = imgt + self.transformer_block(img1, imgt, corr1)
        imgt = self.upsample(torch.cat((imgt_from_img0, imgt_from_img1), 1))
        out = self.fusion_block(torch.cat((imgt, f_up), 1))
        repair_mask = F.max_pool2d(
            repair_mask,
            kernel_size=self.transformer_block.patch_size,
            stride=self.transformer_block.patch_size,
        )
        repair_mask = F.interpolate(repair_mask, size=f_up.shape[-2:], mode='nearest').bool()
        # Apply the fused candidate only inside selected patches. This also
        # prevents convolutional biases and receptive fields leaking repairs.
        return torch.where(repair_mask, out, f_up)


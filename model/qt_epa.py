"""QT-EPA: query-time-guided event temporal matching for DSER++.

Refines deformable sampling offsets while preserving pretrained baseline output
at initialization (zero-initialized offset residual projection).
"""
import torch
from torch import nn
import torch.nn.functional as F


class QTEPA(nn.Module):
    def __init__(self, feature_channels, bins=8, hidden=32, radius=2, max_offset=2.0):
        super().__init__()
        if bins < 2:
            raise ValueError("QT-EPA requires bins >= 2")
        self.bins = bins
        self.radius = radius
        self.max_offset = max_offset
        self.early = nn.Sequential(nn.Conv2d(bins, hidden, 3, padding=1), nn.ReLU(inplace=True),
                                   nn.Conv2d(hidden, hidden, 3, padding=1))
        self.late = nn.Sequential(nn.Conv2d(bins, hidden, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(hidden, hidden, 3, padding=1))
        self.time_mlp = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        corr_channels = (2 * radius + 1) ** 2
        self.fuse = nn.Sequential(
            nn.Conv2d(feature_channels + corr_channels + hidden + 18 + 2, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True))
        self.offset_head = nn.Conv2d(hidden, 18, 3, padding=1)
        self.gate_head = nn.Conv2d(hidden, 1, 3, padding=1)
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)

    def forward(self, feature, voxel, query_time, base_offset):
        b, _, h, w = feature.shape
        if voxel.size(1) != self.bins:
            raise ValueError(f"Expected {self.bins} voxel channels, got {voxel.size(1)}")
        if base_offset.shape != (b, 18, h, w):
            raise ValueError("base_offset must be [B,18,h,w] matching feature spatial size")
        if not isinstance(query_time, torch.Tensor):
            query_time = torch.as_tensor(query_time, device=feature.device, dtype=feature.dtype)
        query_time = query_time.to(device=feature.device, dtype=feature.dtype).reshape(-1, 1)
        if query_time.shape[0] == 1 and b > 1:
            query_time = query_time.expand(b, -1)
        if query_time.shape[0] != b:
            raise ValueError("query_time batch size mismatch")
        # Cast expensive matching to fp32 under autocast for stable normalization.
        with torch.cuda.amp.autocast(enabled=False):
            v = F.interpolate(voxel.float(), (h, w), mode="area")
            split = self.bins // 2
            ev_early = torch.cat((v[:, :split], torch.zeros_like(v[:, split:])), dim=1)
            ev_late = torch.cat((torch.zeros_like(v[:, :split]), v[:, split:]), dim=1)
            e = F.normalize(self.early(ev_early), p=2, dim=1, eps=1e-6)
            l = F.normalize(self.late(ev_late), p=2, dim=1, eps=1e-6)
            k = 2 * self.radius + 1
            patches = F.unfold(l, kernel_size=k, padding=self.radius)
            patches = patches.reshape(b, e.size(1), k * k, h, w)
            correlation = (e.unsqueeze(2) * patches).sum(1)
            # Activity/difference channels help suppress matches with no events.
            activity = v.abs().mean(1, keepdim=True)
            temporal_difference = (ev_late.abs().sum(1, keepdim=True) -
                                   ev_early.abs().sum(1, keepdim=True))
            temporal_difference = temporal_difference / (v.abs().sum(1, keepdim=True) + 1e-6)
        time_map = self.time_mlp(query_time).view(b, -1, 1, 1).expand(-1, -1, h, w)
        combined = torch.cat((feature, correlation.to(feature.dtype),
                              time_map, base_offset,
                              torch.log1p(activity).to(feature.dtype),
                              temporal_difference.to(feature.dtype)), dim=1)
        state = self.fuse(combined)
        delta = self.max_offset * torch.tanh(self.offset_head(state))
        gate = torch.sigmoid(self.gate_head(state))
        refined = base_offset + gate * delta
        return refined

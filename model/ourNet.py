import argparse

import torch
from torch import nn
from model.blocks import EventRecNet, FeaTNet, DenseBlock, frame_encoder, event_encoder, ref_encoder
from model.trans_sys import Transformer
from util.utils_func import mask_reverse, pyramid_patch_mask


class MyNet(nn.Module):
    def __init__(self, args):
        super(MyNet, self).__init__()
        bins = args.bins
        self.beta = args.beta
        self.mask_patch_size = args.mask_patch_size
        self.event_rec = EventRecNet(2 * bins)
        self.TNet = FeaTNet(bins)
        self.fusion = DenseBlock(64, 3)
        unit_dim = 32
        self.scale = 3
        self.encoder_f = frame_encoder(3, unit_dim // 4)
        self.encoder_e = event_encoder(bins, unit_dim // 2)
        self.encoder_ref = ref_encoder(3, unit_dim // 2)
        self.sys = Transformer(unit_dim * 2)

    def forward(self, imgs, voxels, mask, bins):
        img0 = imgs[:, :3]
        img1 = imgs[:, 6:9]
        v0t = voxels[:, :bins]
        vt1 = voxels[:, bins:bins * 2]
        v1t = voxels[:, bins * 2:bins * 3]
        v01 = torch.cat((v0t, vt1), 1)
        # 第一阶段
        pure_rec = self.event_rec(v01)
        rec = pure_rec * mask

        # 第二阶段
        F0 = self.TNet(img0, img1, v0t, rec)
        F1 = self.TNet(img1, img0, v1t, rec)
        Ft = self.fusion(torch.cat((F0, F1), 1))

        # 第三阶段
        frame_feature = []
        f_frame0 = self.encoder_f(img0)
        f_frame1 = self.encoder_f(img1)

        event_feature = []
        f_event_0t = self.encoder_e(v0t)
        f_event_t1 = self.encoder_e(vt1)

        ref_feature = self.encoder_ref(Ft)
        for idx in range(self.scale):
            frame_feature.append(torch.cat((f_frame0[idx], f_frame1[idx]), dim=1))
            event_feature.append(torch.cat((f_event_0t[idx], f_event_t1[idx]), dim=1))

        corr0_list = pyramid_patch_mask(img0, Ft, mask, patch_size=self.mask_patch_size,
                                        beta=self.beta)
        corr1_list = pyramid_patch_mask(img1, Ft, mask, patch_size=self.mask_patch_size,
                                        beta=self.beta)

        for i in range(len(corr0_list)):
            corr0_list[i], corr1_list[i] = mask_reverse(corr0_list[i], corr1_list[i])

        img_t_third = self.sys(event_feature, frame_feature, ref_feature, f_frame0, f_frame1, corr0_list,
                               corr1_list)

        corr_list = []
        for i in range(3):
            corr_list.append(torch.clamp(corr0_list[i] + corr1_list[i], max=1))

        corr_show = torch.clamp(corr0_list[0] + corr1_list[0], max=1)
        rec = rec.repeat(1, 3, 1, 1)
        pure_rec = pure_rec.repeat(1, 3, 1, 1)

        return pure_rec, rec, Ft, img_t_third, corr_show, corr_list

import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from util.event import EventSequence
from util.utils_func import events_to_channels, process_mask

import torch
from util.voxelization import to_voxel_grid
import warnings

device = torch.device("cuda")
warnings.filterwarnings('ignore')


def psnr_(gt, pred):
    return peak_signal_noise_ratio(gt, pred)


def ssim_(gt, pred):
    multichannel = len(gt.shape) == 3 and gt.shape[2] == 3
    return structural_similarity(gt, pred, data_range=gt.max() - gt.min(), multichannel=multichannel,
                                 gaussian_weights=True, channel_axis=2)


def looking_for_event_index_by_timestamp(path, bins, img_index0, gt_index, img_index1, h, w, frames, real=False,
                                         bsergb=False):
    event_folder = os.path.join(path, 'events')
    if not real:
        timestamps_path = os.path.join(path, 'timestamps.txt')
        with open(timestamps_path, 'r') as f:
            timestamps = f.read().splitlines()

        start_ind = timestamps.index(str(img_index0 / frames))
        gt_ind = timestamps.index(str(gt_index / frames))
        end_ind = timestamps.index(str(img_index1 / frames))

        before_event_paths = [os.path.join(event_folder, str(i).zfill(10) + ".npz") for i in
                              range(start_ind, gt_ind)]
        after_event_paths = [os.path.join(event_folder, str(i).zfill(10) + ".npz") for i in
                             range(gt_ind, end_ind)]
    else:
        before_event_paths = [os.path.join(event_folder, str(i).zfill(6) + ".npz") for i in
                              range(img_index0, gt_index)]
        after_event_paths = [os.path.join(event_folder, str(i).zfill(6) + ".npz") for i in
                             range(gt_index, img_index1)]
    try:
        events_0t = EventSequence.from_npz_files(before_event_paths, h, w, bsergb=bsergb, size=[0, 0, h, w])
    except:
        events_0t = None
    try:
        events_t1 = EventSequence.from_npz_files(after_event_paths, h, w, bsergb=bsergb, size=[0, 0, h, w])
    except:
        events_t1 = None
    if events_0t is None:
        event_0t_voxel = torch.zeros([bins, h, w])
        ec_0t = torch.zeros([2, h, w])
    else:
        event_0t_voxel = to_voxel_grid(events_0t, bins)
        ec_0t = events_to_channels(events_0t)
    if events_t1 is None:
        event_t1_voxel = torch.zeros([bins, h, w])
        event_1t_voxel = torch.zeros([bins, h, w])
        ec_t1 = torch.zeros([2, h, w])
    else:
        event_t1_voxel = to_voxel_grid(events_t1, bins)
        ec_t1 = events_to_channels(events_t1)
        event_1t = events_t1.reverse()
        event_1t_voxel = to_voxel_grid(event_1t, bins)
    event_cnt = torch.cat((ec_0t, ec_t1), 0)
    event_cnt = process_mask(event_cnt)
    event_voxel = torch.cat((event_0t_voxel, event_t1_voxel, event_1t_voxel), 0).unsqueeze(0).to(device,
                                                                                                 non_blocking=True)
    mask = event_cnt.unsqueeze(0).to(device, non_blocking=True)
    return event_voxel, mask


def get_voxel_and_mask(num_bins, events_0t, events_t1, h, w):
    if events_0t is None:
        event_0t_voxel = torch.zeros([num_bins, h, w])
        ec_0t = torch.zeros([2, h, w])
    else:
        event_0t_voxel = to_voxel_grid(events_0t, num_bins)
        ec_0t = events_to_channels(events_0t)
    if events_t1 is None:
        event_t1_voxel = torch.zeros([num_bins, h, w])
        event_1t_voxel = torch.zeros([num_bins, h, w])
        ec_t1 = torch.zeros([2, h, w])
    else:
        event_t1_voxel = to_voxel_grid(events_t1, num_bins)
        ec_t1 = events_to_channels(events_t1)
        event_1t = events_t1.reverse()
        event_1t_voxel = to_voxel_grid(event_1t, num_bins)
    event_cnt = torch.cat((ec_0t, ec_t1), 0)
    event_cnt = process_mask(event_cnt)
    event_voxel = torch.cat((event_0t_voxel, event_t1_voxel, event_1t_voxel), 0).unsqueeze(0).to(device,
                                                                                                 non_blocking=True)
    mask = event_cnt.unsqueeze(0).to(device, non_blocking=True)
    return event_voxel, mask


def get_imgs(img_paths, multi, index):
    img0 = cv2.imread(img_paths[index])
    img1 = cv2.imread(img_paths[index + multi + 1])
    img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
    gray_img0 = torch.zeros_like(img0)
    gray_img1 = torch.zeros_like(img0)
    gt = torch.zeros_like(img0)
    gray_gt = gt
    img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
    imgs = torch.concat((img0, gt, img1, gray_img0, gray_gt, gray_img1), 0).unsqueeze(0).to(device,
                                                                                            non_blocking=True) / 255.
    return imgs


def get_scenes(path):
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

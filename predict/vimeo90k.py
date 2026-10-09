import cv2
import numpy as np
import os
import torch
import glob

from predict.pred_utils import get_voxel_and_mask, psnr_, ssim_
from util.event import EventSequence


def predict_vimeo90k(model, num_bins, device, save_path, data_root, isSave=False, isTestPer=False,
                     saveIndexRange=None):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    image_root = os.path.join(data_root, 'sequences')
    test_fn = os.path.join(data_root, 'tri_testlist.txt')
    with open(test_fn, 'r') as f:
        testlist = f.read().splitlines()
    cnt = int(len(testlist) * 0.5)
    psnr_all = []
    ssim_all = []
    for index in range(len(testlist)):
        if index != "":
            img_folder = os.path.join(image_root, testlist[index])
            event_folder = os.path.join(img_folder, 'events')
            names = os.listdir()
            img_paths = sorted(glob.glob(os.path.join(img_folder, "im[0-9].png")))

            img0 = cv2.imread(img_paths[0])
            img1 = cv2.imread(img_paths[2])
            img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
            img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
            gray_img0 = torch.zeros_like(img0)
            gray_img1 = torch.zeros_like(img0)
            gt = torch.zeros_like(img0)
            gray_gt = gt
            imgs = torch.concat((img0, gt, img1, gray_img0, gray_gt, gray_img1), 0).unsqueeze(0).to(device,
                                                                                                    non_blocking=True) / 255.
            gt = cv2.imread(img_paths[1])
            h, w, _ = gt.shape

            timestamps_path = os.path.join(img_folder, 'timestamps.txt')
            with open(timestamps_path, 'r') as f:
                timestamps = f.read().splitlines()

            event_ind0 = 0
            event_ind1 = timestamps.index(str(0.3333333333333333))
            event_ind2 = timestamps.index(str(0.6666666666666666))
            before_event_paths = [os.path.join(event_folder, str(i).zfill(10) + ".npz") for i in
                                  range(event_ind0, event_ind1)]
            after_event_paths = [os.path.join(event_folder, str(i).zfill(10) + ".npz") for i in
                                 range(event_ind1, len(os.listdir(event_folder)))]
            events_0t = EventSequence.from_npz_files(before_event_paths, h, w)
            events_t1 = EventSequence.from_npz_files(after_event_paths, h, w)
            event_voxel, mask = get_voxel_and_mask(num_bins, events_0t, events_t1, h, w)
            with torch.no_grad():
                tau = torch.tensor([0.5], device=device)
                pred = model.inference(imgs, event_voxel, mask, tau=tau)
            pred_out = (pred[0].clip(0.0, 1.0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            psnr = psnr_(gt, pred_out)
            ssim = ssim_(gt, pred_out)
            psnr_all.append(psnr)
            ssim_all.append(ssim)
            save_name = os.path.join(save_path, str(index).zfill(6) + ".png")
            if isTestPer:
                print(psnr, "  ", ssim)

            if isSave:
                if isinstance(saveIndexRange, list):
                    if index >= saveIndexRange[0] and index <= saveIndexRange[1]:
                        cv2.imwrite(save_name, pred_out)
                else:
                    cv2.imwrite(save_name, pred_out)
    print("PSNR: ", np.array(psnr_all).mean())
    print("SSIM: ", np.array(ssim_all).mean())

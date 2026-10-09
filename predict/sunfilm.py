import cv2
import numpy as np
import os
import torch

from predict.pred_utils import get_voxel_and_mask, psnr_, ssim_
from util.event import EventSequence


def predict_snufilm(model, num_bins, device, save_path, data_root, difficulties=['extreme', 'hard'], isSave=False, isTestPer=False,
                    saveSpecificScene=None):
    print("Start test SNU-FILM!")
    test_path = data_root
    for difficulty in difficulties:
        psnr_mul = []
        ssim_mul = []
        if difficulty == 'hard':
            test_fn = os.path.join(test_path, 'test-hard.txt')
        else:
            test_fn = os.path.join(test_path, 'test-extreme.txt')
        with open(test_fn, 'r') as f:
            test_list = f.read().splitlines()
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        val_folder = os.path.join(save_path, difficulty)
        if not os.path.exists(val_folder):
            os.mkdir(val_folder)
        for index in range(len(test_list)):
            if index != "":
                dataset, scene, name_0, name_gt, name_1 = test_list[index].split(" ")[:5]
                img_folder = os.path.join(test_path, dataset, scene, "imgs")
                img0 = cv2.imread(os.path.join(img_folder, name_0))

                img1 = cv2.imread(os.path.join(img_folder, name_1))
                img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
                gray_img0 = torch.zeros_like(img0)
                gray_img1 = torch.zeros_like(img0)
                gt = torch.zeros_like(img0)
                gray_gt = gt
                img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
                imgs = torch.concat((img0, gt, img1, gray_img0, gray_gt, gray_img1), 0).unsqueeze(0).to(device,
                                                                                                        non_blocking=True) / 255.
                gt = cv2.imread(os.path.join(img_folder, name_gt))
                h, w, _ = gt.shape
                event_folder = os.path.join(test_path, dataset, scene, "events")
                event_index = [int(i) for i in test_list[index].split(" ")[5:]]
                before_event_paths = [os.path.join(event_folder, str(i).zfill(10) + ".npz") for i in
                                      range(event_index[0], event_index[1])]
                after_event_paths = [os.path.join(event_folder, str(i).zfill(10) + ".npz") for i in
                                     range(event_index[1], event_index[2])]
                events_0t = EventSequence.from_npz_files(before_event_paths, h, w)
                events_t1 = EventSequence.from_npz_files(after_event_paths, h, w)
                event_voxel, mask = get_voxel_and_mask(num_bins, events_0t, events_t1, h, w)
                with torch.no_grad():
                    tau = torch.tensor([(event_index[1] - event_index[0]) / (event_index[2] - event_index[0])], device=device)
                    pred = model.inference(imgs, event_voxel, mask, tau=tau)
                pred_out = (pred[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                psnr = psnr_(gt, pred_out)
                ssim = ssim_(gt, pred_out)
                psnr_mul.append(psnr)
                ssim_mul.append(ssim)
                if isTestPer:
                    print(psnr, "  ", ssim)

                save_name = os.path.join(val_folder, str(index).zfill(6) + ".png")
                if isSave:
                    if isinstance(saveSpecificScene, list):
                        if scene in saveSpecificScene:
                            cv2.imwrite(save_name, pred_out)
                    else:
                        cv2.imwrite(save_name, pred_out)
        print(difficulty, " is ", np.array(psnr_mul).mean())
        print(difficulty, " is ", np.array(ssim_mul).mean())

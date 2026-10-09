import cv2
import numpy as np
import os
import torch

from predict.pred_utils import get_imgs, looking_for_event_index_by_timestamp, psnr_, ssim_


def predict_gopro(model, bins, device, save_path, data_root, multis=[7,15], isSave=False, isTestPer=False, 
                  saveSpecificScene=None):
    print("Start test GOPRO!")
    h, w = 720, 1280
    test_path = data_root
    val_path = save_path
    for multi in multis:
        psnr_mul = []
        ssim_mul = []
        for scene in os.listdir(test_path):
            if scene != '':
                psnr_all = []
                ssim_all = []
                val_folder = os.path.join(val_path, scene + '_' + str(multi))
                img_folder = os.path.join(test_path, scene, 'imgs')
                img_names = os.listdir(img_folder)
                img_names.sort()
                if not os.path.exists(val_path):
                    os.mkdir(val_path)
                if not os.path.exists(val_folder):
                    os.mkdir(val_folder)
                img_paths = [os.path.join(img_folder, i) for i in img_names]
                index = 0
                while (index + multi + 1) < len(img_names):
                    imgs = get_imgs(img_paths, multi, index)
                    for i in range(multi):
                        gt = cv2.imread(img_paths[index + i + 1])
                        img_index0, gt_index, img_index1 = index, index + i + 1, index + multi + 1
                        event_voxel, mask = looking_for_event_index_by_timestamp(os.path.join(test_path, scene), bins,
                                                                                    img_index0, gt_index,
                                                                                    img_index1, h, w, 240)
                        mask = mask.to(device, non_blocking=True)
                        event_voxel = event_voxel.to(device, non_blocking=True)
                        with torch.no_grad():
                            tau = torch.tensor([(gt_index - img_index0) / (img_index1 - img_index0)], device=device)
                            pred = model.inference(imgs, event_voxel, mask, tau=tau)
                        pred_out = (pred[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        psnr = psnr_(gt, pred_out)
                        ssim = ssim_(gt, pred_out)
                        psnr_all.append(psnr)
                        psnr_mul.append(psnr)
                        ssim_all.append(ssim)
                        ssim_mul.append(ssim)
                        if isTestPer:
                            print(psnr, "  ", ssim)
                            
                        save_name = os.path.join(val_folder, img_names[index + i + 1])
                        if isSave:
                            if isinstance(saveSpecificScene, list):
                                if scene in saveSpecificScene:
                                    cv2.imwrite(save_name, pred_out)
                            else:
                                cv2.imwrite(save_name, pred_out)
                    index = index + multi
                print(scene, "\'s ", multi, "\'s PSNR is ", np.array(psnr_all).mean())
                print(scene, "\'s ", multi, "\'s SSIM is ", np.array(ssim_all).mean())
        print(multi, "\'s PSNR is ", np.array(psnr_mul).mean())
        print(multi, "\'s SSIM is ", np.array(ssim_mul).mean())
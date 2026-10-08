import os
import cv2
import numpy as np
import torch

from predict.pred_utils import get_imgs, looking_for_event_index_by_timestamp, psnr_, ssim_


indexing_skip_ind = {
    "basket_09": [31, 32, 33, 34],
    "may29_rooftop_handheld_02": [17, 70],
    "may29_rooftop_handheld_03": [306],
    "may29_rooftop_handheld_05": [21],
}


def _mean_or_nan(values):
    return float(np.mean(values)) if values else float("nan")


def predict_bsergb(model,num_bins,device,save_path,data_root,multis=(1, 3, 5),isSave=False,isTestPer=False,saveSpecificScene=None,):

    if not os.path.isdir(data_root):
        raise FileNotFoundError("BS-ERGB data root does not exist: {}".format(data_root))
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    if not multis or any(multi <= 0 for multi in multis):
        raise ValueError("multis must contain positive integers")

    scene_names = sorted(name for name in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, name)))
    if not scene_names:
        raise RuntimeError("No BS-ERGB scene directories found under {}".format(data_root))

    print("Start test BSERGB!")
    for multi in multis:
        psnr_multi = []
        ssim_multi = []

        for scene in scene_names:
            scene_root = os.path.join(data_root, scene)
            image_folder = os.path.join(scene_root, "images")
            event_folder = os.path.join(scene_root, "events")
            if not os.path.isdir(image_folder) or not os.path.isdir(event_folder):
                raise FileNotFoundError(
                    "Scene {!r} must contain images/ and events/ directories".format(scene)
                )

            image_names = sorted(
                name
                for name in os.listdir(image_folder)
                if name.lower().endswith((".png", ".jpg", ".jpeg"))
            )
            image_paths = [os.path.join(image_folder, name) for name in image_names]
            if len(image_paths) < multi + 2:
                print(
                    "Skip {} for multi={}: only {} images".format(
                        scene, multi, len(image_paths)
                    )
                )
                continue

            first_image = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
            if first_image is None:
                raise RuntimeError("Failed to read image: {}".format(image_paths[0]))
            height, width = first_image.shape[:2]

            should_save = isSave and (
                not isinstance(saveSpecificScene, list)
                or scene in saveSpecificScene
            )
            val_folder = os.path.join(save_path, "{}_{}".format(scene, multi))
            if should_save:
                os.makedirs(val_folder, exist_ok=True)

            psnr_scene = []
            ssim_scene = []
            index = 0
            while index + multi + 1 < len(image_paths):
                imgs = get_imgs(image_paths, multi, index)

                for offset in range(multi):
                    image_index0 = index
                    target_index = index + offset + 1
                    image_index1 = index + multi + 1

                    excluded = indexing_skip_ind.get(scene, ())
                    if any(
                        image_index0 <= excluded_index < image_index1
                        for excluded_index in excluded
                    ):
                        continue

                    gt = cv2.imread(image_paths[target_index], cv2.IMREAD_COLOR)
                    if gt is None:
                        raise RuntimeError(
                            "Failed to read image: {}".format(image_paths[target_index])
                        )

                    event_voxel, mask = looking_for_event_index_by_timestamp(
                        scene_root,
                        num_bins,
                        image_index0,
                        target_index,
                        image_index1,
                        height,
                        width,
                        240,
                        real=True,
                        bsergb=True,
                    )
                    event_voxel = event_voxel.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)

                    with torch.no_grad():
                        prediction = model.inference(imgs, event_voxel, mask)
                    prediction = prediction[0].clamp(0.0, 1.0)
                    pred_out = (
                        prediction.cpu().numpy().transpose(1, 2, 0) * 255.0
                    ).round().astype(np.uint8)

                    psnr_value = psnr_(gt, pred_out)
                    ssim_value = ssim_(gt, pred_out)
                    psnr_scene.append(psnr_value)
                    psnr_multi.append(psnr_value)
                    ssim_scene.append(ssim_value)
                    ssim_multi.append(ssim_value)

                    if isTestPer:
                        print(
                            "{} frame {}: PSNR {:.6f}, SSIM {:.6f}".format(
                                scene, target_index, psnr_value, ssim_value
                            )
                        )

                    if should_save:
                        save_name = os.path.join(
                            val_folder, os.path.basename(image_paths[target_index])
                        )
                        if not cv2.imwrite(save_name, pred_out):
                            raise RuntimeError("Failed to save image: {}".format(save_name))

                # Preserve the evaluation stride used by the released code.
                index += multi

            print(
                "{} multi={}: PSNR {:.6f}, SSIM {:.6f}".format(
                    scene, multi, _mean_or_nan(psnr_scene), _mean_or_nan(ssim_scene)
                )
            )

        print(
            "multi={}: PSNR {:.6f}, SSIM {:.6f}".format(
                multi, _mean_or_nan(psnr_multi), _mean_or_nan(ssim_multi)
            )
        )

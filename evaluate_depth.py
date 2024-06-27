import os
from pathlib import Path

import cv2
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

import datasets
import networks
from utils.config import load_config
from utils.layers import transformation_from_parameters, disp_to_depth
from utils.utils import vis_tensor
splits_dir = "splits"

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(cfg):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = cfg.dataset.min_depth
    MAX_DEPTH = cfg.dataset.max_depth

    device = torch.device(cfg.device)

    frames_to_load = [0] + [-1 * i for i in range(1, cfg.num_matching_frames + 1)]

    load_weights_folder = Path(cfg.load_weights_folder)
    assert load_weights_folder.exists(), f"Cannot find a folder at {load_weights_folder}"
    print("-> Loading weights from {}".format(cfg.load_weights_folder))

    # ================= MODEL SETUP =================
    encoder_dict = torch.load(load_weights_folder / 'encoder.pth', map_location='cpu')
    min_depth_bin = encoder_dict.get('min_depth_bin')
    max_depth_bin = encoder_dict.get('max_depth_bin')

    print(f"-> Computing predictions with size {cfg.height}x{cfg.width}")

    models = dict()
    models["encoder"] = networks.StuEnc(cfg)
    model_dict = models["encoder"].state_dict()

    models["encoder"].load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    models["encoder"].to(device)
    models["encoder"].eval()

    models["depth"] = networks.DepthDec(cfg, models["encoder"].num_ch_enc)
    models["depth"].load_state_dict(torch.load(load_weights_folder / 'depth.pth', map_location='cpu'))
    models["depth"].to(device)
    models["depth"].eval()

    models["mono_enc"] = networks.TeaEnc(cfg)
    models["mono_enc"].load_state_dict(torch.load(load_weights_folder / 'mono_enc.pth', map_location='cpu'))
    models["mono_enc"].to(device)
    models["mono_enc"].eval()

    models["mono_dec"] = networks.DepthDec(cfg, models["mono_enc"].num_ch_enc)
    models["mono_dec"].load_state_dict(torch.load(load_weights_folder / 'mono_dec.pth', map_location='cpu'))
    models["mono_dec"].to(device)
    models["mono_dec"].eval()

    models["posenet"] = getattr(networks, cfg.posenet.version)(cfg)
    models["posenet"].load_state_dict(torch.load(load_weights_folder / 'posenet.pth', map_location='cpu'))
    models["posenet"].to(device)
    models["posenet"].eval()

    # ================= setup dataloaders =================
    test_set = datasets.get_test_dataset(cfg)
    dataloader = DataLoader(test_set,
                            cfg.eval.batch_size,
                            shuffle=False,
                            num_workers=cfg.eval.num_workers,
                            pin_memory=True)

    pred_disps = []

    # =============================== inference ===============================
    with torch.no_grad():
        for i, data in tqdm.tqdm(enumerate(dataloader)):
            images = {frame: data["color", frame, 0].to(device) for frame in frames_to_load}

            if cfg.eval.eval_teacher:
                output = models["mono_enc"](images[0])
                output = models["mono_dec"](output)
            else:
                # pose predictions
                axisangle, translation = models["posenet"](images[-1], images[0])
                pose = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=True)

                relative_poses = [pose]
                relative_poses = torch.stack(relative_poses, 1).to(device)

                K = data[('K', 2)].to(device)  # quarter resolution for matching
                invK = data[('inv_K', 2)].to(device)
                lookup_frames = torch.stack([images[-1]], 1)

                output, lowest_cost, costvol = models["encoder"](images[0], lookup_frames,
                                                       relative_poses,
                                                       K,
                                                       invK,
                                                       min_depth_bin, max_depth_bin)
                output = models["depth"](output)

            pred_disp, _ = disp_to_depth(output[("disp", 0)], cfg.dataset.min_depth, cfg.dataset.max_depth)
            pred_disp = pred_disp.cpu()[:, 0].numpy()
            pred_disps.append(pred_disp)

    pred_disps = np.concatenate(pred_disps)

    print('finished predicting!')
    if cfg.save_pred_disps:
        if cfg.eval.eval_teacher:
            tag = "teacher"
        else:
            tag = "multi"
        output_path = os.path.join(
            cfg.load_weights_folder, "{}_{}_split.npy".format(tag, cfg.eval.split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if cfg.eval.split == 'benchmark':
        save_dir = os.path.join(cfg.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    elif cfg.eval.split == 'cityscapes':
        print('loading cityscapes gt depths individually due to their combined size!')
        gt_depths = os.path.join(splits_dir, cfg.eval.split, "gt_depths")
    else:
        gt_path = os.path.join(splits_dir, cfg.eval.split, "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    errors = []
    ratios = []
    for i in tqdm.tqdm(range(pred_disps.shape[0])):

        if cfg.eval.split == 'cityscapes':
            gt_depth = np.load(os.path.join(gt_depths, str(i).zfill(3) + '_depth.npy'))
            gt_height, gt_width = gt_depth.shape[:2]
            # crop ground truth to remove ego car -> this has happened in the dataloader for input
            # images
            gt_height = int(round(gt_height * 0.75))
            gt_depth = gt_depth[:gt_height]

        else:
            gt_depth = gt_depths[i]
            gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = np.squeeze(pred_disps[i])
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if cfg.eval.split == 'cityscapes':
            # when evaluating cityscapes, we centre crop to the middle 50% of the image.
            # Bottom 25% has already been removed - so crop the sides and the top here
            gt_depth = gt_depth[256:, 192:1856]
            pred_depth = pred_depth[256:, 192:1856]

        if cfg.eval.split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        elif cfg.eval.split == 'cityscapes':
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= cfg.eval.pred_depth_scale_factor
        if not cfg.eval.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if cfg.save_pred_disps:
        print("saving errors")
        if cfg.eval.eval_teacher:
            tag = "mono"
        else:
            tag = "multi"
        output_path = os.path.join(
            cfg.load_weights_folder, "{}_{}_errors.npy".format(tag, cfg.eval.split))
        np.save(output_path, np.array(errors))

    if not cfg.eval.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.4f} | std: {:0.4f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel",
                                           "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.4f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    txt_dir = Path(cfg.load_weights_folder)
    with open(txt_dir / 'results.txt', 'w') as f:
        for i in mean_errors.tolist():
            f.write(f'{i:.4f}\t')
    with open(txt_dir.parent / 'results.txt', 'a') as f:
        for i in mean_errors.tolist():
            f.write(f'{i:.4f}\t')
        f.write(f'weights_{txt_dir.name.split("_")[-1]}\n')


if __name__ == "__main__":
    opt = load_config()
    evaluate(opt)

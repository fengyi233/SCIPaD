import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import networks
from utils.config import load_config
from utils.layers import transformation_from_parameters


# from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        xyzs.append(cam_to_world[:3, 3])
    return xyzs


# from https://github.com/tinghuiz/SfMLearner
def compute_ate(gtruth_xyz, pred_xyz_o):
    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]

    # Optimize the scaling factor
    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2)) / gtruth_xyz.shape[0]
    return rmse


def evaluate(cfg):
    """Evaluate odometry on the KITTI dataset
    """
    device = torch.device(cfg.device)

    assert cfg.eval.split == "odom_09" or cfg.eval.split == "odom_10", \
        "eval_split should be either odom_09 or odom_10"

    sequence_id = int(cfg.eval.split.split("_")[1])
    frame_ids = [0, 1]
    # ================= dataloader setup =================
    test_set = datasets.get_test_dataset(cfg)
    dataloader = DataLoader(test_set,
                            batch_size=1,
                            shuffle=False,
                            num_workers=cfg.eval.num_workers,
                            pin_memory=True)

    posenet = getattr(networks, cfg.posenet.version)(cfg)
    load_weights_folder = Path(cfg.load_weights_folder)
    posenet.load_state_dict(torch.load(load_weights_folder / 'posenet.pth', map_location='cpu'))
    posenet.to(device)
    posenet.eval()

    pred_poses = []
    global_pose = np.eye(4)
    poses = [global_pose[0:3, :].reshape(1, 12)]

    print("-> Computing pose predictions")

    with torch.no_grad():
        for inputs in tqdm(dataloader):
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()

            axisangle, translation = posenet(inputs[("color_aug", 0, 0)], inputs[("color_aug", 1, 0)])
            pose = transformation_from_parameters(axisangle[:, 0], translation[:, 0])
            pred_poses.append(pose.cpu().numpy())

            pose_mat = pose.squeeze(0).cpu().numpy()
            global_pose = global_pose @ np.linalg.inv(pose_mat)
            poses.append(global_pose[0:3, :].reshape(1, 12))

    pred_poses = np.concatenate(pred_poses)

    gt_poses_path = os.path.join(cfg.dataset.data_path, "poses", "{:02d}.txt".format(sequence_id))
    gt_global_poses = np.loadtxt(gt_poses_path).reshape(-1, 3, 4)
    gt_global_poses = np.concatenate(
        (gt_global_poses, np.zeros((gt_global_poses.shape[0], 1, 4))), 1)
    gt_global_poses[:, 3, 3] = 1
    gt_xyzs = gt_global_poses[:, :3, 3]

    gt_local_poses = []
    for i in range(1, len(gt_global_poses)):
        gt_local_poses.append(
            np.linalg.inv(np.dot(np.linalg.inv(gt_global_poses[i - 1]), gt_global_poses[i])))

    ates = []
    num_frames = gt_xyzs.shape[0]
    track_length = 5
    for i in range(0, num_frames - 1):
        local_xyzs = np.array(dump_xyz(pred_poses[i:i + track_length - 1]))
        gt_local_xyzs = np.array(dump_xyz(gt_local_poses[i:i + track_length - 1]))

        ates.append(compute_ate(gt_local_xyzs, local_xyzs))

    print("\n   Trajectory error: {:0.3f}, std: {:0.3f}\n".format(np.mean(ates), np.std(ates)))

    # save_path = os.path.join(load_weights_folder, "poses.npy")
    # np.save(save_path, pred_poses)
    # print("-> Predictions saved to", save_path)
    poses = np.concatenate(poses, axis=0)
    np.savetxt(load_weights_folder/f"{sequence_id:02d}.txt", poses, delimiter=' ', fmt='%1.8e')


if __name__ == "__main__":
    opt = load_config()
    evaluate(opt)

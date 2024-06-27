import os

import PIL.Image as pil
import numpy as np
import skimage.transform
from PIL import Image
from torchvision import transforms

from datasets.mono_dataset import DataSetUsage
from datasets.mono_dataset import MonoDataset
from utils.kitti_utils import generate_depth_map
from utils.seg_utils import *


class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """

    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)
        self.K = []
        self.inv_K = []
        self.init_k()

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def init_k(self):
        for s in self.scales:
            base_K = torch.tensor([[0.58, 0, 0.5, 0],
                                   [0, 1.92, 0.5, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], dtype=torch.float32)
            base_K[0, :] *= self.width // (2 ** s)
            base_K[1, :] *= self.height // (2 ** s)
            self.K.append(base_K)
            self.inv_K.append(torch.linalg.pinv(base_K))

    def get_intrinsics(self, scale, inv=False):
        if inv is True:
            return self.inv_K[scale]
        else:
            return self.K[scale]

    def parse_filename(self, index):
        """Convert index in the dataset to a folder name, frame_idx and any other bits
        """
        line = self.filenames[index].split()
        folder = line[0]
        if len(line) == 3:
            frame_index = int(line[1])
            side = line[2]
        else:
            frame_index = 0
            side = None

        return folder, frame_index, side

    def get_rgb(self, folder, frame_index, side, do_flip):
        rgb_path = os.path.join(self.data_path, folder, f"image_0{self.side_map[side]}"
                                                        f"/data/{frame_index:010d}{self.img_ext}")
        rgb = Image.open(rgb_path).convert('RGB')
        if do_flip:
            rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
        return rgb

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def get_semantics(self, folder, frame_index, side, do_flip):
        sem_path = os.path.join(self.data_path, 'segmentation', folder, f"image_0{self.side_map[side]}"
                                                                        f"/{frame_index:010d}.png")
        semantics = Image.open(sem_path)
        seg_copy = np.array(semantics.copy())
        for k in np.unique(semantics):
            seg_copy[seg_copy == k] = labels[k].trainId
        semantics = Image.fromarray(seg_copy, mode='P')
        if do_flip:
            semantics = semantics.transpose(Image.FLIP_LEFT_RIGHT)
        return semantics


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """

    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)
        if self.dataset_usage == DataSetUsage.TEST:
            # segmentation is only needed when training or validating.
            return
        self.resize_seg = transforms.Resize((self.height, self.width,),
                                            interpolation=Image.BILINEAR)

    def get_image_path(self, folder, frame_index, side, seg=False):
        f_str = "{:010d}{}".format(frame_index, '.png' if seg else self.img_ext)
        assert side is not None
        if seg:
            image_path = os.path.join(
                self.data_path, folder, "image_0{}".format(self.side_map[side]), f_str)
        else:
            image_path = os.path.join(
                self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_item_custom(self, inputs, folder, frame_index, side, do_flip):
        if self.dataset_usage == DataSetUsage.TEST:
            # semantic segmentation is not needed when testing (inferring).
            return
        raw_seg = self.get_seg_map(folder, frame_index, side, do_flip)
        seg = self.resize_seg(raw_seg)
        inputs[('seg', 0, 0)] = torch.tensor(np.array(seg)).float().unsqueeze(0)

    def get_seg_map(self, folder, frame_index, side, do_flip):
        path = self.get_image_path(folder, frame_index, side, True)
        path = path.replace(folder, f'segmentation/{folder}')

        seg = self.loader(path, mode='P')
        seg_copy = np.array(seg.copy())

        for k in np.unique(seg):
            seg_copy[seg_copy == k] = labels[k].trainId
        seg = Image.fromarray(seg_copy, mode='P')

        if do_flip:
            seg = seg.transpose(pil.FLIP_LEFT_RIGHT)
        return seg

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """

    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_rgb(self, folder, frame_index, side, do_flip):
        rgb_path = os.path.join(self.data_path,
                                f"sequences/{int(folder):02d}/image_{self.side_map[side]}",
                                f"{frame_index:06d}{self.img_ext}")
        rgb = Image.open(rgb_path).convert('RGB')
        if do_flip:
            rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
        return rgb


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """

    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

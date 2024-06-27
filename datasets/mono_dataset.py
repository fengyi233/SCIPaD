import os
import random
from enum import Enum

import numpy as np
import torch.utils.data as data
from PIL import Image  # using pillow-simd for increased speed
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from utils.utils import readlines


class DataSetUsage(Enum):
    TRAIN = 0
    VALIDATE = 1
    TEST = 2


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders
    """

    def __init__(self, cfg, dataset_usage):
        super(MonoDataset, self).__init__()
        self.cfg = cfg
        self.dataset_name = cfg.dataset.name
        self.dataset_usage = dataset_usage

        self.data_path = cfg.dataset.data_path
        self.height = cfg.height
        self.width = cfg.width
        self.scales = cfg.scales
        self.frame_ids = cfg.frame_ids
        self.img_ext = cfg.dataset.img_ext
        if dataset_usage == DataSetUsage.TRAIN:
            self.filenames = readlines(os.path.join("splits", cfg.dataset.split, "train_files.txt"))
        elif dataset_usage == DataSetUsage.VALIDATE:
            self.filenames = readlines(os.path.join("splits", cfg.dataset.split, "val_files.txt"))
        else:
            if cfg.eval.split in ['odom_09', 'odom_10']:
                sequence_id = int(cfg.eval.split[-2:])
                self.filenames = readlines(f'splits/odom/test_files_{sequence_id:02d}.txt')
                self.frame_ids = [0, 1]
            elif cfg.eval.split in ['eigen', 'eigen_benchmark', 'cityscapes', 'vkitti2']:
                self.filenames = readlines(f'splits/{cfg.eval.split}/test_files.txt')
            else:
                raise NotImplementedError

        self.to_tensor = transforms.ToTensor()
        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
        self.saturation = (0.8, 1.2)
        self.hue = (-0.1, 0.1)

        self.resize = [transforms.Resize((self.height // 2 ** i, self.width // 2 ** i),
                                         interpolation=InterpolationMode.NEAREST)
                       for i in self.scales]

    def preprocess(self, inputs, color_aug):
        """Resize images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for i in self.frame_ids:  # [0,-1,1]
            for scale in self.scales:  # [0,1,2,3]
                inputs[("color", i, scale)] = self.to_tensor(self.resize[scale](inputs[("color", i, -1)]))
            inputs[("color_aug", i, 0)] = color_aug(inputs[("color", i, 0)])
            del inputs[("color", i, -1)]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "depth_gt"                              for ground truth depth maps

        <frame_id> is:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        # Do data augmentation only in training
        do_color_aug = self.dataset_usage == DataSetUsage.TRAIN and random.random() > 0.5
        do_flip = self.dataset_usage == DataSetUsage.TRAIN and random.random() > 0.5
        folder, frame_index, side = self.parse_filename(index)

        # ========================= get rgb ====================================
        for i in self.frame_ids:
            try:
                inputs[("color", i, -1)] = self.get_rgb(folder, frame_index + i, side, do_flip)
            except FileNotFoundError as e:
                if i != 0:
                    inputs[("color", i, -1)] = inputs[("color", 0, -1)]
                else:
                    raise FileNotFoundError(e)
        if do_color_aug:
            color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)
        # ==================== get intrinsics ====================================
        for scale in self.scales:
            inputs[("K", scale)] = self.get_intrinsics(scale)
            inputs[("inv_K", scale)] = self.get_intrinsics(scale, inv=True)

        # ==================== get semantics ====================================
        if self.dataset_name in ['kitti']:
            # if self.dataset_name in ['kitti', 'vkitti2']:
            inputs[('seg', 0, 0)] = self.resize[0](
                self.to_tensor(self.get_semantics(folder, frame_index, side, do_flip)))

        # ==================== get depth ====================================
        if self.dataset_usage == DataSetUsage.TEST and self.dataset_name in ['vkitti2']:
            inputs['depth_gt'] = self.get_depth(folder, frame_index, side, do_flip)

        # ==================== get poses ====================================
        # if self.dataset_usage == DataSetUsage.TEST and self.dataset_name in ['vkitti2']:
        if self.dataset_name in ['vkitti2']:
            inputs[('rel_pose', -1)], inputs[('rel_pose', 1)] = self.get_rel_pose(folder, frame_index, side)
        return inputs


    def get_rgb(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def get_semantics(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def get_intrinsics(self, scale, inv=False):
        return NotImplementedError

    def get_rel_pose(self, folder, frame_index, side):
        return NotImplementedError

    def parse_filename(self, index):
        return NotImplementedError

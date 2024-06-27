import random

import numpy as np
import torch
from matplotlib import pyplot as plt

_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)  # for plotting


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def vis_tensor(tensor, save_filename=None):
    """
    可视化PyTorch张量为图像并保存图像。
    Args:
        tensor (torch.Tensor): 要可视化的PyTorch张量 [c,h,w]或[h,w]。
        save_filename (str, optional): 要保存的图像文件名。默认为None，不保存图像。
    """
    if len(tensor.shape) == 2:
        image = tensor.cpu().detach().numpy()
    else:
        if tensor.shape[0] == 3:  # rgb
            image = tensor.cpu().detach().numpy() * 0.225 + 0.45
            image = np.transpose(image, (1, 2, 0))
        else:
            image = tensor.squeeze().cpu().detach().numpy()

    plt.figure()
    plt.imshow(image)
    plt.axis('off')
    if save_filename:
        plt.savefig('vis/' + save_filename)
    else:
        plt.show()


def colormap(inputs, normalize=True, torch_transpose=True):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()

    vis = inputs
    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d

    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def seed_all(seed):
    if not seed:
        seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

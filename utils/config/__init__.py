import argparse
import pathlib
import torch
import yacs.config

from .config_node import ConfigNode
from .defaults import get_default_config


def update_config(config):
    if config.dataset.name in ['CIFAR10', 'CIFAR100']:
        dataset_dir = f'~/.torch/datasets/{config.dataset.name}'
        config.dataset.dataset_dir = dataset_dir
        config.dataset.image_size = 32
        config.dataset.n_channels = 3
        config.dataset.n_classes = int(config.dataset.name[5:])
    elif config.dataset.name in ['MNIST', 'FashionMNIST', 'KMNIST']:
        dataset_dir = '~/.torch/datasets'
        config.dataset.dataset_dir = dataset_dir
        config.dataset.image_size = 28
        config.dataset.n_channels = 1
        config.dataset.n_classes = 10

    if not torch.cuda.is_available():
        config.device = 'cpu'

    return config


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    if args.config is not None:
        config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    config = update_config(config)
    config.freeze()
    return config


def save_config(config: yacs.config.CfgNode,
                output_path: pathlib.Path) -> None:
    with open(output_path, 'w') as f:
        f.write(str(config))


def find_config_diff(config: yacs.config.CfgNode):
    def _find_diff(node: yacs.config.CfgNode,
                   default_node: yacs.config.CfgNode):
        root_node = ConfigNode()
        for key in node:
            val = node[key]
            if isinstance(val, yacs.config.CfgNode):
                new_node = _find_diff(node[key], default_node[key])
                if new_node is not None:
                    root_node[key] = new_node
            else:
                if node[key] != default_node[key]:
                    root_node[key] = node[key]
        return root_node if len(root_node) > 0 else None

    default_config = get_default_config()
    new_config = _find_diff(config, default_config)
    return new_config

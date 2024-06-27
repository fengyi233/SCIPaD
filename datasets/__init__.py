from .kitti_dataset import KITTIDataset, KITTIOdomDataset
from .mono_dataset import DataSetUsage

def get_train_val_dataset(cfg):
    datasets_dict = {"kitti": KITTIDataset,
                     "kitti_odom": KITTIOdomDataset}
    train_set = datasets_dict[cfg.dataset.name](cfg, dataset_usage=DataSetUsage.TRAIN)
    val_set = datasets_dict[cfg.dataset.name](cfg, dataset_usage=DataSetUsage.VALIDATE)
    return train_set, val_set


def get_test_dataset(cfg):
    datasets_dict = {"kitti": KITTIDataset,
                     "kitti_odom": KITTIOdomDataset}
    test_set = datasets_dict[cfg.dataset.name](cfg, dataset_usage=DataSetUsage.TEST)
    return test_set

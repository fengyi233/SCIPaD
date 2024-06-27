from .config_node import ConfigNode

config = ConfigNode()

config.device = 0
config.seed = 1

config.output_dir = 'runs'
config.exp_name = 'debug'
config.ckpt_path = ''

config.freeze_teacher_and_pose = False
config.num_matching_frames = 1
config.height = 192
config.width = 640
config.frame_ids = [0, -1, 1]
config.scales = [0, 1, 2, 3]
config.load_weights_folder = ''
config.mono_weights_folder = ''
config.models_to_load = ["encoder", "depth", "mono_enc", "mono_dec", "posenet"]
config.save_frequency = 1
config.log_frequency = 100
config.no_matching_augmentation = False
config.disable_motion_masking = False
config.disable_automasking = False
config.avg_reprojection = False
config.disable_triplet_loss = False
config.disable_isolated_triplet = False
config.disable_hardest_neg = False
config.sgt = 0.1
config.sgt_scales = [3, 2, 1]
config.sgt_margin = 0.35
config.sgt_isolated_margin = 0.65
config.sgt_isolated_margin = 0.35
config.sgt_kernel_size = [5, 5, 5]

config.save_pred_disps = False

config.dataset = ConfigNode()
# choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test",
#          "cityscapes_preprocessed", "vkitti2"]
config.dataset.name = 'kitti'
config.dataset.data_path = ''
config.dataset.img_ext = '.jpg'
# choices=["eigen", "eigen_benchmark", "benchmark", "cityscapes", "vkitti2"]
config.dataset.split = 'eigen_zhou'
config.dataset.img_ext = '.jpg'
config.dataset.max_depth = 80.0
config.dataset.min_depth = 0.1

# config.depthnet = ConfigNode()
# config.depthnet.backbone = 'resnet18'
# config.depthnet.num_layers = 18
config.teanet = ConfigNode()
config.teanet.backbone = 'resnet18'

config.stunet = ConfigNode()
config.stunet.backbone = 'resnet18'
config.stunet.num_depth_bins = 96

config.posenet = ConfigNode()
config.posenet.version = 'PoseNet'  # original posenet in tridepth
config.posenet.backbone = 'resnet18'
config.posenet.embed_dim = 96

# choices = ['rgb', 'rgb_disp_raw', 'rgb_disp', 'rgb_depth', 'rgb_xyz']
config.posenet.inputs = 'rgb'

config.train = ConfigNode()
config.train.checkpoint = ''
config.train.resume = ''

config.train.batch_size = 8
config.train.scales = [0, 1, 2, 3]
config.train.batch_size = 12
config.train.epochs = 20
config.train.epoch_size = 1.0
config.train.val_mode = 'depth'
config.train.lr = 1e-4
config.train.scheduler_step_size = 15
config.train.num_workers = 12

# evaluation
config.eval = ConfigNode()
config.eval.batch_size = 12
config.eval.num_workers = 12
config.eval.split = 'eigen'  # ["eigen", "eigen_benchmark", "benchmark", "cityscapes", "vkitti2", "odom_09", "odom_10"]
config.eval.stereo = False
config.eval.zero_cost_volume = False
config.eval.save_pred_disps = False
config.eval.eval_teacher = False
config.eval.disable_median_scaling = False
config.eval.pred_depth_scale_factor = 1


def get_default_config():
    return config.clone()

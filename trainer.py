import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import networks
from datasets import get_train_val_dataset
from utils.config import save_config, find_config_diff
from utils.layers import SSIM, BackprojectDepth, Project3D, transformation_from_parameters, \
    disp_to_depth
from utils.logger import create_logger
from utils.utils import sec_to_hm_str, colormap


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(f"cuda:{cfg.device}")
        self.log_path = Path(self.cfg.output_dir) / self.cfg.exp_name
        self.log_path.mkdir(parents=True, exist_ok=True)

        self.logger = create_logger(name=cfg.exp_name,
                                    output_dir=self.log_path,
                                    filename='log.txt')
        self.system_check()

        self.models = {}
        self.parameters_to_train = []

        self.train_teacher_and_pose = not self.cfg.freeze_teacher_and_pose
        if self.train_teacher_and_pose:
            self.logger.info('using adaptive depth binning!')
            self.min_depth_tracker = 0.1
            self.max_depth_tracker = 10.0
        else:
            self.logger.info('fixing pose network and monocular network!')

        # check the frames we need the dataloader to load
        self.matching_ids = [0, -1]
        self.logger.info('Loading frames: {}'.format(self.cfg.frame_ids))

        # ================= model setup =================
        self.models["encoder"] = networks.StuEnc(cfg)
        self.models["encoder"].to(self.device)

        self.models["depth"] = networks.DepthDec(cfg, self.models["encoder"].num_ch_enc)
        self.models["depth"].to(self.device)

        self.models["mono_enc"] = networks.TeaEnc(cfg)
        self.models["mono_enc"].to(self.device)

        self.models["mono_dec"] = networks.DepthDec(cfg, self.models["mono_enc"].num_ch_enc)
        self.models["mono_dec"].to(self.device)

        self.models["posenet"] = getattr(networks, cfg.posenet.version)(self.cfg)
        self.models["posenet"].to(self.device)

        self.parameters_to_train += list(self.models["encoder"].parameters())
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.train_teacher_and_pose:
            self.parameters_to_train += list(self.models["mono_enc"].parameters())
            self.parameters_to_train += list(self.models["mono_dec"].parameters())
            self.parameters_to_train += list(self.models["posenet"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.cfg.train.lr)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.cfg.train.scheduler_step_size, 0.1)

        if self.cfg.load_weights_folder:
            self.load_model()

        # ================= dataset setup =================
        train_dataset, val_dataset = get_train_val_dataset(cfg)
        self.logger.info(f"There are {len(train_dataset)} training items and {len(val_dataset)} validation items\n")
        self.train_loader = DataLoader(train_dataset, cfg.train.batch_size, shuffle=True,
                                       num_workers=cfg.train.num_workers, pin_memory=True, drop_last=True,
                                       worker_init_fn=seed_worker)

        self.num_total_steps = len(train_dataset) // self.cfg.train.batch_size * self.cfg.train.epochs

        self.writer = SummaryWriter(self.log_path / 'train')

        self.ssim = SSIM()
        self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}

        for scale in self.cfg.scales:
            h = self.cfg.height // (2 ** scale)
            w = self.cfg.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.cfg.train.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.cfg.train.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.cfg.dataset.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for k, m in self.models.items():
            if self.train_teacher_and_pose:
                m.train()
            else:
                if k in ['depth', 'encoder']:
                    m.train()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.cfg.train.epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.cfg.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs, is_train=True)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.cfg.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                self.log(inputs, outputs, losses)

            self.step += 1
        self.model_lr_scheduler.step()

    def process_batch(self, inputs, is_train=False):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        mono_outputs = {}
        outputs = {}

        # single frame path
        if self.train_teacher_and_pose:
            feats = self.models["mono_enc"](inputs["color_aug", 0, 0])
            mono_outputs.update(self.models['mono_dec'](feats))
        else:
            with torch.no_grad():
                feats = self.models["mono_enc"](inputs["color_aug", 0, 0])
                mono_outputs.update(self.models['mono_dec'](feats))
        b,_,h,w = mono_outputs[("disp", 0)].shape
        _, depth = disp_to_depth(mono_outputs[("disp", 0)], self.cfg.dataset.min_depth, self.cfg.dataset.max_depth)
        point_cloud = self.backproject_depth[0](depth, inputs[("inv_K", 0)])[:, :3, :].view(b, -1, h, w)  #

        # predict poses for all frames
        if self.train_teacher_and_pose:
            pose_pred = self.predict_poses(inputs, point_cloud)
        else:
            with torch.no_grad():
                pose_pred = self.predict_poses(inputs, point_cloud)
        outputs.update(pose_pred)
        mono_outputs.update(pose_pred)

        self.generate_images_pred(inputs, mono_outputs)
        mono_losses = self.compute_losses(inputs, mono_outputs, is_multi=False)

        # grab poses + frames and stack for input to the multi frame network
        relative_poses = [inputs[('relative_pose', idx)] for idx in self.matching_ids[1:]]
        relative_poses = torch.stack(relative_poses, 1)
        lookup_frames = [inputs[('color_aug', idx, 0)] for idx in self.matching_ids[1:]]
        lookup_frames = torch.stack(lookup_frames, 1)  # batch x frames x 3 x h x w

        # apply static frame and zero cost volume augmentation
        batch_size = len(lookup_frames)
        augmentation_mask = torch.zeros([batch_size, 1, 1, 1]).to(self.device).float()
        if is_train and not self.cfg.no_matching_augmentation:
            for batch_idx in range(batch_size):
                rand_num = random.random()
                # static camera augmentation -> overwrite lookup frames with current frame
                if rand_num < 0.25:
                    replace_frames = \
                        [inputs[('color', 0, 0)][batch_idx] for _ in self.matching_ids[1:]]
                    replace_frames = torch.stack(replace_frames, 0)
                    lookup_frames[batch_idx] = replace_frames
                    augmentation_mask[batch_idx] += 1
                # missing cost volume augmentation -> set all poses to 0, the cost volume will
                # skip these frames
                elif rand_num < 0.5:
                    relative_poses[batch_idx] *= 0
                    augmentation_mask[batch_idx] += 1
        outputs['augmentation_mask'] = augmentation_mask

        min_depth_bin = self.min_depth_tracker
        max_depth_bin = self.max_depth_tracker

        # update multi frame outputs dictionary with single frame outputs
        for key in list(mono_outputs.keys()):
            _key = list(key)
            if _key[0] in ['depth', 'disp']:
                _key[0] = 'mono_' + key[0]
                _key = tuple(_key)
                outputs[_key] = mono_outputs[key]

        # multi frame path
        features, lowest_cost, confidence_mask = self.models["encoder"](inputs["color_aug", 0, 0],
                                                                        lookup_frames,
                                                                        relative_poses,
                                                                        inputs[('K', 2)],
                                                                        inputs[('inv_K', 2)],
                                                                        min_depth_bin=min_depth_bin,
                                                                        max_depth_bin=max_depth_bin)
        outputs.update(self.models["depth"](features))

        outputs["lowest_cost"] = F.interpolate(lowest_cost.unsqueeze(1),
                                               [self.cfg.height, self.cfg.width],
                                               mode="nearest")[:, 0]
        outputs["consistency_mask"] = F.interpolate(confidence_mask.unsqueeze(1),
                                                    [self.cfg.height, self.cfg.width],
                                                    mode="nearest")[:, 0]

        if not self.cfg.disable_motion_masking:
            outputs["consistency_mask"] = (outputs["consistency_mask"] *
                                           self.compute_matching_mask(outputs))

        self.generate_images_pred(inputs, outputs, is_multi=True)
        losses = self.compute_losses(inputs, outputs, is_multi=True)

        # update losses with single frame losses
        if self.train_teacher_and_pose:
            for key, val in mono_losses.items():
                losses[key] += val

        # update adaptive depth bins
        if self.train_teacher_and_pose:
            self.update_adaptive_depth_bins(outputs)

        return outputs, losses

    def update_adaptive_depth_bins(self, outputs):
        """Update the current estimates of min/max depth using exponental weighted average"""

        min_depth = outputs[('mono_depth', 0, 0)].detach().min(-1)[0].min(-1)[0]
        max_depth = outputs[('mono_depth', 0, 0)].detach().max(-1)[0].max(-1)[0]

        min_depth = min_depth.mean().cpu().item()
        max_depth = max_depth.mean().cpu().item()

        # increase range slightly
        min_depth = max(self.cfg.dataset.min_depth, min_depth * 0.9)
        max_depth = max_depth * 1.1

        self.max_depth_tracker = self.max_depth_tracker * 0.99 + max_depth * 0.01
        self.min_depth_tracker = self.min_depth_tracker * 0.99 + min_depth * 0.01

    def predict_poses(self, inputs, pc):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.cfg.frame_ids}
        for f_i in self.cfg.frame_ids[1:]:
            if f_i != "s":
                # To maintain ordering we always pass frames in temporal order
                if f_i < 0:
                    axisangle, translation = self.models["posenet"](pose_feats[f_i], pose_feats[0], pc)
                else:
                    axisangle, translation = self.models["posenet"](pose_feats[0], pose_feats[f_i], pc)

                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation

                # Invert the matrix if the frame id is negative
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        # now we need poses for matching - compute without gradients
        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.matching_ids}
        with torch.no_grad():
            # compute pose from 0->-1, -1->-2, -2->-3 etc and multiply to find 0->-3
            for fi in self.matching_ids[1:]:
                if fi < 0:
                    axisangle, translation = self.models["posenet"](pose_feats[fi], pose_feats[fi + 1], pc)
                    pose = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=True)

                    # now find 0->fi pose
                    if fi != -1:
                        pose = torch.matmul(pose, inputs[('relative_pose', fi + 1)])

                else:
                    axisangle, translation = self.models["posenet"](pose_feats[fi - 1], pose_feats[fi], pc)
                    pose = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=False)

                    # now find 0->fi pose
                    if fi != 1:
                        pose = torch.matmul(pose, inputs[('relative_pose', fi - 1)])

                # set missing images to 0 pose
                for batch_idx, feat in enumerate(pose_feats[fi]):
                    if feat.sum() == 0:
                        pose[batch_idx] *= 0

                inputs[('relative_pose', fi)] = pose

        return outputs

    def generate_images_pred(self, inputs, outputs, is_multi=False):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.cfg.scales:
            disp = outputs[("disp", scale)]
            disp = F.interpolate(
                disp, [self.cfg.height, self.cfg.width], mode="bilinear", align_corners=False)
            source_scale = 0

            _, depth = disp_to_depth(disp, self.cfg.dataset.min_depth, self.cfg.dataset.max_depth)
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.cfg.frame_ids[1:]):

                T = outputs[("cam_T_cam", 0, frame_id)]
                if is_multi:
                    # don't update posenet based on multi frame prediction
                    T = T.detach()

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

                if not self.cfg.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        Args:
            pred: [b,3,h,w], synthetic color image, original resolution
            target: [b,3,h,w], input color image, original resolution
        Math:
            Loss = 0.15 * mean(|target - pred|) + 0.85 * ssim(pred, target)
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    @staticmethod
    def compute_loss_masks(reprojection_loss, identity_reprojection_loss):
        """ Compute loss masks for each of standard reprojection and depth hint
        reprojection"""

        if identity_reprojection_loss is None:
            # we are not using automasking - standard reprojection loss applied to all pixels
            reprojection_loss_mask = torch.ones_like(reprojection_loss)

        else:
            # we are using automasking
            all_losses = torch.cat([reprojection_loss, identity_reprojection_loss], dim=1)
            idxs = torch.argmin(all_losses, dim=1, keepdim=True)
            reprojection_loss_mask = (idxs == 0).float()
            # vis_tensor(reprojection_loss_mask[0])
        return reprojection_loss_mask

    def compute_matching_mask(self, outputs):
        """Generate a mask of where we cannot trust the cost volume, based on the difference
        between the cost volume and the teacher, monocular network"""

        mono_output = outputs[('mono_depth', 0, 0)]
        matching_depth = 1 / outputs['lowest_cost'].unsqueeze(1).to(self.device)

        # mask where they differ by a large amount
        mask = ((matching_depth - mono_output) / mono_output) < 1.0
        mask *= ((mono_output - matching_depth) / matching_depth) < 1.0
        return mask[:, 0]

    def compute_losses(self, inputs, outputs, is_multi=False):
        """Compute the reprojection, smoothness and proxy supervised losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.cfg.scales:
            loss = 0
            reprojection_losses = []
            source_scale = 0
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.cfg.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.cfg.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.cfg.frame_ids[1:]:
                    # frame_0 <--> frame1,-1
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))  # b,1,h,w

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)  # b,frame_num,h,w

                if self.cfg.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # differently to Monodepth2, compute mins as we go
                    identity_reprojection_loss, _ = torch.min(identity_reprojection_losses, dim=1,
                                                              keepdim=True)  # b,1,h,w
            else:
                identity_reprojection_loss = None

            if self.cfg.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                # differently to Monodepth2, compute mins as we go
                reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)  # b,1,h,w

            if not self.cfg.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).to(self.device) * 0.00001

            # find minimum losses from [reprojection, identity]
            reprojection_loss_mask = self.compute_loss_masks(reprojection_loss,
                                                             identity_reprojection_loss)

            # find which pixels to apply reprojection loss to, and which pixels to apply
            # consistency loss to
            if is_multi:
                reprojection_loss_mask = torch.ones_like(reprojection_loss_mask)
                if not self.cfg.disable_motion_masking:
                    reprojection_loss_mask = (reprojection_loss_mask *
                                              outputs['consistency_mask'].unsqueeze(1))
                if not self.cfg.no_matching_augmentation:
                    reprojection_loss_mask = (reprojection_loss_mask *
                                              (1 - outputs['augmentation_mask']))
                consistency_mask = (1 - reprojection_loss_mask).float()

            # standard reprojection loss
            reprojection_loss = reprojection_loss * reprojection_loss_mask
            reprojection_loss = reprojection_loss.sum() / (reprojection_loss_mask.sum() + 1e-7)

            # consistency loss:
            # encourage multi frame prediction to be like singe frame where masking is happening
            if is_multi:
                multi_depth = outputs[("depth", 0, scale)]
                # no gradients for mono prediction!
                mono_depth = outputs[("mono_depth", 0, scale)].detach()
                consistency_loss = torch.abs(multi_depth - mono_depth) * consistency_mask
                consistency_loss = consistency_loss.mean()

                # save for logging to tensorboard
                consistency_target = (mono_depth.detach() * consistency_mask +
                                      multi_depth.detach() * (1 - consistency_mask))
                consistency_target = 1 / consistency_target
                outputs["consistency_target/{}".format(scale)] = consistency_target
                losses['consistency_loss/{}'.format(scale)] = consistency_loss
            else:
                consistency_loss = 0

            losses['reproj_loss/{}'.format(scale)] = reprojection_loss

            loss += (reprojection_loss + consistency_loss) / (2 ** scale)
            total_loss = total_loss + loss
            losses["loss/{}".format(scale)] = loss

        if not self.cfg.disable_triplet_loss:
            sgt_loss = self.compute_sgt_loss(inputs, outputs)
            losses['sgt_loss'] = sgt_loss
            total_loss = total_loss + sgt_loss * self.cfg.sgt
        losses["loss"] = total_loss

        return losses

    # If you want to port our redesigned triplet loss into your model to
    # achieve a superior result, simply add this function to the loss calculation.
    def compute_sgt_loss(self, inputs, outputs):
        seg_target = inputs[('seg', 0, 0)]
        N, _, H, W = seg_target.shape
        total_loss = 0

        for s, kernel_size in zip(self.cfg.sgt_scales, self.cfg.sgt_kernel_size):
            # s: [3, 2, 1]
            pad = kernel_size // 2
            h, w = self.cfg.height // 2 ** s, self.cfg.width // 2 ** s
            seg = F.interpolate(seg_target, size=(h, w), mode='nearest')
            seg_pad = F.pad(seg, pad=[pad] * 4, value=-1)
            patches = seg_pad.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
            aggregated_label = patches - seg.unsqueeze(-1).unsqueeze(-1)
            pos_idx = (aggregated_label == 0).float()
            neg_idx = (aggregated_label != 0).float()
            pos_num = pos_idx.sum(dim=(-1, -2))
            neg_num = neg_idx.sum(dim=(-1, -2))

            is_boundary = (pos_num >= kernel_size - 1) & (neg_num >= kernel_size - 1)

            feature = outputs[('d_feature', s)]
            affinity = self.compute_affinity(feature, kernel_size=kernel_size)
            neg_dist = neg_idx * affinity

            if not self.cfg.disable_hardest_neg:
                neg_dist[neg_dist == 0] = 1e3
                neg_dist_x, arg_min_x = torch.min(neg_dist, dim=-1)
                neg_dist, arg_min_y = torch.min(neg_dist_x, dim=-1)
                neg_dist = neg_dist[is_boundary]
            else:
                neg_dist = neg_dist.sum(dim=(-1, -2))[is_boundary] / \
                           neg_num[is_boundary]

            pos_dist = ((pos_idx * affinity).sum(dim=(-1, -2)) / pos_num)[is_boundary]

            zeros = torch.zeros(pos_dist.shape, device=self.device)
            if not self.cfg.disable_isolated_triplet:
                loss = pos_dist + torch.max(zeros, self.cfg.sgt_isolated_margin - neg_dist)
            else:
                loss = torch.max(zeros, self.cfg.sgt_margin + pos_dist - neg_dist)
            total_loss = total_loss + loss.mean() / (2 ** s)
        return total_loss

    @staticmethod
    def compute_affinity(feature, kernel_size):
        pad = kernel_size // 2
        feature = F.normalize(feature, dim=1)
        unfolded = F.pad(feature, [pad] * 4).unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
        feature = feature.unsqueeze(-1).unsqueeze(-1)
        similarity = (feature * unfolded).sum(dim=1, keepdim=True)
        # eps = torch.zeros(similarity.shape).to(similarity.device) + 1e-9
        affinity = torch.clamp(2 - 2 * similarity, min=1e-9).sqrt()
        return affinity

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.cfg.train.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                       " | loss: {:.5f} | time elapsed: {} | time left: {}"
        self.logger.info(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                             sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        for l, v in losses.items():
            self.writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.cfg.train.batch_size)):  # write a maxmimum of four images
            s = 0  # log only max scale
            for frame_id in self.cfg.frame_ids:
                self.writer.add_image(
                    "color_{}_{}/{}".format(frame_id, s, j),
                    inputs[("color", frame_id, s)][j].data, self.step)
                if s == 0 and frame_id != 0:
                    self.writer.add_image(
                        "color_pred_{}_{}/{}".format(frame_id, s, j),
                        outputs[("color", frame_id, s)][j].data, self.step)

            disp = colormap(outputs[("disp", s)][j, 0])
            self.writer.add_image(
                "disp_multi_{}/{}".format(s, j),
                disp, self.step)

            disp = colormap(outputs[('mono_disp', s)][j, 0])
            self.writer.add_image(
                "disp_mono/{}".format(j),
                disp, self.step)

            if outputs.get("lowest_cost") is not None:
                lowest_cost = outputs["lowest_cost"][j]

                consistency_mask = \
                    outputs['consistency_mask'][j].cpu().detach().unsqueeze(0).numpy()

                min_val = np.percentile(lowest_cost.numpy(), 10)
                max_val = np.percentile(lowest_cost.numpy(), 90)
                lowest_cost = torch.clamp(lowest_cost, min_val, max_val)
                lowest_cost = colormap(lowest_cost)

                self.writer.add_image(
                    "lowest_cost/{}".format(j),
                    lowest_cost, self.step)
                self.writer.add_image(
                    "lowest_cost_masked/{}".format(j),
                    lowest_cost * consistency_mask, self.step)
                self.writer.add_image(
                    "consistency_mask/{}".format(j),
                    consistency_mask, self.step)

                consistency_target = colormap(outputs["consistency_target/0"][j])
                self.writer.add_image(
                    "consistency_target/{}".format(j),
                    consistency_target[0], self.step)

    def save_opts(self):
        save_config(self.cfg, self.log_path / 'config.yaml')
        diff = find_config_diff(self.cfg)
        save_config(diff, self.log_path / 'config_min.yaml')

    def save_model(self):
        save_folder = self.log_path / "models" / f"weights_{self.epoch}"
        save_folder.mkdir(parents=True, exist_ok=True)

        for model_name, model in self.models.items():
            save_path = save_folder / f"{model_name}.pth"
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save estimates of depth bins
                to_save['min_depth_bin'] = self.min_depth_tracker
                to_save['max_depth_bin'] = self.max_depth_tracker
            torch.save(to_save, save_path)

    def load_model(self):
        """Load model from disk
        """
        load_weights_folder = Path(self.cfg.load_weights_folder)
        assert load_weights_folder.is_dir(), f"{load_weights_folder} does not exist."
        self.logger.info("loading model from {}".format(load_weights_folder))
        assert len(self.cfg.models_to_load) > 0, "have not specified any models to load"
        self.logger.info(f"loading models: {self.cfg.models_to_load}")

        for model in self.cfg.models_to_load:
            path = load_weights_folder / f"{model}.pth"
            model_dict = self.models[model].state_dict()
            pretrained_dict = torch.load(path)

            if model == 'encoder':
                self.min_depth_tracker = pretrained_dict.get('min_depth_bin')
                self.max_depth_tracker = pretrained_dict.get('max_depth_bin')
                self.logger.info(f'min depth：{self.min_depth_tracker}')
                self.logger.info(f'max_depth：{self.max_depth_tracker}')

                if self.min_depth_tracker is not None:
                    self.logger.info('setting depth bins!')
                    self.models['encoder'].compute_depth_bins(self.min_depth_tracker, self.max_depth_tracker)

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[model].load_state_dict(model_dict)

    def system_check(self):
        # checking height and width are multiples of 32
        assert self.cfg.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.cfg.width % 32 == 0, "'width' must be a multiple of 32"
        assert self.cfg.frame_ids[0] == 0, "frame_ids must start with 0"
        assert len(self.cfg.frame_ids) > 1, "frame_ids must have more than 1 frame specified"
        self.logger.info(f"Training model named: {self.cfg.exp_name}")
        self.logger.info(f"Saving results to: {self.cfg.output_dir}")
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Using split: {self.cfg.dataset.split}")

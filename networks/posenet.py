"""
PoseNetV2 + 'alpha' weights to balance position and sem features.


"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo
from torchvision import models

from utils.config import get_default_config
from utils.utils import vis_tensor


class CAFFE(nn.Module):
    """
    Confidence Aware Feature Flow Estimator.
    Input:
        - tgt_feat_list, ref_feat_list:
            [B, 256, H/4, W/4]
            [B, 512, H/8, W/8]
            [B, 1024, H/16, W/16]

    Return:
        - feat_flow_list:
            [B, H/4, W/4, 2]
            [B, H/8, W/8, 2]
            [B, H/16, W/16, 2]
        - confidence_list:
            [B, H/4, W/4, 1]
            [B, H/8, W/8, 1]
            [B, H/16, W/16, 1]


    """

    def __init__(self, cfg):
        super(CAFFE, self).__init__()
        self.device = torch.device(f"cuda:{cfg.device}")
        self.ws = [2 ** n + 1 for n in range(3, 0, -1)]  # window size list: [9, 5, 3]
        self.win_indices_list = self.gen_window_indices()

    def gen_window_indices(self):
        indices_list = []
        for ws in self.ws:
            indices = torch.meshgrid([torch.arange(ws), torch.arange(ws)], indexing='ij')
            indices = torch.stack(indices, dim=0).unsqueeze(0).to(torch.float32)  # [1, 2, ws, ws]
            indices = indices - ws // 2
            indices_list.append(indices.to(self.device).unsqueeze(0).view(1, 1, 2, -1))  # [1, 1, 2, ws*ws]
        return indices_list

    @staticmethod
    def calculate_feature_affinity(feat1, feat2, ws):
        """
        calculate the affinity between feature1 and feature2 in a window of given size.
        :param feat1: shape [b, c, h, w]
        :param feat2: shape [b, c, h, w]
        :param ws: window size
        :return feature affinity: shape [b, h*w, ws*ws]
        """

        b, c, h, w = feat1.size()
        feat1 = feat1.view(b, c, -1).permute(0, 2, 1).unsqueeze(-2)  # [b, h*w, 1, c]

        feat2_unfold = F.unfold(feat2, kernel_size=ws, stride=1, padding=ws // 2)  # [b, c*ws*ws, h*w]
        feat2_unfold = feat2_unfold.view(b, c, ws * ws, h * w)  # [b, c, ws*ws, h*w]
        feat2_unfold = feat2_unfold.permute(0, 3, 1, 2)  # [b, h*w, c, ws*ws]

        feats = torch.matmul(feat1, feat2_unfold)  # [b, h*w, 1, ws*ws]
        feats = feats.squeeze(-2)  # [b, h*w, ws*ws]
        return feats

    def forward(self, tgt_feat_list, ref_feat_list):
        feat_flow_list = []
        confidence_list = []
        for tgt_feat, ref_feat, ws, wi in zip(tgt_feat_list, ref_feat_list, self.ws, self.win_indices_list):
            b, c, h, w = tgt_feat.size()
            affinity = self.calculate_feature_affinity(tgt_feat, ref_feat, ws)  # [b, h*w, ws*ws]
            affinity_softmax = nn.functional.softmax(affinity, dim=-1)  # [b, h*w, ws*ws]
            feature_flow = torch.sum(affinity_softmax.unsqueeze(-2) * wi, dim=-1)
            feat_flow_list.append(feature_flow.view(b, h, w, 2))
            confidence = torch.max(affinity, dim=-1).values * torch.max(affinity_softmax, dim=-1).values
            confidence_list.append(confidence.view(b, h, w, 1))

        return feat_flow_list, confidence_list


class PCA(nn.Module):
    """
    Positional Clue Aggregator.
    Input:
        - feat_flow_list:
            [B, H/4, W/4, 2]
            [B, H/8, W/8, 2]
            [B, H/16, W/16, 2]
        - confidence_list:
            [B, H/4, W/4, 1]
            [B, H/8, W/8, 1]
            [B, H/16, W/16, 1]
        - point_cloud:
            [B, 3, H, W]

    Return:
        position_feat_list:
            [B, embed_dim, H/4, W/4]
            [B, embed_dim, H/8, W/8]
            [B, embed_dim, H/16, W/16]

    """

    def __init__(self, cfg):
        super(PCA, self).__init__()
        self.cfg = cfg
        self.device = torch.device(f"cuda:{cfg.device}")
        embed_dim = cfg.posenet.embed_dim
        self.pos_embed = nn.Sequential(
            nn.Conv2d(2, embed_dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(embed_dim),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        )
        self.pc_embed = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(embed_dim),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        )

        self.abs_pos_list = self.init_abs_positions()

    def init_abs_positions(self):
        h, w = self.cfg.height, self.cfg.width
        abs_position_list = []
        for n in range(2, 5):
            mesh = [torch.arange(h // 2 ** n, dtype=torch.float32), torch.arange(w // 2 ** n, dtype=torch.float32)]
            indices = torch.meshgrid(mesh, indexing='ij')
            abs_position = torch.stack(indices, dim=0).unsqueeze(0)  # [1, 2, h, w]
            abs_position = self.pos_norm(abs_position)
            abs_position_list.append(abs_position.to(self.device))
        return abs_position_list

    def pos_norm(self, pos):
        """
        normalize the position into range [-1, 1]
        :param pos: [b, 2, h, w]
        """
        b, _, h, w = pos.shape
        pos[:, 0, :, :] = pos[:, 0, :, :] / h
        pos[:, 1, :, :] = pos[:, 1, :, :] / w  # normalized into 0~1
        pos = (pos - 0.5) * 2  # normalized into -1~1
        return pos

    def pc_norm(self, point_cloud):
        pc = (point_cloud - point_cloud.min()) / (point_cloud.max() - point_cloud.min() + 1e-8)
        return pc

    def forward(self, feat_flow_list, confidence_list, point_cloud):
        position_feat_list = []
        for feat_flow, abs_pos, conf in zip(feat_flow_list, self.abs_pos_list, confidence_list):
            feat_flow = self.pc_norm(feat_flow.permute(0, 3, 1, 2))  # [b,2,h,w]
            conf = conf.permute(0, 3, 1, 2)  # [b,1,h,w]
            feat_flow_feat = self.pos_embed(feat_flow)
            abs_pos_feat = self.pos_embed(abs_pos)
            pc = F.interpolate(point_cloud, feat_flow.size()[2:], mode='bilinear', align_corners=True)
            pc_feat = self.pc_embed(self.pc_norm(pc))

            position_feat = conf * (feat_flow_feat + abs_pos_feat) + pc_feat
            position_feat_list.append(position_feat)
        return position_feat_list


class HPEI(nn.Module):
    """
    Hierarchical Positional Embedding Injector.
    Input:
        - sem_feat_list:
            [B, C, H/4, W/4]
            [B, 2C, H/8, W/8]
            [B, 4C, H/16, W/16]
            [B, 8C, H/32, W/32]
        - pos_feat_list:
            [B, embed_dim, H/4, W/4]
            [B, embed_dim, H/8, W/8]
            [B, embed_dim, H/16, W/16]

    Return:
        fused_feat:



    """

    def __init__(self, cfg):
        super(HPEI, self).__init__()
        self.device = torch.device(f"cuda:{cfg.device}")
        self.embed_dim = cfg.posenet.embed_dim
        self.chn_list = [64, 128, 256, 512]
        if int(cfg.posenet.backbone[6:]) > 34:  # num_layers
            self.num_ch_enc[1:] *= 4
        self.alpha = nn.Parameter(torch.zeros(4, device=self.device, dtype=torch.float32), requires_grad=True)
        self.chn_reduction = nn.ModuleList([nn.Conv2d(self.embed_dim, chn, kernel_size=1) for chn in self.chn_list])
        self.conv = nn.ModuleList([
            nn.Conv2d(chn_in, chn_out, kernel_size=3, padding=1)
            for chn_in, chn_out in zip(self.chn_list[:-1], self.chn_list[1:])
        ])

    def forward(self, sem_feat_list, pos_feat_list):
        last_fused_feat = torch.zeros_like(sem_feat_list[0])
        for i, (sem_feat, pos_feat) in enumerate(zip(sem_feat_list[:-1], pos_feat_list)):
            pos_feat = self.alpha[i] * self.chn_reduction[i](pos_feat)
            sem_feat = (1 - self.alpha[i]) * sem_feat
            fused_feat = self.conv[i](pos_feat + sem_feat + last_fused_feat)
            fused_feat = F.relu(fused_feat)
            fused_feat = F.interpolate(fused_feat, scale_factor=0.5, mode='bilinear')
            last_fused_feat = fused_feat

        fused_feat = sem_feat_list[-1] + fused_feat
        return fused_feat


class FrozenFeatureExtractor(nn.Module):
    """
    Extract features from a frozen resnet50.
    Input:
        - img: (B, 3, H, W)

    Return:
        - feature_list:
            [B, 256, H/4, W/4]
            [B, 512, H/8, W/8]
            [B, 1024, H/16, W/16]
            [B, 2048, H/32, W/32]
    """

    def __init__(self, cfg):
        super(FrozenFeatureExtractor, self).__init__()
        self.device = torch.device(f"cuda:{cfg.device}")
        extractor = models.resnet50(weights='DEFAULT').to(self.device)
        self.layer0 = nn.Sequential(extractor.conv1,
                                    extractor.bn1,
                                    extractor.relu,
                                    extractor.maxpool)
        self.layer1 = extractor.layer1
        self.layer2 = extractor.layer2
        self.layer3 = extractor.layer3
        self.layer4 = extractor.layer4
        self.layer1[-1].relu = nn.Identity()
        self.layer2[-1].relu = nn.Identity()
        self.layer3[-1].relu = nn.Identity()
        self.layer4[-1].relu = nn.Identity()

    @torch.no_grad()
    def forward(self, img):
        img = (img - 0.45) / 0.225
        feature_list = []
        feature = self.layer0(img)
        for i in range(1, 5):
            layer = getattr(self, 'layer{}'.format(i))
            feature = layer(feature)
            feature_list.append(F.normalize(feature))
            feature = F.relu(feature)
        return feature_list


class SemanticFeatureExtractor(nn.Module):
    """
    Extracts semantic features.
    Input:
        - imgs: 2 concatenated images [B, 6, H, W]

    Return:
        - feature_list:
            [B, C, H/4, W/4]
            [B, 2C, H/8, W/8]
            [B, 4C, H/16, W/16]
            [B, 8C, H/32, W/32]

    """

    def __init__(self, cfg):
        super(SemanticFeatureExtractor, self).__init__()
        self.device = torch.device(f"cuda:{cfg.device}")
        extractor = getattr(models, cfg.posenet.backbone)(weights='DEFAULT').to(self.device)
        self.layer0 = nn.Sequential(nn.Conv2d(6, 64, 7, 2, 3, bias=False),
                                    extractor.bn1,
                                    extractor.relu,
                                    extractor.maxpool)
        self.layer1 = extractor.layer1
        self.layer2 = extractor.layer2
        self.layer3 = extractor.layer3
        self.layer4 = extractor.layer4
        loaded = model_zoo.load_url(getattr(models.resnet, f'ResNet{18}_Weights').DEFAULT.url)
        self.layer0[0].weight.data = torch.cat([loaded['conv1.weight']] * 2, 1) / 2

    def forward(self, imgs):
        imgs = (imgs - 0.45) / 0.225
        feature_list = []
        feature = self.layer0(imgs)
        for i in range(1, 5):
            layer = getattr(self, 'layer{}'.format(i))
            feature = layer(feature)
            feature_list.append(feature)
        return feature_list


class PoseDecoder(nn.Module):
    def __init__(self, cfg):
        super(PoseDecoder, self).__init__()
        self.device = torch.device(f"cuda:{cfg.device}")
        last_chn = 512
        if cfg.posenet.backbone in ['resnet50', 'resnet101', 'resnet152']:
            last_chn *= 4

        self.conv_squeeze = nn.Conv2d(last_chn, 256, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
        )

    def forward(self, feature):
        b, c, h, w = feature.size()
        out = F.relu(self.conv_squeeze(feature))
        out = self.avg_pool(out).view(b, -1)
        out = self.mlp(out)

        out = 0.01 * out.view(-1, 1, 1, 6)
        angle = out[..., :3]
        translation = out[..., 3:]
        return angle, translation


class PoseNet(nn.Module):
    """
    Predict relative camera poses.

    Input:
        img1: input frame 1 with shape of [b,3,h,w]
        img2: input frame 2 with shape of [b,3,h,w]
        point_cloud: input depth map with shape of [b,1,h,w]

    Output:
        angle: [b,1,3]
        translation: [b,1,3]
    """

    def __init__(self, cfg):
        super(PoseNet, self).__init__()
        self.device = torch.device(f"cuda:{cfg.device}")

        self.frozen_feature_extractor = FrozenFeatureExtractor(cfg)
        self.sem_feature_extractor = SemanticFeatureExtractor(cfg)

        self.caffe = CAFFE(cfg)
        self.pca = PCA(cfg)
        self.hpei = HPEI(cfg)
        self.decoder = PoseDecoder(cfg)

    def forward(self, img_tgt, img_ref, point_cloud=None):
        if point_cloud is None:
            point_cloud = torch.zeros_like(img_tgt)
        sem_feat_list = self.sem_feature_extractor(torch.cat([img_tgt, img_ref], dim=1))
        feat_tgt_list = self.frozen_feature_extractor(img_tgt)
        feat_ref_list = self.frozen_feature_extractor(img_ref)

        feat_flow_list, confidence_list = self.caffe(feat_tgt_list[:-1], feat_ref_list[:-1])
        pos_embedding_list = self.pca(feat_flow_list, confidence_list, point_cloud)
        features = self.hpei(sem_feat_list, pos_embedding_list)
        pose = self.decoder(features)

        return pose


if __name__ == '__main__':
    config = get_default_config()
    posenet = PoseNet(config).to('cuda')
    img1 = torch.rand(10, 3, 192, 640).to('cuda')
    img2 = torch.rand(10, 3, 192, 640).to('cuda')
    pc = torch.rand(10, 3, 192, 640).to('cuda')
    axisangle, translation = posenet(img1, img2, pc)
    print(axisangle.shape)

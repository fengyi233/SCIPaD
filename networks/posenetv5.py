
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo
from torchvision import models

from utils.config import get_default_config
from utils.utils import vis_tensor

  
class FeatureReweightLayer(nn.Module):
    def __init__(self):
        super(FeatureReweightLayer, self).__init__()

        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class PositionEmbeddingLayer(nn.Module):
    def __init__(self, cfg):
        super(PositionEmbeddingLayer, self).__init__()
        self.device = torch.device(f"cuda:{cfg.device}")
        self.embed_dim = cfg.posenet.embed_dim
        h, w = cfg.height, cfg.width

        # position encoder
        self.pos_embed = nn.Sequential(nn.Conv2d(2, self.embed_dim, kernel_size=1),
                                       # nn.BatchNorm2d(embed_dim),
                                       nn.ReLU(),
                                       nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1),
                                       # nn.BatchNorm2d(embed_dim)
                                       )
        # absolute position generation
        self.abs_position_list = []
        for scale in range(3):
            n = scale + 2
            mesh = [torch.arange(h // 2 ** n), torch.arange(w // 2 ** n)]
            indices = torch.meshgrid(mesh, indexing='ij')
            abs_position = torch.stack(indices, dim=0).unsqueeze(0).to(torch.float32)  # [1, 2, h, w]
            abs_position = self.pos_normalization(abs_position)
            self.abs_position_list.append(abs_position.to(self.device))

    def pos_normalization(self, pos):
        """
        normalize the position into range [-1, 1]
        :param pos: [b, 2, h, w]
        """
        b, h, w, _ = pos.shape
        pos[:, 0, :, :] = pos[:, 0, :, :] / h
        pos[:, 1, :, :] = pos[:, 1, :, :] / w  # normalized into 0~1
        pos = (pos - 0.5) * 2  # normalized into -1~1
        return pos

    def forward(self, offsets, scale):
        """
        :param offsets: [b,2,h,w]
        """
        offsets = self.pos_normalization(offsets)
        rel_pos_embed = self.pos_embed(offsets)
        abs_pos = self.abs_position_list[scale]
        abs_pos = self.pos_normalization(abs_pos)
        abs_pos_embed = self.pos_embed(abs_pos)

        return rel_pos_embed + abs_pos_embed


# class PosTokenAggregation(nn.Module):
#     def __init__(self, embed_dim, kernel_size=3, stride=1, padding=1):
#         super(PosTokenAggregation, self).__init__()
#         self.depthwise = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, kernel_size,
#                                                  stride, padding, groups=embed_dim),
#                                        nn.BatchNorm2d(embed_dim),
#                                        nn.LeakyReLU(0.1, inplace=True))
#         self.pointwise = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, 1),
#                                        nn.BatchNorm2d(embed_dim),
#                                        nn.LeakyReLU(0.1, inplace=True))
#
#     def forward(self, pos_feature_list):
#         pos_feats = pos_feature_list[0] + pos_feature_list[1]
#         pos_feats = self.depthwise(pos_feats)
#         pos_feats = self.pointwise(pos_feats)
#         return pos_feats
#
#         # feats = pos_feature_list[-1]
#         # _, _, h, w = feats.shape
#         # for pos_feature in pos_feature_list[:-1]:
#         #     feats += F.interpolate(pos_feature, size=(h, w), mode='bilinear')
#         # feats = self.depthwise(feats)
#         # feats = self.pointwise(feats)
#         # return feats


class PoseEncoder(nn.Module):
    """
    Scale 0: img_res = (H/4, W/4),    ws = 9
    Scale 1: img_res = (H/8, W/8),    ws = 5
    Scale 2: img_res = (H/16, W/16),  ws = 3
    Scale 3: img_res = (H/32, W/32)

    """

    def __init__(self, cfg):
        super(PoseEncoder, self).__init__()
        self.device = torch.device(f"cuda:{cfg.device}")
        self.ws = [2 ** n + 1 for n in range(3, 0, -1)]  # window size list: [17, 9, 5, 3]
        self.embed_dim = cfg.posenet.embed_dim  # default: 96
        posenet_backbone = cfg.posenet.backbone
        backbone = getattr(models, posenet_backbone)(weights='DEFAULT')
        self.layer0 = nn.Sequential(nn.Conv2d(6, 64, 7, 2, 3, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        loaded = model_zoo.load_url(getattr(models.resnet, f'ResNet{18}_Weights').DEFAULT.url)
        self.layer0[0].weight.data = torch.cat([loaded['conv1.weight']] * 2, 1) / 2

        # ======================= position matching =================================
        extractor = models.resnet50(weights='DEFAULT')

        self.feature_extractor_layer0 = nn.Sequential(extractor.conv1,
                                                      extractor.bn1,
                                                      extractor.relu,
                                                      extractor.maxpool)
        self.feature_extractor_layer1 = extractor.layer1
        self.feature_extractor_layer2 = extractor.layer2
        self.feature_extractor_layer3 = extractor.layer3
        self.feature_extractor_layer4 = extractor.layer4
        self.sigmoid = nn.Sigmoid()

        self.reweight_layer = FeatureReweightLayer().to(self.device)
        self.pos_embed = PositionEmbeddingLayer(cfg).to(self.device)
        # self.pos_token_aggr = PosTokenAggregation(embed_dim=self.embed_dim).to(self.device)

        self.win_indices_list = self.gen_window_indices()

    def gen_window_indices(self):
        indices_list = []
        for ws in self.ws:
            indices = torch.meshgrid([torch.arange(ws), torch.arange(ws)], indexing='ij')
            indices = torch.stack(indices, dim=0).unsqueeze(0).to(torch.float32)  # [1, 2, ws, ws]
            indices = indices - ws // 2
            indices_list.append(indices.to(self.device))
        return indices_list

    def window_feature_affinity(self, feat1, feat2, ws):
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

    def soft_argmax(self, mat, ws):
        """
        Calculate the soft argmax value of the given matrix.
        :param mat: input matrix with shape of [b, h*w, ws*ws]
        :return: soft_argmax value with shape of [b, h*w, 1]
        """
        assert ws ** 2 == mat.shape[-1]  # check the window size
        b, c, _ = mat.shape
        soft_max = nn.functional.softmax(mat, dim=-1)
        # indices = torch.arange(start=0, end=ws ** 2).unsqueeze(0).to(self.device)
        indices = torch.meshgrid([torch.arange(ws), torch.arange(ws)], indexing='ij')
        indices = torch.stack(indices, dim=0).unsqueeze(0).to(torch.float32)  # [1, 2, ws, ws]

        soft_argmax = (soft_max * indices).sum(dim=-1, keepdim=True)

        return soft_argmax

    def soft_argmax_2d(self, mat, scale):
        """
        Calculate the soft argmax value of the given matrix.
        :param mat: input matrix with shape of [b, h*w, ws*ws]
        :return: soft_argmax value with shape of [b, h*w, 2], where '2' represents the max value position of the last
        two dimension.
        """
        ws = self.ws[scale]
        assert ws ** 2 == mat.shape[-1]  # check the window size
        # b, hw, _ = mat.shape
        soft_max = nn.functional.softmax(mat, dim=-1).unsqueeze(-2)  # [b, h*w, 1, ws*ws]
        indices = self.win_indices_list[scale].unsqueeze(0).view(1, 1, 2, -1)  # [1, 1, 2, ws*ws]
        # soft_argmax = torch.matmul(soft_max,
        #                            indices.transpose(-1, -2))  # [b, h*w, 1, ws*ws] X [1, 1, ws*ws, 2]
        soft_argmax = torch.sum(soft_max * indices, dim=-1)
        return soft_argmax  # [b, h*w, 2]

    @staticmethod
    def feature_normalization(feats):
        return F.normalize(feats, p=2, dim=1)

    def get_feature_list(self, img, freeze=False):
        """
        Get features from :
            1) concatenated images: train
            2) single image: freeze
        Args:
            img (tensor): [b,6,h,w] or [b,3,h,w]
            freeze (bool, optional): True for position feature extraction. False for semantic feature extraction.
        """
        img = (img - 0.45) / 0.225
        feature_list = []
        if freeze:
            with torch.no_grad():
                feature_list.append(self.feature_normalization(self.feature_extractor_layer0(img)))
                feature_list.append(self.feature_normalization(self.feature_extractor_layer1(feature_list[-1])))
                feature_list.append(self.feature_normalization(self.feature_extractor_layer2(feature_list[-1])))
                feature_list.append(self.feature_normalization(self.feature_extractor_layer3(feature_list[-1])))
                feature_list.append(self.feature_normalization(self.feature_extractor_layer4(feature_list[-1])))
        else:
            feature_list.append(self.layer0(img))
            feature_list.append(self.layer1(feature_list[-1]))
            feature_list.append(self.layer2(feature_list[-1]))
            feature_list.append(self.layer3(feature_list[-1]))
            feature_list.append(self.layer4(feature_list[-1]))
        return feature_list[0:1] + feature_list[2:]

    def compute_feature_offset(self, affinity, scale):
        soft_argmax = self.soft_argmax_2d(affinity, scale)  # [b, h*w, 2]
        offset = soft_argmax.permute(0, 2, 1)  # [b,2,h*w]
        return offset

    def forward(self, img1, img2, depths=None, mask=None):
        """
        Args:
            img1: input frame 1 with shape of [b,3,h,w]
            img2: input frame 2 with shape of [b,3,h,w]
            depths: input depth map with shape of [b,2,h,w]
            mask: input mask with shape of [b,1,h,w]
        Return:
            feature_list: list of rgb features with resolutions from hw/4 to hw/32
        """
        imgs = torch.cat([img1, img2], dim=1)
        # ====================== feature extraction =================================
        sem_feat_list = self.get_feature_list(imgs, freeze=False)
        feat1_list = self.get_feature_list(img1, freeze=True)
        feat2_list = self.get_feature_list(img2, freeze=True)

        # ====================== feature comparison =================================
        # ws = self.ws[1]
        # feat1, feat2 = feat1_list[0], feat2_list[0]
        # b, c, h, w = feat1.size()
        # affinity = self.window_feature_affinity(feat1, feat2, ws)
        # offsets = self.compute_feature_offset(affinity, ws).view(b, 2, h, w)
        # pos_token = self.pos_embed(offsets, scale=2)  # [b, embed_dim, h, w]
        # reweight_mask = self.reweight_layer(feat1)  # [b,1,h,w]
        # pos_token_list.append(reweight_mask * pos_token)
        #
        # feat1, feat2 = feat1_list[1], feat2_list[1]
        # b, c, h, w = feat1.size()
        # affinity = self.window_feature_affinity(feat1, feat2, ws)
        # offsets = self.compute_feature_offset(affinity, ws).view(b, 2, h, w)  # [b,2,h*w]
        # pos_token = self.pos_embed(offsets, scale=2)  # [b, embed_dim, h, w]
        # reweight_mask = self.reweight_layer(feat1)  # [b,1,h,w]
        # pos_token_list.append(reweight_mask * pos_token)
        # pos_feats = self.pos_token_aggr(pos_token_list)  # [b,embed_dim, h, w]

        # ======================= feature visualization =======================================
        # vis_tensor((img1[0] - 0.45) / 0.225)
        # vis_tensor((img2[0] - 0.45) / 0.225)
        # vis_tensor(feat1_list[0][0].sum(dim=0))
        # vis_tensor(feat2_list[0][0].sum(dim=0))
        # =====================================================================================

        pos_feat_list = []
        for scale, (feat1, feat2) in enumerate(zip(feat1_list, feat2_list)):
            if scale == 3:
                continue
            ws = self.ws[scale]
            b, c, h, w = feat1.size()
            # calculate feature similarity
            affinity = self.window_feature_affinity(feat1, feat2, ws)  # [b, h*w, ws*ws]

            # calculate offset
            offsets = self.compute_feature_offset(affinity, scale).view(b, 2, h, w)

            # ====================== position embedding =================================
            pos_token = self.pos_embed(offsets, scale)  # [b, embed_dim, h, w]

            # ====================== token reweighting =================================
            reweight_mask = self.reweight_layer(feat1)  # [b,1,h,w]
            pos_feat_list.append(reweight_mask * pos_token)
        # pos_feats = self.pos_token_aggr(pos_token_list)  # [b,embed_dim, h, w]

        # fused_feats = self.hfa(pos_feat_list, sem_feat_list)
        return pos_feat_list, sem_feat_list


class HierarchicalFeatureAggregation(nn.Module):
    def __init__(self, cfg):
        super(HierarchicalFeatureAggregation, self).__init__()
        self.device = torch.device(f"cuda:{cfg.device}")
        self.embed_dim = cfg.posenet.embed_dim
        self.num_sem_enc = [64, 128, 256, 512]

        self.alpha = nn.Parameter(torch.zeros(4), requires_grad=True)

        if int(cfg.posenet.backbone[6:]) > 34:  # num_layers
            self.num_ch_enc[1:] *= 4

        self.chn_reduction = nn.Conv2d(self.embed_dim, 1, kernel_size=1, stride=1, padding=0)

        self.conv = nn.ModuleList([
            nn.Conv2d(self.num_sem_enc[0], self.num_sem_enc[1], kernel_size=3, padding=1),
            nn.Conv2d(self.num_sem_enc[1], self.num_sem_enc[2], kernel_size=3, padding=1),
            nn.Conv2d(self.num_sem_enc[2], self.num_sem_enc[3], kernel_size=3, padding=1),
            nn.Conv2d(self.num_sem_enc[3], self.embed_dim, kernel_size=3, padding=1),
        ])

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, pos_feature_list, sem_feature_list):
        """
        :param pos_feature_list: list of positional encoded features, shape: (B, C, H/4 ~ H/16, W/4 ~ W/16), 3 items.
        :param sem_feature_list: list of semantic encoded features, shape: (B, C, H/4 ~ H/32, W/4 ~ W/32), 4 items.
        """
        last_fused_feat = torch.zeros_like(sem_feature_list[0])
        for i, (pos_feat, sem_feat) in enumerate(zip(pos_feature_list, sem_feature_list[:3])):
            fused_feat = self.conv[i](
                self.alpha[i] * self.chn_reduction(pos_feat) + (1 - self.alpha[i]) * sem_feat + last_fused_feat)
            fused_feat = self.relu(fused_feat)
            fused_feat = F.interpolate(fused_feat, scale_factor=0.5, mode='bilinear')
            last_fused_feat = fused_feat

        return sem_feature_list[3] + fused_feat


class PoseDecoder(nn.Module):
    def __init__(self, cfg):
        super(PoseDecoder, self).__init__()
        self.device = torch.device(f"cuda:{cfg.device}")
        enc_last_channel = 512
        if cfg.posenet.backbone in ['resnet50', 'resnet101', 'resnet152']:
            enc_last_channel *= 4

        self.hfa = HierarchicalFeatureAggregation(cfg)

        self.conv_squeeze = nn.Conv2d(enc_last_channel, 256, 1)

        self.conv_pose = [
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.Conv2d(256, 6, 1)
        ]
        self.conv_pose_list = nn.ModuleList(self.conv_pose)

        self.relu = nn.ReLU()

    def forward(self, pos_feats, sem_feats):
        out = self.hfa(pos_feats, sem_feats)
        out = self.relu(self.conv_squeeze(out))
        for i in range(3):
            out = self.conv_pose[i](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)  # [b,6,h,w] -> [b,6]

        out = 0.01 * out.view(-1, 1, 1, 6)

        angle = out[..., :3]
        translation = out[..., 3:]
        return angle, translation


class PoseNetV5(nn.Module):
    """
    Predict relative camera poses.

    Input:
        imgs, [b,6,h,w]
        depths, [b,2,h,w]
        mask, [b,1,h,w]

    Output:
        angle: [b,1,3]
        translation: [b,1,3]
    """

    def __init__(self, cfg):
        super(PoseNetV5, self).__init__()
        self.pose_enc = PoseEncoder(cfg)
        self.pose_dec = PoseDecoder(cfg)

    def forward(self, img1, img2, disps=None, mask=None):
        pos_feats, sem_feats = self.pose_enc(img1, img2, disps, mask)
        pose = self.pose_dec(pos_feats, sem_feats)
        return pose


if __name__ == '__main__':
    config = get_default_config()
    posenet = PoseNetV5(config).to('cuda')
    img1 = torch.rand(10, 3, 192, 640).to('cuda')
    img2 = torch.rand(10, 3, 192, 640).to('cuda')
    axisangle, translation = posenet(img1, img2)
    print(axisangle.shape)

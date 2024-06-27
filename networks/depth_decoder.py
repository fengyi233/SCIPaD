import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from utils.layers import ConvBlock, Conv3x3, upsample


class DepthDec(nn.Module):
    def __init__(self, cfg, num_ch_enc):
        super(DepthDec, self).__init__()
        self.cfg = cfg
        self.scales = cfg.scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], 1)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]

        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)

            x = [upsample(x)]
            if i > 0:
                x += [input_features[i - 1]]

            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

            self.outputs[('d_feature', i)] = x

            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs

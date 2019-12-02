import torch.nn as nn
import numpy as np
from ..registry import DECODER


class ASPP_V2(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, outplanes):
        super(ASPP_V2, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out

class DeeplabV2Decoder(nn.Module):
    def __init__(self, input_channels):
        super(DeeplabV2Decoder, self).__init__()
        self.aspp = ASPP_V2(input_channels[-1], [6, 12, 18, 24], [6, 12, 18, 24], 64)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.out_channels = [64]


    def forward(self, features):
        if len(features) != 5:
            raise ValueError(
                "Expected 5 features, got {} features instead.".format(
                    len(features)))
        x = self.aspp(features[-1])
        return self.upsample(x)


'''
when DeepLabV2Dedoder is used, encoder must be "R-DL-..." or "RX-DL-..."
'''
@DECODER.register("DeepLabV2Dedoder")
def build_Deeplabv2decoder(input_channels):
    return DeeplabV2Decoder(input_channels)
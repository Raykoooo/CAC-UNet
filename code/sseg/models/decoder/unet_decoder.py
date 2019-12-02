import torch.nn as nn
# import pdb
from ..modules.upsample import UnetUpsample
from ..registry import DECODER

class UnetDecoder(nn.Module):
    def __init__(self, out_layers, input_channels):
        super(UnetDecoder,self).__init__()
        self.out_layers = out_layers
        
        self.up1 = UnetUpsample(input_channels[-1], input_channels[-2])
        self.up2 = UnetUpsample(input_channels[-2], input_channels[-3])
        self.up3 = UnetUpsample(input_channels[-3], input_channels[-4])
        self.up4 = UnetUpsample(input_channels[-4], input_channels[-5])
        self.upsamples = [self.up1, self.up2, self.up3, self.up4]

        self.out_channels = [2048, 1024, 512, 256, 64]

    def forward(self, features):
        if len(features) != 5:
            raise ValueError(
                "Expected 5 features, got {} features instead.".format(
                    len(features)))

        outs = []
        decode_feature = features[4]
        outs.append(decode_feature)
        for i in range(4):
            out = self.upsamples[i](decode_feature, features[4-(i+1)])
            decode_feature = out
            outs.append(out)
        
        selected_outs = []
        for i, out in enumerate(outs):
            if i in self.out_layers:
                selected_outs.append(out)
        # pdb.set_trace()
        return selected_outs

@DECODER.register("UnetDecoder-D1-D5")
def build_unetdecoder_d1_d5(input_channels):
    out_layers = [0, 1, 2, 3, 4]
    return UnetDecoder(out_layers, input_channels)


@DECODER.register("UnetDecoder-D5")
def build_unetdecoder_d5(input_channels):
    out_layers = [4]
    return UnetDecoder(out_layers, input_channels)




    
        
        
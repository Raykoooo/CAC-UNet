import torch.nn as nn
import torch.nn.functional as F

from ..modules.upsample import UnetUpsample
from ..registry import DECODER


class FCN8sDecoder(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(input_channels[-1], input_channels[-2], kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(input_channels[-2])
        self.deconv2 = nn.ConvTranspose2d(input_channels[-2], input_channels[-3], kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(input_channels[-3])
        self.deconv3 = nn.ConvTranspose2d(input_channels[-3], input_channels[-4], kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(input_channels[-4])
        self.deconv4 = nn.ConvTranspose2d(input_channels[-4], input_channels[-5], kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(input_channels[-5])
        self.deconv5 = nn.ConvTranspose2d(input_channels[-5], 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(64)

        self.out_channels = [64]

    def forward(self, x):
        x5 = x[4]  # size=(x.H/32, x.W/32)
        x4 = x[3]  # size=(x.H/16, x.W/16)
        x3 = x[2]  # size=(x.H/8,  x.W/8)

        score = self.relu(self.deconv1(x5))               
        score = self.bn1(score + x4)                      
        score = self.relu(self.deconv2(score))            
        score = self.bn2(score + x3)                     
        score = self.bn3(self.relu(self.deconv3(score)))  
        score = self.bn4(self.relu(self.deconv4(score))) 
        score = self.bn5(self.relu(self.deconv5(score)))  

        return score 

        

@DECODER.register("FCNdecoder-8")
def build_fcn8(input_channels):
    return FCN8sDecoder(input_channels)





    
        
        
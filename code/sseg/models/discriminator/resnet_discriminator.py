import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import pdb
from ..ops.snconv import SNConv2d

from ..registry import DISCRIMINATOR

__all__ = ['build_Decoder_H']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return SNConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return SNConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, use_bn=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if not use_bn:
            self.use_bn = False
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out) if self.use_bn else out
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out) if self.use_bn else out

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, use_bn=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if not use_bn:
            self.use_bn = False
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out) if not self.with_ibn else self.ibnc(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)  if self.use_bn else out
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)  if self.use_bn else out

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, 
                channels,
                selected_layer,
                num_classes=1000, zero_init_residual=False,
                groups=1, width_per_group=64, replace_stride_with_dilation=None,
                norm_layer=None, with_ibn=False, 
                is_fcn=False,
                fcn_up_scale=2**6,
                use_bn=False,
                 ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if not use_bn:
            self.use_bn = False
        self._norm_layer = norm_layer

        self.channels = channels
        self.selected_layer = selected_layer
        self.is_multi_input = len(self.selected_layer)>1
        self.origin_planes = self.channels[self.selected_layer[0]]
        self.add_planes_list = [(channels[x] if x != -1 else 0) for x in selected_layer[1:]] if self.is_multi_input else [0]*5
        # self.origin_planes = sum([self.channels[i] for i in self.selected_layer])

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = SNConv2d(self.origin_planes, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], 
                                       add_planes=self.add_planes_list[0]
                                       )
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       add_planes=self.add_planes_list[1]
                                       )
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       add_planes=self.add_planes_list[2]
                                       )
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       add_planes=self.add_planes_list[3]
                                       )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.out_channels = [64] + [x * block.expansion for x in [64, 128, 256, 512]]
        self.is_fcn = is_fcn
        self.fcn_up_scale = fcn_up_scale
        self.up = nn.Sequential(
            conv1x1(block.expansion*512, num_classes),
            nn.Upsample(scale_factor=self.fcn_up_scale, mode='bilinear', align_corners=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, add_planes=0):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion or self.is_multi_input:
            downsample = nn.Sequential(
                conv1x1(self.inplanes + add_planes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes + add_planes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, features):
        if not isinstance(features, list) and not isinstance(features, tuple):
            features = [features]
        input = features[self.selected_layer[0]]
        x = self.conv1(input)
        x = self.bn1(x)  if self.use_bn else x
        x = self.relu(x)
        x = torch.cat((x, features[self.selected_layer[1]]), dim=1) if self.is_multi_input and self.selected_layer[1] != -1 else x
        x = self.maxpool(x)
        
        x = self.layer1(x)

        x = torch.cat((x, features[self.selected_layer[2]]), dim=1) if self.is_multi_input and self.selected_layer[2] != -1 else x
        x = self.layer2(x)

        x = torch.cat((x, features[self.selected_layer[3]]), dim=1) if self.is_multi_input and self.selected_layer[3] != -1 else x
        x = self.layer3(x)

        x = torch.cat((x, features[self.selected_layer[4]]), dim=1) if self.is_multi_input and self.selected_layer[4] != -1 else x
        x = self.layer4(x)

        if not self.is_fcn:
            x = self.avgpool(x)
            x = x.reshape(x.size(0), -1)
            x = self.fc(x)
        else:
            x = self.up(x)
            if self.fcn_up_scale == 2**5:
                # padding 
                diffY = input.size()[2] - x.size()[2]
                diffX = input.size()[3] - x.size()[3]
                # print([diffX, diffY])
                x = F.pad(x, (0, diffX, 0, diffY))
        return x


def _resnet(arch, block, layers, pretrained, progress, num_classes, 
            with_ibn,
            channels,
            selected_layer,
            is_fcn=False,
            fcn_up_scale=2**6,
            **kwargs):
    model = ResNet(block, layers, 
                    channels,
                    selected_layer,
                    num_classes=num_classes, 
                    with_ibn=with_ibn,
                    is_fcn=is_fcn,
                    fcn_up_scale=fcn_up_scale,
                    **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

@DISCRIMINATOR.register("Decoder-H")
def build_Decoder_H(channels, pretrained=False, progress=True,with_ibn=False,**kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, num_classes=1,
                    with_ibn=False, 
                    channels=channels,
                    selected_layer=[5],
                   **kwargs)


@DISCRIMINATOR.register("R18-Decoder-H")
def r18_build_Decoder_H(channels, pretrained=False, progress=True, with_ibn=False,**kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, num_classes=1,
                    with_ibn=False, 
                    channels=channels,
                    selected_layer=[-1],
                   **kwargs)



@DISCRIMINATOR.register("R18-Encoder-C5")
def r18_build_Encoder_H(channels, pretrained=False, progress=True, with_ibn=False,**kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, num_classes=1,
                    with_ibn=False, 
                    channels=channels,
                    selected_layer=[4],
                   **kwargs)

@DISCRIMINATOR.register("R18-Encoder-M")
def r18_build_Encoder_M(channels, pretrained=False, progress=True, with_ibn=False,**kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, num_classes=1,
                    with_ibn=False, 
                    channels=channels,
                    selected_layer=[0,1,2,3,4],
                   **kwargs)

@DISCRIMINATOR.register("R18-Decoder-M")
def r18_build_Decoder_Multi(channels, pretrained=False, progress=True, with_ibn=False,**kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, num_classes=1,
                    with_ibn=False, 
                    channels=channels,
                    selected_layer=[4,3,2,1,0],
                   **kwargs)

@DISCRIMINATOR.register("R18-Decoder-M-FCN")
def r18_build_Decoder_Multi_fcn(channels, pretrained=False, progress=True, with_ibn=False,**kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, num_classes=1,
                    with_ibn=False, 
                    channels=channels,
                    selected_layer=[4,3,2,1,0],
                    is_fcn=True,
                    fcn_up_scale=2**6,
                    **kwargs)

@DISCRIMINATOR.register("R18-Decoder-D5-FCN")
def r18_build_Decoder_D5_fcn(channels, pretrained=False, progress=True, with_ibn=False,**kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, num_classes=1,
                    with_ibn=False, 
                    channels=channels,
                    selected_layer=[4],
                    is_fcn=True,
                    fcn_up_scale=2**6,
                    **kwargs)

@DISCRIMINATOR.register("R18-Encoder-M-FCN")
def r18_build_Encoder_Multi_fcn(channels, pretrained=False, progress=True, with_ibn=False,**kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, num_classes=1,
                    with_ibn=False, 
                    channels=channels,
                    selected_layer=[0,1,2,3,4],
                    is_fcn=True,
                    fcn_up_scale=2**6,
                    **kwargs)

@DISCRIMINATOR.register("R18-Encoder-M-FCN-R18")
def r18_build_Decoder_Multi_fcn_R18(channels, pretrained=False, progress=True, with_ibn=False,**kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, num_classes=1,
                    with_ibn=False, 
                    channels=channels,
                    selected_layer=[0,1,2,3,4],
                    is_fcn=True,
                    fcn_up_scale=2**6,
                    **kwargs)

@DISCRIMINATOR.register("R18-Encoder-C3-C4-C5")
def r18_build_Encoder_C3_C4_C5(channels, pretrained=False, progress=True, with_ibn=False,**kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, num_classes=1,
                    with_ibn=False, 
                    channels=channels,
                    selected_layer=[2,3,4,-1,-1],
                    **kwargs)


@DISCRIMINATOR.register("R18-Encoder-C4-C5")
def r18_build_Encoder_C4_C5(channels, pretrained=False, progress=True, with_ibn=False,**kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, num_classes=1,
                    with_ibn=False, 
                    channels=channels,
                    selected_layer=[3,4,-1,-1,-1],
                    **kwargs)

@DISCRIMINATOR.register("R18-Encoder-C4-C5-FCN")
def r18_build_Encoder_C4_C5_FCN(channels, pretrained=False, progress=True, with_ibn=False,**kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, num_classes=1,
                    with_ibn=False, 
                    channels=channels,
                    selected_layer=[3,4,-1,-1,-1],
                    is_fcn=True,
                    fcn_up_scale=2**6,
                    **kwargs)

@DISCRIMINATOR.register("R18-Predictor")
def r18_build_Predictor(channels, pretrained=False, progress=True, with_ibn=False,**kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, num_classes=1,
                    with_ibn=False, 
                    channels=channels,
                    selected_layer=[0],
                    **kwargs)


@DISCRIMINATOR.register("R18-Predictor-FCN")
def r18_build_Predictor_FCN(channels, pretrained=False, progress=True, with_ibn=False,**kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, num_classes=1,
                    with_ibn=False, 
                    channels=channels,
                    selected_layer=[0],
                    is_fcn=True,
                    fcn_up_scale=2**5,
                    **kwargs)

@DISCRIMINATOR.register("R18-Semantic")
def r18_build_Semantic(channels, pretrained=False, progress=True, with_ibn=False,**kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, num_classes=1,
                    with_ibn=False, 
                    channels=channels,
                    selected_layer=[0],
                    **kwargs)

@DISCRIMINATOR.register("R18-Semantic-FCN")
def r18_build_Semantic_FCN(channels, pretrained=False, progress=True, with_ibn=False,**kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, num_classes=1,
                    with_ibn=False, 
                    channels=channels,
                    selected_layer=[0],
                    is_fcn=True,
                    fcn_up_scale=2**5,
                    **kwargs)
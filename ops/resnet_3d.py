import torch
import torchvision
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv1x3x3(in_planes, out_planes, stride=1, t_stride=1):
    """1x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1,3,3), stride=(t_stride, stride, stride),
                     padding=(0,1,1), bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=(1, stride, stride), bias=False)

def conv3x1x1(in_planes, out_planes, t_stride=1):
    "3x1x1 convolution"
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3,1,1), stride=(t_stride,1,1),
        padding=(1,0,0), bias=False)

def conv3x3x3(in_planes, out_planes, t_stride=1, stride=1):
    "3x3x3 convolution"
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=(t_stride,stride,stride),
        padding=(1,1,1), bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, 
        temporal_conv=False, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if temporal_conv:
            conv = conv3x3x3
        else:
            conv = conv1x3x3
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv(inplanes, planes, stride=stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, temporal_conv = False,
         norm_layer=None, t_stride=1):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        if temporal_conv:
            self.conv1 = conv3x1x1(inplanes, planes, t_stride=t_stride)
        else:
            self.conv1 = conv1x1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv1x3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, top_heavy=True, num_classes=1000, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.conv1 = nn.Conv3d(3, self.inplanes, kernel_size=(1,7,7), stride=(1,2,2), 
            padding=(0,3,3), bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        print("top heavy: {}".format(top_heavy))
        if top_heavy:
            self.temporal_conv = [False, False, True, True]
        else:
            self.temporal_conv = [True, True, True, True]
        self.layer1 = self._make_layer(block, 64, layers[0], temporal_conv=self.temporal_conv[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, temporal_conv=self.temporal_conv[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, temporal_conv=self.temporal_conv[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, temporal_conv=self.temporal_conv[3])
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, temporal_conv=False, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, 
            temporal_conv=temporal_conv, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            # temporal_conv = not temporal_conv
            layers.append(block(self.inplanes, planes, 
            temporal_conv=temporal_conv, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def part_state_dict(state_dict, model_dict):
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    pretrained_dict = inflate_state_dict(pretrained_dict, model_dict)
    model_dict.update(pretrained_dict)
    return model_dict


def inflate_state_dict(pretrained_dict, model_dict):
    for k in pretrained_dict.keys():
        if pretrained_dict[k].size() != model_dict[k].size():
            assert(pretrained_dict[k].size()[:2] == model_dict[k].size()[:2]), \
                   "To inflate, channel number should match."
            assert(pretrained_dict[k].size()[-2:] == model_dict[k].size()[-2:]), \
                   "To inflate, spatial kernel size should match."
            # print("Layer {} needs inflation.".format(k))
            shape = list(pretrained_dict[k].shape)
            shape.insert(2, 1)
            t_length = model_dict[k].shape[2]
            pretrained_dict[k] = pretrained_dict[k].reshape(shape)
            if t_length != 1:
                pretrained_dict[k] = pretrained_dict[k].expand_as(model_dict[k]) / t_length
            assert(pretrained_dict[k].size() == model_dict[k].size()), \
                   "After inflation, model shape should match."
    return pretrained_dict

def _resnet(arch, block, layers, pretrained, progress, top_heavy=True, **kwargs):
    model = ResNet(block, layers, top_heavy=top_heavy, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        new_state_dict = part_state_dict(state_dict, model.state_dict())                                              
        model.load_state_dict(new_state_dict)
    return model

def resnet50(pretrained=False, progress=True, **kwargs):
    # print('Add temporal conv every other block')
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2],  pretrained, progress,
                   top_heavy=True, **kwargs)                   

def resnet26(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet26', Bottleneck, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)                     
import torch
import torch.nn as nn
from torch import sigmoid

class AddSpatialMask(nn.Module):
    def __init__(self, net, in_channels):
        super(AddSpatialMask, self).__init__()
        self.net = net
        self.conv = nn.Conv3d(in_channels, 1, kernel_size=3, padding=1)

    def forward(self, x):
        weight = self.conv(x)
        weight -sigmoid(weight)
        x = weight * x
        x = self.net(x)
    
def make_spatial_dropout(net, insert_layer=['3.0','4.0']):
    print("Adding spatial dropout module in bock {}".format(insert_layer))
    stage_list = [net.conv1, net.laye1, net.layer2, net.layer3, net.layer4]
    for layer in insert_layer:
        stage_num, block_List = layer.split('.')
        net_stage = stage_list[int(stage_num)]
        for block_num in block_List:
            block = net_stage[int(block_num)]
            in_channels = block.conv1.in_channels
            block = AddSpatialMask(block, in_channels)

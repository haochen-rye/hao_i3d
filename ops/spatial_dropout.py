import torch
import torch.nn as nn
from torch import sigmoid, threshold

DROP = True
DROP_3D = True
TEMPORAL_DROP = False

class AddSpatialMask(nn.Module):
    def __init__(self, net, in_channels, thres=0.3, group=1, 
        random=False, use_relu=False):
        super(AddSpatialMask, self).__init__()
        self.net = net
        self.thres = thres
        self.group = group
        self.random = random
        self.use_relu = use_relu
        if DROP:
            if DROP_3D:
                self.drop = nn.Dropout3d(thres)
            else:
                self.drop = nn.Dropout2d(thres)
        else:
            if use_relu:
                self.conv = nn.Sequential(
                    nn.Conv3d(in_channels, self.group, kernel_size=3, padding=1),
                    nn.BatchNorm3d(self.group),
                    nn.ReLU6(inplace=True),
                )
            else:
                self.conv = nn.Conv3d(in_channels, self.group, kernel_size=3, padding=1)

    def forward(self, x):
        n,c,t,h,w = x.size()
        if DROP:
            if DROP_3D:
                if TEMPORAL_DROP:
                    x = x.permute(0,2,1,3,4)
                    x = self.drop(x)
                    x = x.permute(0,2,1,3,4)
                else:
                    x = self.drop(x)
            else:
                x = x.view(n,-1,h,w)
                x = self.drop(x)
        else:
            weight = self.conv(x)
            if not self.use_relu:
                weight = sigmoid(weight)
                weight = threshold(weight, self.thres, 0.)        
            if self.random:
                noise = torch.rand_like(weight)
                noise[:,:,::2] = 1.
                weight = noise * weight.clone()        
            x = weight[:,:,None] * x.view(n,self.group,-1,t,h,w)
        return self.net(x.view(n,c,t,h,w))
    
def make_spatial_dropout(net, sigmoid_layer=['3.0','4.0'], group=1, 
        thres=0.2, sigmoid_random=False):
    print("Adding spatial dropout module in bock {}".format(sigmoid_layer))
    print("Group for dropout module: {}".format(group))
    print("Adding noise for sigmoid value in spatial dropout module: {}".format(sigmoid_random))
    print("3d dropout: {}".format(DROP_3D))
    print("Temporal 3d dropout: {}".format(TEMPORAL_DROP))
    use_relu = (thres>=1)
    if DROP:
        print("using 2d_dropout for regulization: {}".format(thres))
    else:
        if use_relu:
            print("Using RELU6 to threshold")
        else:
            print('Threshold for sigmoid mask: {}'.format(thres))
    stage_list = [net.conv1, net.layer1, net.layer2, net.layer3, net.layer4]
    for layer in sigmoid_layer:
        stage_num, block_List = layer.split('.')
        net_stage = stage_list[int(stage_num)]
        for block_num in block_List:
            block = net_stage[int(block_num)]
            in_channels = block.conv1.in_channels
            net_stage[int(block_num)] = AddSpatialMask(block, in_channels, thres=thres, 
                group=group, random=sigmoid_random, use_relu=use_relu)
    

import torch
import torch.nn as nn
import torchvision
from math import floor
from torch.nn import ReplicationPad2d

class CorrelationFusion(nn.Module):
    def __init__(self, net, n_segment, t_out, n_group=1, fuse_dilation=False,
        fuse_spatial_dilation=1,  correlation_neighbor=3):
        super(CorrelationFusion, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.t_out = t_out
        self.n_group = n_group
        self.correlation_num = correlation_neighbor
        self.in_channels = correlation_neighbor * correlation_neighbor * (n_segment - 1)
        self.in_channels *= n_group 
        self.inter_channels = t_out * n_group * n_segment
        self.bn_out = nn.BatchNorm2d(self.in_channels)
        self.dilation = fuse_dilation
        self.fuse_spatial_dilation=fuse_spatial_dilation
        spatial_padding = fuse_spatial_dilation * 2
        print('=> Using n_group: {} in Temporal Fusion'.format(self.n_group))
        self.weight = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
            kernel_size=1, groups=n_group),
            # nn.BatchNorm2d(self.inter_channels)
        )

    def forward(self, x):
        x = self.net(x)
        nt, c, h, w = x.size()
        c_per_group = c // self.n_group
        n_batch = nt // self.n_segment
        neighbor_k = self.correlation_num
        boundary_pad = neighbor_k // 2
        m = ReplicationPad2d(boundary_pad)
        x_pad = m(x)
        x = x.view((n_batch, self.n_segment) + x.shape[-3:])
        x_pad = x_pad.view((n_batch, self.n_segment) + x_pad.shape[-3:])
        correlation_list = []
        for i in range(neighbor_k):
            for j in range(neighbor_k):
                correlation = x[:,:-1] * x_pad[:,1:,:, i:i+h,j:j+w]
                correlation = correlation.view(n_batch*(self.n_segment - 1), self.n_group, c_per_group, h, w)
                correlation_list.append(correlation.sum(dim=2))
        x_correlation = torch.stack(correlation_list).permute(1,2,0,3,4).contiguous().view(n_batch,
            self.n_segment-1, self.n_group, -1, h, w)
        x_correlation = x_correlation.permute(0,2,1,3,4,5).contiguous().view(n_batch, -1, h, w)
        wx = self.weight(x_correlation)
        wx = wx.view(n_batch, self.n_group, self.t_out, self.n_segment, h, w).permute(2,1,0,3,4,5)
        x = x.view(n_batch, self.n_segment, c_per_group, self.n_group, h, w).permute(2,3,0,1,4,5)
        x_list = []
        for i in range(wx.shape[0]):
            i_in = floor(i * self.n_segment / self.t_out)
            x_sum = (wx[i] * x).sum(dim=3).permute(2,1,0,3,4) + \
                x[:,:,:,i_in,:,:].permute(2,1,0,3,4)
            x_list.append(x_sum.contiguous().view(n_batch, c, h, w))
        x_out = torch.stack(x_list).permute(1,0,2,3,4)
        return x_out.contiguous().view(n_batch*self.t_out, c, h, w)

def make_correlation_fusion(net, n_segment, n_group=1, fuse_layer=['3.03','4.0'], fuse_dilation=False,
    fuse_spatial_dilation=1, fuse_downsample=False, correlation_neighbor=3):
    if isinstance(net, torchvision.models.ResNet) or isinstance(net, archs.small_resnet.ResNet):
        stage_list = [net.conv1, net.layer1, net.layer2, net.layer3, net.layer4]
        temporal_stride = 1
        for layer_str in fuse_layer:
            stage_num, block_list = layer_str.split('.')
            net_stage = stage_list[int(stage_num)]
            for block_num, block in enumerate(net_stage):
                if str(block_num) in block_list:
                    t_in = n_segment // temporal_stride
                    t_out = t_in // 2 if fuse_downsample else t_in
                    net_stage[block_num] = CorrelationFusion(block, t_in, t_out,
                        n_group=n_group, fuse_dilation=fuse_dilation,
                        fuse_spatial_dilation=fuse_spatial_dilation,
                        correlation_neighbor=correlation_neighbor)
                    if fuse_downsample:
                        temporal_stride *= 2
    else:
        raise NotImplementedError
import torch
import torch.nn as nn
import torchvision
from math import ceil
from torch.nn import ReplicationPad2d, ReplicationPad1d

class TemporalFusion(nn.Module):
    def __init__(self, net, n_segment, in_channels, n_group=1, fuse_dilation=False,
        fuse_spatial_dilation=1, fuse_correlation=False, fuse_ave=False,
        fuse_downsample=False, correlation_neighbor=3, GroupConv=False):
        super(TemporalFusion, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.n_group = n_group
        self.fuse_correlation = fuse_correlation
        self.correlation_num = correlation_neighbor
        if self.fuse_correlation:
            self.in_channels = correlation_neighbor * correlation_neighbor
            self.in_channels *= n_group if GroupConv else 1
            if fuse_downsample:
                self.inter_channels = 2 * self.n_group * correlation_neighbor * correlation_neighbor
            else: 
                self.in_channels *= 3
                self.inter_channels = 3 * self.n_group * correlation_neighbor * correlation_neighbor
        else:
            self.in_channels = in_channels
            self.inter_channels = 2 * self.n_group 
        self.bn_out = nn.BatchNorm2d(in_channels)
        self.dilation = fuse_dilation
        self.fuse_spatial_dilation=fuse_spatial_dilation
        self.fuse_ave = fuse_ave
        self.fuse_downsample = fuse_downsample
        self.GroupConv = GroupConv
        spatial_padding = fuse_spatial_dilation * 2
        print('=> Using n_group: {} in Temporal Fusion'.format(self.n_group))
        print('using group convolution: {}'.format(GroupConv))
        if self.dilation:
            print('Using dilation in Temporal fusion')
            temporal_dilation = 1
            max_dilation = 1 + n_segment // 2
            self.weight_list = []
            if temporal_dilation <= max_dilation:
                temporal_padding = (temporal_dilation -1) // 2
                self.weight_list = nn.Sequential(
                    nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels, 
                        kernel_size=(2,3,3), stride=(2,1,1), 
                        padding=(temporal_padding, spatial_padding,spatial_padding), 
                        padding_mode='circular', 
                        dilation=(temporal_dilation,fuse_spatial_dilation,fuse_spatial_dilation) ),
                    # nn.BatchNorm3d(self.inter_channels)
                )
                temporal_dilation += 2
        else:
            if fuse_correlation:
                if fuse_downsample:
                    self.weight = nn.Sequential(
                        nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        groups=n_group if GroupConv else 1, kernel_size=1),
                        # nn.BatchNorm2d(self.inter_channels)
                    )
                else:
                    self.weight = nn.Sequential(
                        nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels,
                        kernel_size=1,groups=n_group,)
                    )
            else:
                if fuse_downsample:
                    self.weight =  nn.Sequential(
                        nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels, 
                            kernel_size=(2,3,3), stride=(2,1,1), padding_mode='circular', 
                            groups=n_group if GroupConv else 1,
                            padding=(0,spatial_padding,spatial_padding),
                            dilation=(1,fuse_spatial_dilation, fuse_spatial_dilation)),
                        # nn.BatchNorm3d(self.inter_channels)
                        )  
                else:
                    self.weight = nn.Conv3d(in_channels=self.in_channels, out_channels=3*n_group,
                    kernel_size=3 , padding=2, padding_mode='circular',
                    groups=n_group if GroupConv else 1)

    def forward(self, x):
        x = self.net(x)
        nt, c, h, w = x.size()
        c_per_group = c // self.n_group
        n_batch = nt // self.n_segment
        if self.fuse_correlation:
            neighbor_k = self.correlation_num
            boundary_pad = neighbor_k // 2
            if self.fuse_downsample:                                                                                               
                m=ReplicationPad2d(boundary_pad) 
                x_pad=m(x) 
                correlation_list=[]                               
                for i in range(neighbor_k):                                                                                                                     
                    for j in range(neighbor_k):                                                                                                                 
                        correlation = x[::2]*x_pad[1::2,:,i:i+h,j:j+w]         
                        if self.GroupConv:                                                                                         
                            correlation = correlation.view(nt//2, self.n_group, c_per_group, h, w)                                                                                        
                            correlation = correlation.sum(dim=2)                                                                                                  
                        else:
                            correlation = correlation.sum(dim=1)
                        correlation_list.append(correlation) 
                if self.GroupConv:
                    x_correlation = torch.stack(correlation_list).permute(1,0,2,3,4).contiguous().view(nt//2,self.in_channels,h,w)
                else:
                    NotImplementedError
                x_correlation = self.weight(x_correlation)
                x_pad = x_pad.view((nt, c_per_group, self.n_group) + x_pad.shape[-2:]).permute(1,0,2,3,4)  
                x_correlation = x_correlation.view(nt//2, neighbor_k, neighbor_k, self.n_group*2, h, w)
                x_list = []
                for i in range(neighbor_k):                                                                                                                     
                    for j in range(neighbor_k):    
                        x_list.append(x_pad[:,::2,:,i:i+h, j:j+w]*x_correlation[:,i,j,::2] + \
                            x_pad[:,1::2,:,i:i+h, j:j+w]*x_correlation[:,i,j,1::2])
                x = torch.stack(x_list).sum(dim=0).permute(1,0,2,3,4).contiguous().view(nt//2, c, h, w)
                x = x/2 if self.fuse_ave else x
                return self.bn_out(x)
                # x = x.view(nt, c_per_group, self.n_group, h, w).permute(1,0,2,3,4)
                # x = x[:,::2]*x_correlation[:,::2] + x[:,1::2]*x_correlation[:,1::2]
                # x = x.permute(1,0,2,3,4).view(nt//2, c, h, w)
                # x = x/2 if self.fuse_ave else x
                # return self.bn_out(x)
            else:
                temporal_pad = 1
                m1 = ReplicationPad1d(temporal_pad)
                from pdb import set_trace;set_trace()
                from IPython import embed;embed()     
                x = x.view(n_batch, self.n_segment, c, h, w).permute(0, 2,3,4,1)
                x = x.contiguous().view(n_batch*c, -1, self.n_segment)
                x = m1(x)
                pad_t = x.shape[-1]
                x = x.view(n_batch*c,h,w,pad_t).permute(0,3,1,2)
                m2 = ReplicationPad2d(boundary_pad) 
                x_pad = m2(x)
                correlation_list=[]                               
                for i in range(neighbor_k):                                                                                                                     
                    for j in range(neighbor_k):             
                        correlation_past = x[:,1:1+self.n_segment]*x_pad[:,:self.n_segment,i:i+h,j:j+w]
                        correlation_now = x[:,1:1+self.n_segment]*x_pad[:,1:1+self.n_segment,i:i+h,j:j+w]
                        correlation_future = x[:,1:1+self.n_segment]*x_pad[:,2:2+self.n_segment,i:i+h,j:j+w]
                        if self.GroupConv:                                                                                         
                            correlation_past = correlation_past.view(nt, self.n_group, c_per_group, h, w)                                                                                        
                            correlation_past = correlation_past.sum(dim=2)                                                                                                  
                            correlation_now = correlation_now.view(nt, self.n_group, c_per_group, h, w)                                                                                        
                            correlation_now = correlation_now.sum(dim=2)                                                                                                  
                            correlation_future = correlation_future.view(nt, self.n_group, c_per_group, h, w)                                                                                        
                            correlation_future = correlation_future.sum(dim=2)                                                                                                  
                        else:
                            correlation = correlation.sum(dim=1)
                        correlation_list.append(correlation_past)
                        correlation_list.append(correlation_now)
                        correlation_list.append(correlation_future) 
                if self.GroupConv:
                    x_correlation = torch.stack(correlation_list).permute(1,0,2,3,4).contiguous().view(nt,self.in_channels,h,w)
                else:
                    NotImplementedError
                x_correlation = self.weight(x_correlation)
                x_pad = x_pad.view((n_batch, c_per_group, self.n_group) + x_pad.shape[-3:]) 
                x_correlation = x_correlation.view(n_batch, self.n_segment, neighbor_k, neighbor_k, self.n_group*3, h, w)
                x_list = []
                for i in range(neighbor_k):                                                                                                                     
                    for j in range(neighbor_k):    
                        x_list.append(x_pad[:,::2,:,i:i+h, j:j+w]*x_correlation[:,i,j,::2] + \
                            x_pad[:,1::2,:,i:i+h, j:j+w]*x_correlation[:,i,j,1::2])
                x = torch.stack(x_list).sum(dim=0).permute(1,0,2,3,4).contiguous().view(nt//2, c, h, w)
                x = x/2 if self.fuse_ave else x
                return self.bn_out(x)
                # x = x.view(nt, c_per_group, self.n_group, h, w).permute(1,0,2,3,4)
                # x = x[:,::2]*x_correlation[:,::2] + x[:,1::2]*x_correlation[:,1::2]
                # x = x.permute(1,0,2,3,4).view(nt//2, c, h, w)
                # x = x/2 if self.fuse_ave else x
                # return self.bn_out(x)
  
        else:
            x = x.view(n_batch, self.n_segment, c, h, w).permute(0,2,1,3,4) # x: [n, c, t, h, w ]
            if self.dilation:
                x_list = []
                for weight_layer in self.weight_list:
                    weight = weight_layer(x)
                    weight = weight.permute(1,0,2,3,4) # weight: [n, t//2, 2*n_group, h,w]
                    xx = x.reshape(n_batch,c_per_group,self.n_group,self.n_segment,h,w).permute(1,2,0,3,4,5) #xx: [c_per_group, n_group, n, t, h, w]
                    xx = xx[:,:,:,::2]*weight[::2]+xx[:,:,:,1::2]*weight[1::2]
                    x_list.append(xx.permute(2,3,0,1,4,5).contiguous().view(nt//2,c,h,w))
                x = torch.stack(x_list).sum(dim=0, keepdim=True)
                return x.squeeze(0)
            else:
                if self.fuse_downsample:
                    weight = self.weight(x).permute(1,0,2,3,4)  # weight: [2*n_group, n, t//2, h,w]
                    x =  x.reshape(n_batch,c_per_group,self.n_group,self.n_segment,h,w).permute(1,2,0,3,4,5) #xx: [c_per_group, n_group, n, t, h, w]
                    x = x[:,:,:,::2]*weight[::2] + x[:,:,:,1::2]*weight[1::2] 
                    x = x.permute(2,3,0,1,4,5).contiguous().view(nt//2, c, h, w)
                    x = x/2 if self.fuse_ave else x
                    return self.bn_out(x)
                else:
                    weight = self.weight(x)
                    x = x.permute(1,0,3,4,2).contiguous().view(c, -1,self.n_segment) 
                    m = ReplicationPad1d(1)
                    # from pdb import set_trace;set_trace()
                    # from IPython import embed;embed()
                    x = m(x).view(c_per_group, -1, self.n_segment+2)
                    weight = weight.permute(1,0,3,4,2).contiguous().view(3,-1,self.n_segment).permute(2,1,0)
                    x_list = []
                    for i in range(self.n_segment):
                        x_list.append((x[:,:,i:i+3] * weight[i]).sum(dim=2))
                    x = torch.stack(x_list).view(self.n_segment,c,n_batch,h*w).permute(2,0,1,3).contiguous().view(nt,c,h,w)
                    x = x/3 if self.fuse_ave else x
                    return self.bn_out(x)
        # else:
        #     x = x.view(n_batch, self.n_segment, c, h, w).permute(0,2,1,3,4) # x: [n, c, t, h, w ]
        #     weight = self.weight(x)
        #     x = x.permute(1,0,3,4,2).contiguous().view(c_per_group,-1,self.n_segment) 
        #     m = ReplicationPad1d(1)
        #     x = m(x)
        #     weight = weight.permute(1,0,3,4,2).contiguous().view(3,-1,self.n_segment).permute(2,1,0)
        #     x_list = []
        #     for i in range(self.n_segment):
        #         x_list.append((x[:,:,i:i+3] * weight[i]).sum(dim=2))
        #     x = torch.stack(x_list).view(self.n_segment,c,n_batch,h*w).permute(2,0,1,3).contiguous().view(nt,c,h,w)
        #     x = x/3 if self.fuse_ave else x
        #     return self.bn_out(x)



def make_temporal_fusion(net, n_segment, n_group=1, fuse_layer=['3.03','4.0'], fuse_dilation=False,
    fuse_spatial_dilation=1, fuse_correlation=False, fuse_ave=False, fuse_downsample=False,
    correlation_neighbor=3, GroupConv=False):
    if isinstance(net, torchvision.models.ResNet) or isinstance(net, archs.small_resnet.ResNet):
        stage_list = [net.conv1, net.layer1, net.layer2, net.layer3, net.layer4]
        temporal_stride = 1
        for layer_str in fuse_layer:
            stage_num, block_list = layer_str.split('.')
            net_stage = stage_list[int(stage_num)]
            for block_num, block in enumerate(net_stage):
                if str(block_num) in block_list:
                    net_stage[block_num] = TemporalFusion(block, n_segment // temporal_stride,
                        block.bn3.num_features, n_group=n_group, fuse_dilation=fuse_dilation,
                        fuse_spatial_dilation=fuse_spatial_dilation,
                        fuse_correlation=fuse_correlation, fuse_ave=fuse_ave, 
                        fuse_downsample=fuse_downsample, correlation_neighbor=correlation_neighbor, 
                        GroupConv=GroupConv)
                    if fuse_downsample:
                        temporal_stride *= 2
    else:
        raise NotImplementedError
